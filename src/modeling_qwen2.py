# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Qwen2 model."""
import inspect
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import warnings
from typing import List, Optional, Tuple, Union
try:
    from typing import Unpack
except ImportError:
    # Python < 3.11
    from typing_extensions import Unpack

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
try:
    from transformers.utils import TransformersKwargs
except ImportError:
    # 旧版本transformers没有TransformersKwargs，定义一个空的
    class TransformersKwargs:
        pass
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.masking_utils import create_causal_mask


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

from .configuration_qwen2 import Qwen2Config
from .modeling_lychee import LycheeMemory
from .modeling_utils import optional_grad_ctx, compute_loss, get_rope, ModelOutput, rotate_half


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "Qwen/Qwen2-7B-beta"
_CONFIG_FOR_DOC = "Qwen2Config"

QWEN2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Qwen/Qwen2-7B-beta",
    # See all Qwen2 models at https://huggingface.co/models?filter=qwen2
]


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    """与官方Qwen2实现一致的RoPE应用逻辑"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.mistral.modeling_mistral.Qwen2MLP with Qwen2->Qwen2
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = get_rope(self.head_dim, config.rope_theta, config.max_position_embeddings, getattr(config, "rope_scaling", None))

        # NOTE: add extra parameters for lychee_memory tokens
        # skip post initialization to speed up loading
        if "q" in config.lychee_memory_param:
            self.lychee_memory_q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.q_proj.bias is not None)
            # NOTE: initialize the lychee_memory parameters as zero
            self.lychee_memory_q_proj.weight.data.zero_()
            self.lychee_memory_q_proj._is_hf_initialized = True
        if "k" in config.lychee_memory_param:
            self.lychee_memory_k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.k_proj.bias is not None)
            self.lychee_memory_k_proj.weight.data.zero_()
            self.lychee_memory_k_proj._is_hf_initialized = True
        if "v" in config.lychee_memory_param:
            self.lychee_memory_v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.v_proj.bias is not None)
            self.lychee_memory_v_proj.weight.data.zero_()
            self.lychee_memory_v_proj._is_hf_initialized = True
        if "o" in config.lychee_memory_param:
            self.lychee_memory_o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.o_proj.bias is not None)
            self.lychee_memory_o_proj.weight.data.zero_()
            self.lychee_memory_o_proj._is_hf_initialized = True

    def _init_lychee_memory_proj(self, missing_keys):
        """Initialize the lychee_memory projection weight with that of the ordinal projection."""
        lychee_memory_param = self.config.lychee_memory_param
        
        if is_deepspeed_zero3_enabled():
            # FIXME: after deepspeed initialization, some weights becomes non-zero
            # For Mistral, there are rows that are full of zeros
            # For Mistral, there are values bigger than 1e29...

            import deepspeed
            if "q" in lychee_memory_param:
                params = [self.lychee_memory_q_proj.weight, self.q_proj.weight]
                if self.q_proj.bias is not None:
                    params.extend([self.lychee_memory_q_proj.bias, self.q_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
                    if (self.lychee_memory_q_proj.weight.sum(-1) == 0).any() or (self.lychee_memory_q_proj.weight > 1e29).any():
                        self.lychee_memory_q_proj.weight.data[:] = self.q_proj.weight.data
                        if self.q_proj.bias is not None:
                            self.lychee_memory_q_proj.bias.data[:] = self.q_proj.bias.data
            if "k" in lychee_memory_param:
                params = [self.lychee_memory_k_proj.weight, self.k_proj.weight]
                if self.k_proj.bias is not None:
                    params.extend([self.lychee_memory_k_proj.bias, self.k_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
                    if (self.lychee_memory_k_proj.weight.sum(-1) == 0).any() or (self.lychee_memory_k_proj.weight > 1e29).any():
                        self.lychee_memory_k_proj.weight.data[:] = self.k_proj.weight.data
                        if self.k_proj.bias is not None:
                            self.lychee_memory_k_proj.bias.data[:] = self.k_proj.bias.data
            if "v" in lychee_memory_param:
                params = [self.lychee_memory_v_proj.weight, self.v_proj.weight]
                if self.v_proj.bias is not None:
                    params.extend([self.lychee_memory_v_proj.bias, self.v_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
                    if (self.lychee_memory_v_proj.weight.sum(-1) == 0).any() or (self.lychee_memory_v_proj.weight > 1e29).any():
                        self.lychee_memory_v_proj.weight.data[:] = self.v_proj.weight.data
                        if self.v_proj.bias is not None:
                            self.lychee_memory_v_proj.bias.data[:] = self.v_proj.bias.data
            if "o" in lychee_memory_param:
                params = [self.lychee_memory_o_proj.weight, self.o_proj.weight]
                if self.o_proj.bias is not None:
                    params.extend([self.lychee_memory_o_proj.bias, self.o_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
                    if (self.lychee_memory_o_proj.weight.sum(-1) == 0).any() or (self.lychee_memory_o_proj.weight > 1e29).any():
                        self.lychee_memory_o_proj.weight.data[:] = self.o_proj.weight.data
                        if self.o_proj.bias is not None:
                            self.lychee_memory_o_proj.bias.data[:] = self.o_proj.bias.data
        else:
            # only copy the value in-place, without tieing the weight
            if "q" in lychee_memory_param and any("lychee_memory_q_proj" in missing_key for missing_key in missing_keys):
                # FIXME: some lychee_memory weights are not initialized as zero for mistral model, why? 
                # if (self.lychee_memory_q_proj.weight == 0).all():
                    self.lychee_memory_q_proj.weight.data[:] = self.q_proj.weight.data
                    if self.q_proj.bias is not None:
                        self.lychee_memory_q_proj.bias.data[:] = self.q_proj.bias.data
            if "k" in lychee_memory_param and any("lychee_memory_k_proj" in missing_key for missing_key in missing_keys):
                # if (self.lychee_memory_k_proj.weight == 0).all():
                    self.lychee_memory_k_proj.weight.data[:] = self.k_proj.weight.data
                    if self.k_proj.bias is not None:
                        self.lychee_memory_k_proj.bias.data[:] = self.k_proj.bias.data
            if "v" in lychee_memory_param and any("lychee_memory_v_proj" in missing_key for missing_key in missing_keys):
                # if (self.lychee_memory_v_proj.weight == 0).all():
                    self.lychee_memory_v_proj.weight.data[:] = self.v_proj.weight.data
                    if self.v_proj.bias is not None:
                        self.lychee_memory_v_proj.bias.data[:] = self.v_proj.bias.data
            if "o" in lychee_memory_param and any("lychee_memory_o_proj" in missing_key for missing_key in missing_keys):
                # if (self.lychee_memory_o_proj.weight == 0).all():
                    self.lychee_memory_o_proj.weight.data[:] = self.o_proj.weight.data
                    if self.o_proj.bias is not None:
                        self.lychee_memory_o_proj.bias.data[:] = self.o_proj.bias.data

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def qkv_proj_with_lychee_memory(self, hidden_states, lychee_memory_size, lychee_memory_indices):
        if lychee_memory_size > 0:
            # NOTE: when lychee_memory_pos == "interleave", the lychee_memory_indices points to all lychee_memory tokens in the current window (cached activations + input_ids), so we shall slice out the part corresponding to the input_ids
            cur_lychee_memory_indices = lychee_memory_indices[-hidden_states.shape[1]:]

            # NOTE: there is slight redundant computation because ordinal tokens should never be projected by lychee_memory matrices, but we are doing this for efficiency
            if "q" in self.config.lychee_memory_param:
                ordinal_query_states = self.q_proj(hidden_states)
                lychee_memory_query_states = self.lychee_memory_q_proj(hidden_states)
                query_states = torch.where((cur_lychee_memory_indices == 0)[:, None], ordinal_query_states, lychee_memory_query_states)
                if (cur_lychee_memory_indices == 2).any():
                    # lychee_memory_indices == 2 means the lychee_memory token is used to replicate the ones in previous window for parallel encoding
                    # we should slice out all lychee_memory tokens then copy them to the replicate lychee_memory tokens
                    query_states[:, cur_lychee_memory_indices == 2] = lychee_memory_query_states[:, cur_lychee_memory_indices == 1][:, :(cur_lychee_memory_indices == 2).sum()]
            else:
                query_states = self.q_proj(hidden_states)

            if "k" in self.config.lychee_memory_param:
                ordinal_key_states = self.k_proj(hidden_states)
                lychee_memory_key_states = self.lychee_memory_k_proj(hidden_states)
                key_states = torch.where((cur_lychee_memory_indices == 0)[:, None], ordinal_key_states, lychee_memory_key_states)
                if (cur_lychee_memory_indices == 2).any():
                    # lychee_memory_indices == 2 means the lychee_memory token is used to replicate the ones in previous window for parallel encoding
                    # we should slice out all lychee_memory tokens then copy them to the replicate lychee_memory tokens
                    key_states[:, cur_lychee_memory_indices == 2] = lychee_memory_key_states[:, cur_lychee_memory_indices == 1][:, :(cur_lychee_memory_indices == 2).sum()]
            else:
                key_states = self.k_proj(hidden_states)

            if "v" in self.config.lychee_memory_param:
                ordinal_value_states = self.v_proj(hidden_states)
                lychee_memory_value_states = self.lychee_memory_v_proj(hidden_states)
                value_states = torch.where((cur_lychee_memory_indices == 0)[:, None], ordinal_value_states, lychee_memory_value_states)
                if (cur_lychee_memory_indices == 2).any():
                    # lychee_memory_indices == 2 means the lychee_memory token is used to replicate the ones in previous window for parallel encoding
                    # we should slice out all lychee_memory tokens then copy them to the replicate lychee_memory tokens
                    value_states[:, cur_lychee_memory_indices == 2] = lychee_memory_value_states[:, cur_lychee_memory_indices == 1][:, :(cur_lychee_memory_indices == 2).sum()]
            else:
                value_states = self.v_proj(hidden_states)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        return query_states, key_states, value_states
    
    def o_proj_with_lychee_memory(self, attn_output, lychee_memory_size, lychee_memory_indices):
        if lychee_memory_size > 0:
            # NOTE: when lychee_memory_pos == "interleave", the lychee_memory_indices points to all lychee_memory tokens in the current window (cached activations + input_ids), so we shall slice out the part corresponding to the input_ids
            cur_lychee_memory_indices = lychee_memory_indices[-attn_output.shape[1]:]

            if "o" in self.config.lychee_memory_param:
                ordinal_attn_output = self.o_proj(attn_output)
                lychee_memory_attn_output = self.lychee_memory_o_proj(attn_output)
                attn_output = torch.where((cur_lychee_memory_indices == 0)[:, None], ordinal_attn_output, lychee_memory_attn_output)
            else:
                attn_output = self.o_proj(attn_output)
        else:
            attn_output = self.o_proj(attn_output)
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()
        kv_seq_len = hidden_states.shape[-2]
        past_key, past_value, lychee_memory_size, lychee_memory_indices = past_key_value

        if past_key is not None:
            past_seq_len = past_key.shape[2]
            kv_seq_len += past_seq_len
        else:
            past_seq_len = 0

        query_states, key_states, value_states = self.qkv_proj_with_lychee_memory(hidden_states, lychee_memory_size, lychee_memory_indices)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # return keys and values before rope
        # NOTE: incrementally return keys and values for efficiency 
        past_key_value = (key_states, value_states, lychee_memory_size, lychee_memory_indices)

        if past_key is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj_with_lychee_memory(attn_output, lychee_memory_size, lychee_memory_indices)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2SdpaAttention(Qwen2Attention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        bsz, q_len, _ = hidden_states.size()
        kv_seq_len = hidden_states.shape[-2]
        past_key, past_value, lychee_memory_size, lychee_memory_indices = past_key_value
        if past_key is not None:
            past_seq_len = past_key.shape[2]
            kv_seq_len += past_seq_len
        else:
            past_seq_len = 0

        query_states, key_states, value_states = self.qkv_proj_with_lychee_memory(hidden_states, lychee_memory_size, lychee_memory_indices)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # return keys and values before rope
        # NOTE: incrementally return keys and values for efficiency 
        past_key_value = (key_states, value_states, lychee_memory_size, lychee_memory_indices)

        if past_key is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj_with_lychee_memory(attn_output, lychee_memory_size, lychee_memory_indices)

        return attn_output, None, past_key_value


class Qwen2FlashAttention2(Qwen2Attention):
    """
    Qwen2 flash attention module. This module inherits from `Qwen2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()
        kv_seq_len = hidden_states.shape[-2]

        past_key, past_value, lychee_memory_size, lychee_memory_indices = past_key_value
        if past_key is not None:
            past_seq_len = past_key.shape[2]
            kv_seq_len += past_seq_len
        else:
            past_seq_len = 0

        query_states, key_states, value_states = self.qkv_proj_with_lychee_memory(hidden_states, lychee_memory_size, lychee_memory_indices)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # return keys and values before rope
        # NOTE: incrementally return keys and values for efficiency 
        past_key_value = (key_states, value_states, lychee_memory_size, lychee_memory_indices)

        if past_key is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

        # FlashAttention will automatically handle grouped query attention
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (Qwen2RMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, 
            key_states, 
            value_states, 
            attention_mask, 
            q_len, 
            dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj_with_lychee_memory(attn_output, lychee_memory_size, lychee_memory_indices)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in Qwen2FlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2Attention,
    "sdpa": Qwen2SdpaAttention,
    "flash_attention_2": Qwen2FlashAttention2,
}


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

QWEN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


QWEN2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # LYCHEE_MEMORY: add lychee memory embedding
        self.lychee_memory_embed_tokens = nn.Embedding(1, config.hidden_size, self.padding_idx)
        self.lychee_memory_embed_tokens._is_hf_initialized = True

        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _init_lychee_memory_embed(self, missing_keys):
        """Initialize the lychee_memory token embedding with that of the eos token."""
        if is_deepspeed_zero3_enabled():
            import deepspeed
            params = [self.lychee_memory_embed_tokens.weight, self.embed_tokens.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                # deepspeed will initialize the parameters to zero
                if (self.lychee_memory_embed_tokens.weight == 0).all():
                    if self.config.lychee_memory_embed_init == "bos":
                        self.lychee_memory_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[self.config.bos_token_id]
                    elif self.config.lychee_memory_embed_init == "eos":
                        if isinstance(self.config.eos_token_id, list):
                            eos_token_id = self.config.eos_token_id[0]
                        else:
                            eos_token_id = self.config.eos_token_id
                        self.lychee_memory_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[eos_token_id]
                    else:
                        raise NotImplementedError(f"Make sure lychee_memory_embed_init is either eos or bos, found {self.config.lychee_memory_embed_init}")
        else:
            if any("lychee_memory_embed_tokens" in missing_key for missing_key in missing_keys):
                if self.config.lychee_memory_embed_init == "bos":
                    self.lychee_memory_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[self.config.bos_token_id]
                elif self.config.lychee_memory_embed_init == "eos":
                    if isinstance(self.config.eos_token_id, list):
                        eos_token_id = self.config.eos_token_id[0]
                    else:
                        eos_token_id = self.config.eos_token_id
                    self.lychee_memory_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[eos_token_id]
                else:
                    raise NotImplementedError(f"Make sure lychee_memory_embed_init is either eos or bos, found {self.config.lychee_memory_embed_init}")

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # LYCHEE_MEMORY: always use cache
        use_cache = True

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key, past_value, lychee_memory_size, lychee_memory_indices = past_key_values[0]

        # LYCHEE_MEMORY: separately embed ordinal and lychee memory tokens because ordinal tokens do not receive gradients
        if lychee_memory_size > 0:
            # NOTE: when lychee_memory_pos == "interleave", the lychee_memory_indices points to all lychee_memory tokens in the current window (cached activations + input_ids), so we shall slice out the part corresponding to the input_ids
            cur_lychee_memory_indices = lychee_memory_indices[-input_ids.shape[1]:]

            ordinal_input_ids = input_ids[:, cur_lychee_memory_indices == 0]
            lychee_memory_input_ids = input_ids[:, cur_lychee_memory_indices > 0]
            ordinal_inputs_embeds = self.embed_tokens(ordinal_input_ids)
            lychee_memory_input_embeds = self.lychee_memory_embed_tokens(lychee_memory_input_ids - self.config.vocab_size)
            # create a new embedding tensor
            inputs_embeds = lychee_memory_input_embeds.new_zeros(*input_ids.shape, lychee_memory_input_embeds.shape[-1])
            inputs_embeds[:, cur_lychee_memory_indices == 0] = ordinal_inputs_embeds
            inputs_embeds[:, cur_lychee_memory_indices > 0] = lychee_memory_input_embeds

        else:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        # print(f"input_ids:          {input_ids}")
        # print(f"lychee_memory_indices:     {lychee_memory_indices}")
        # print(f"position_ids:       {position_ids}")
        # print(f"attention_mask:\n{attention_mask == 0}")
        # x = input()
        # if x == "s":
        #     return

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        # LYCHEE_MEMORY: still use tuple to organize cache
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # LYCHEE_MEMORY: slice out past_key_value of this layer
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.similarity_head = nn.Linear(config.hidden_size, 1, bias=True)
        self.memory: LycheeMemory = None
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Override the default from_pretrained to extend vocab size according to lychee_memory_size."""
        # Avoid meta tensor leftovers when custom keys are missing/mapped.
        kwargs.setdefault("low_cpu_mem_usage", False)
        kwargs.update(output_loading_info=True)
        model, loading_info = super().from_pretrained(*args, **kwargs)

        # NOTE: set memory after from_pretrained because there may be another transformer model inside the Memory object, which may cause weird erros during loading
        config = model.config
        model.memory = LycheeMemory(
            model_config=config,
            k_seq_dim=2,
            v_seq_dim=2,
        )

        missing_keys = loading_info["missing_keys"]
        # NOTE: the lychee_memory parameters may or may not be loaded from the checkpoint
        # if it is loaded from the checkpoint, we should not re-initilize it
        model.model._init_lychee_memory_embed(missing_keys)
        # initialize weights of possible q,k,v,o,mlp
        for layer in model.model.layers:
            layer.self_attn._init_lychee_memory_proj(missing_keys)

        return model

    def _native_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # when we directly call _native_forward, the past_key_values would be None
        if past_key_values is None:
            # NOTE: set lychee_memory size to 0 to avoid using any lychee_memory parameters, see Qwen2Attention.forward
            past_key_values = [(None, None, 0, None) for _ in range(self.config.num_hidden_layers)]

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        batch_loss = None
        token_loss = None
        
        if labels is not None:
            loss, batch_loss, token_loss = compute_loss(logits, labels, shift=False)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ModelOutput(
            loss=loss,
            batch_loss=batch_loss,
            token_loss=token_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def _lychee_memory_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        lychee_memory_skip_first: Optional[int] = None,
        lychee_memory_skip_last: Optional[int] = None,
        retain_kv = False,
    ):
        # t1 = time.time()
        # initialize cache
        self.memory.prepare(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels,
            skip_first=lychee_memory_skip_first,
            skip_last=lychee_memory_skip_last,
        )
        

        # t2 = time.time()

        while not self.memory.finish:

            # t3 = time.time()

            input_ids, attention_mask, position_ids, past_key_values, labels = self.memory.step()
            if self.memory.kv_mask is not None:
                attention_mask[:,:self.memory.kv_mask.shape[1]] = self.memory.kv_mask
            # 根据attention_mask动态构建position_ids，mask为1的位置自增，为0保持不变
            if attention_mask is not None:
                # 假设attention_mask形状为(batch, seq_len)
                position_ids = torch.cumsum(attention_mask, dim=1) - 1
                position_ids = position_ids.clamp(min=0).to(torch.long)

            # 每隔8个token把下一个token的position_ids设为1
            # if position_ids is not None and position_ids.dim() == 2:
            #     # 假设batch维在0，seq_len在1
            #     seq_len = position_ids.shape[1]
            #     interval = 8
            #     for i in range(interval, seq_len, interval + 1):
            #         if i < seq_len:
            #             position_ids[:, i] = 1
            # position_ids = torch.ones_like(position_ids)
            
            # t4 = time.time()
            outputs = self._native_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )

            # t5 = time.time()

            # update past_key_values
            self.memory.update_memory(outputs.past_key_values)

            # t6 = time.time()

            if labels is not None:
                # update loss
                self.memory.update_loss(outputs.batch_loss, (labels != -100).sum(-1))

            # t7 = time.time()

            # print(f"step time: {t4-t3}, forward time: {t5-t4}, update time: {t6-t5}, loss time: {t7-t6}")
            # input()

        # t8 = time.time()
        if retain_kv:
            self.memory.retain_kv.append(self.memory.lychee_memory_activations)
        # output loss, past_key_values, and perplexity
        outputs = self.memory.output(outputs)

        # t9 = time.time()

        # print(f"output time:            {t9-t8}")
        # input()

        return outputs

    
    def generate_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        完全模仿原版Qwen2ForCausalLM.forward的实现。
        不使用lychee_memory机制，直接使用标准的DynamicCache。
        
        压缩过的KV cache通过past_key_values参数传入。
        """
        from transformers.cache_utils import DynamicCache, Cache
        
        # 从kwargs中提取output相关参数（向后兼容）
        output_attentions = kwargs.pop("output_attentions", None)
        output_hidden_states = kwargs.pop("output_hidden_states", None)
        return_dict = kwargs.pop("return_dict", True)
        
        gradient_checkpointing = False

        if use_cache is None:
            use_cache = self.config.use_cache

        if gradient_checkpointing and use_cache:
            if not getattr(self, "_gradient_checkpointing_cache_warning", False):
                logger.warning(
                    "梯度检查点已启用，强制关闭use_cache以避免与checkpoint冲突。"
                )
                self._gradient_checkpointing_cache_warning = True
            use_cache = False

        # 1. 规范化past_key_values为DynamicCache格式（如果是lychee_memory/legacy格式）
        past_key_values = self._normalize_to_dynamic_cache(past_key_values)
        
        # 2. 调用标准的Qwen2Model.forward（不使用lychee_memory）
        # 这里完全模仿原版Qwen2的实现
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        
        # 创建DynamicCache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        
        # 计算past_seen_tokens（无论cache_position是否提供，都需要这个值）
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        
        # 准备cache_position（原版逻辑）
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, 
                past_seen_tokens + inputs_embeds.shape[1], 
                device=inputs_embeds.device
            )
        
        # 准备position_ids
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        batch_size, seq_len = inputs_embeds.shape[:2]

        # 处理attention_mask（确保包含past tokens）
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, past_seen_tokens + seq_len), dtype=torch.long, device=inputs_embeds.device
            )
        elif past_seen_tokens > 0 and attention_mask.shape[1] == seq_len:
            past_mask = torch.ones(
                (batch_size, past_seen_tokens), dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat([past_mask, attention_mask], dim=1)

        # 仅在需要时构造mask，避免每步decode都分配大4D张量
        use_flash_attn_fastpath = (
            is_flash_attn_2_available()
            and getattr(self.model.config, "_attn_implementation", "") == "flash_attention_2"
            and not output_attentions
        )
        layer_attention_mask = None
        fallback_causal_mask = None
        if isinstance(attention_mask, dict):
            layer_attention_mask = attention_mask.get("full_attention")
        elif use_flash_attn_fastpath:
            # Decode fast path: single-token step without padding does not need explicit mask.
            if attention_mask is None and seq_len == 1 and past_seen_tokens > 0:
                layer_attention_mask = None
            else:
                mask_kwargs = {
                    "config": self.model.config,
                    "input_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values,
                }
                layer_attention_mask = create_causal_mask(**mask_kwargs)
        else:
            total_len = past_seen_tokens + seq_len
            min_dtype = torch.finfo(inputs_embeds.dtype).min
            fallback_causal_mask = torch.full(
                (seq_len, total_len), min_dtype, device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

            if past_seen_tokens > 0:
                fallback_causal_mask[:, :past_seen_tokens] = 0

            if seq_len > 1:
                mask_cond = torch.arange(seq_len, device=inputs_embeds.device)
                fallback_causal_mask[:, past_seen_tokens:] = torch.where(
                    mask_cond[:, None] >= mask_cond[None, :],
                    0.0,
                    min_dtype,
                )
            else:
                fallback_causal_mask[:, past_seen_tokens:] = 0

            fallback_causal_mask = fallback_causal_mask[None, None, :, :].expand(batch_size, 1, seq_len, total_len)
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    padding_mask = ~attention_mask
                else:
                    padding_mask = attention_mask == 0
                fallback_causal_mask = fallback_causal_mask.masked_fill(padding_mask[:, None, None, :], min_dtype)
        
        rotary_position_ids = position_ids
        if rotary_position_ids.dim() == 1:
            rotary_position_ids = rotary_position_ids.unsqueeze(0)
        if rotary_position_ids.shape[0] != batch_size:
            rotary_position_ids = rotary_position_ids.expand(batch_size, -1)
        rotary_position_ids = rotary_position_ids.to(device=inputs_embeds.device, dtype=torch.long)

        max_position = int(rotary_position_ids.max().item()) + 1
        last_cached = getattr(self, "_generate_rope_cache_len", 0)
        if max_position > last_cached:
            for layer in self.model.layers:
                rotary_module = layer.self_attn.rotary_emb
                cached_len = getattr(rotary_module, "max_seq_len_cached", None)
                if cached_len is not None and max_position > cached_len:
                    rotary_module._set_cos_sin_cache(
                        seq_len=max_position,
                        device=inputs_embeds.device,
                        dtype=inputs_embeds.dtype,
                    )
            self._generate_rope_cache_len = max_position

        # 与原版靠拢：position embeddings 在进入层循环前一次性取好
        shared_rotary = self.model.layers[0].self_attn.rotary_emb
        shared_cos = shared_rotary.cos_cached[rotary_position_ids].to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        shared_sin = shared_rotary.sin_cached[rotary_position_ids].to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )

        # 3. 开始通过所有decoder layers（使用FlashAttention2优化）
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        for idx, decoder_layer in enumerate(self.model.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            attn_dropout_prob = decoder_layer.self_attn.attention_dropout if self.training else 0.0
            residual_sa = hidden_states
            normed = decoder_layer.input_layernorm(hidden_states)
            bsz_, q_len_, _ = normed.size()

            query_states_ = decoder_layer.self_attn.q_proj(normed)
            key_states_ = decoder_layer.self_attn.k_proj(normed)
            value_states_ = decoder_layer.self_attn.v_proj(normed)

            query_states_ = query_states_.view(
                bsz_, q_len_, decoder_layer.self_attn.num_heads, decoder_layer.self_attn.head_dim
            ).transpose(1, 2)
            key_states_ = key_states_.view(
                bsz_, q_len_, decoder_layer.self_attn.num_key_value_heads, decoder_layer.self_attn.head_dim
            ).transpose(1, 2)
            value_states_ = value_states_.view(
                bsz_, q_len_, decoder_layer.self_attn.num_key_value_heads, decoder_layer.self_attn.head_dim
            ).transpose(1, 2)

            cos = shared_cos[:, -q_len_:].to(device=normed.device, dtype=normed.dtype)
            sin = shared_sin[:, -q_len_:].to(device=normed.device, dtype=normed.dtype)
            query_states_, key_states_ = apply_rotary_pos_emb(query_states_, key_states_, cos, sin)

            if past_key_values is not None:
                cache_kwargs = {"cache_position": cache_position}
                key_states_, value_states_ = past_key_values.update(
                    key_states_, value_states_, idx, cache_kwargs
                )

            if use_flash_attn_fastpath:
                attn_output_, _ = flash_attention_forward(
                    decoder_layer.self_attn,
                    query_states_,
                    key_states_,
                    value_states_,
                    layer_attention_mask,
                    dropout=attn_dropout_prob,
                    scaling=getattr(decoder_layer.self_attn, "scaling", None),
                    sliding_window=getattr(decoder_layer.self_attn, "sliding_window", None),
                )
                attn_output_ = attn_output_.reshape(bsz_, q_len_, -1).contiguous()
                attn_weights = None
            else:
                if decoder_layer.self_attn.num_key_value_groups != 1:
                    key_states_ = repeat_kv(key_states_, decoder_layer.self_attn.num_key_value_groups)
                    value_states_ = repeat_kv(value_states_, decoder_layer.self_attn.num_key_value_groups)

                attn_weights = torch.matmul(
                    query_states_, key_states_.transpose(2, 3)
                ) / math.sqrt(decoder_layer.self_attn.head_dim)
                attn_weights = attn_weights + fallback_causal_mask.to(query_states_.dtype)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states_.dtype)
                attn_weights = nn.functional.dropout(
                    attn_weights,
                    p=decoder_layer.self_attn.attention_dropout,
                    training=self.training,
                )
                attn_output_ = torch.matmul(attn_weights, value_states_)
                attn_output_ = attn_output_.transpose(1, 2).contiguous()
                attn_output_ = attn_output_.reshape(bsz_, q_len_, -1)

            attn_output_ = decoder_layer.self_attn.o_proj(attn_output_)
            hidden_states = residual_sa + attn_output_

            residual_mlp = hidden_states
            normed_mlp = decoder_layer.post_attention_layernorm(hidden_states)
            mlp_out = decoder_layer.mlp(normed_mlp)
            hidden_states = residual_mlp + mlp_out
            
            if output_attentions:
                all_self_attns += (attn_weights,)
        
        # 5. Final norm
        hidden_states = self.model.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # 6. 计算logits（原版逻辑：支持logits_to_keep）
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        # 7. 计算loss（如果有labels）
        loss = None
        if labels is not None:
            # 使用原版的loss_function（如果有自定义的compute_loss，保留兼容性）
            if hasattr(self, 'loss_function'):
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            else:
                # 标准的Hugging Face loss计算
                from transformers.modeling_utils import PreTrainedModel
                loss = PreTrainedModel.loss_function(self, logits=logits, labels=labels, vocab_size=self.vocab_size)
        
        # 8. 返回结果（完全匹配原版Qwen2）
        if not return_dict:
            output = (logits,) + (past_key_values if use_cache else None,)
            if output_hidden_states:
                output += (all_hidden_states,)
            if output_attentions:
                output += (all_self_attns,)
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    def clear_retain_kv(self):
        self.memory.retain_kv = []
        self.memory.kv_mask = None

    @staticmethod
    def _normalize_to_dynamic_cache(past_key_values):
        from transformers.cache_utils import DynamicCache

        if past_key_values is None or isinstance(past_key_values, DynamicCache):
            return past_key_values

        legacy_cache = []
        for layer_past in past_key_values:
            if isinstance(layer_past, tuple) and len(layer_past) == 4:
                legacy_cache.append((layer_past[0], layer_past[1]))
            elif isinstance(layer_past, tuple) and len(layer_past) == 2:
                legacy_cache.append(layer_past)
            else:
                legacy_cache.append(layer_past)
        return DynamicCache.from_legacy_cache(legacy_cache)
        
    def aggregate_text(self, text_ids, text_attention_mask, stream_chunk_size: int = 0):
        """
        生成压缩后的KV cache用于memory
        
        关键修复：
        1. self.training影响Memory._step的行为（L627）：
           - training=True: lychee_memory_size=0，不生成lychee_memory_activations
           - training=False: 正常压缩，生成lychee_memory_activations
        
        2. self.training影响optional_grad_ctx（L1465）：
           - training=True: 保留梯度，实现端到端训练
           - training=False: torch.no_grad()，梯度断裂
        
        解决方案：
        - 临时设置training=False（让Memory生成lychee_memory）
        - 但不通过self.eval()（避免触发optional_grad_ctx的no_grad）
        - 而是直接修改self.memory.training标志
        """
        self.clear_retain_kv()

        from transformers.cache_utils import DynamicCache

        # 🔧 关键修复：临时设置memory.training=False以触发压缩逻辑
        # 但保持模型自身的training状态不变，从而保留梯度
        original_memory_training = self.memory.training
        self.memory.training = False  # 让Memory._step执行压缩逻辑

        try:
            self.memory.reset()
            if text_attention_mask is None:
                text_attention_mask = torch.ones_like(text_ids)

            if stream_chunk_size > 0 and text_ids.shape[1] > stream_chunk_size:
                start = 0
                while start < text_ids.shape[1]:
                    end = min(start + stream_chunk_size, text_ids.shape[1])
                    self.forward(
                        input_ids=text_ids[:, start:end],
                        attention_mask=text_attention_mask[:, start:end],
                        retain_kv=True,
                    )
                    start = end
                kv_cache = self.memory.export_lychee_memory_cache()
            else:
                self.forward(
                    input_ids=text_ids,
                    attention_mask=text_attention_mask,
                    retain_kv=True,
                )
                if len(self.memory.retain_kv) > 0:
                    kv_cache = DynamicCache.from_legacy_cache(self.memory.retain_kv[-1])
                else:
                    kv_cache = self.memory.export_lychee_memory_cache()

            kv_mask = self.memory.build_kv_mask_from_attention(text_attention_mask).to(text_ids.device)
        finally:
            # 🔧 恢复原始状态
            self.memory.training = original_memory_training
            self.memory.reset()
            self.clear_retain_kv()
        
        return kv_cache, kv_mask
    
    
    def set_past_doc(self, doc_ids:list):
        self.memory.reset()
        # 🔧 优化：使用列表推导式，减少临时变量分配
        self.memory.lychee_memory_activations = [
            (
                torch.cat([self.memory.retain_kv[doc_id][layer_idx][0] for doc_id in doc_ids], dim=-2),
                torch.cat([self.memory.retain_kv[doc_id][layer_idx][1] for doc_id in doc_ids], dim=-2)
            )
            for layer_idx in range(self.config.num_hidden_layers)
        ]
            
    
    def forward(self, **kwargs):
        """Forward computation over a batch of sequences.
        """
        # only allow gradient when training
        if "aggregate_text" in kwargs:
            # 只保留 aggregate_text 需要的键值对
            keys_to_keep = ["text_ids", "text_attention_mask", "stream_chunk_size"]
            kwargs = {k: v for k, v in kwargs.items() if k in keys_to_keep}
            return self.aggregate_text(**kwargs)
        with optional_grad_ctx(with_grad=self.training):
            # fallback to native Qwen2 path when LycheeMemory is disabled
            if hasattr(self, "_enable_lychee_memory") and self._enable_lychee_memory == False:
                return self._native_forward(**kwargs)
            elif self.memory.use_lychee_memory:
                return self._lychee_memory_forward(**kwargs)
            else:
                return self.generate_forward(**kwargs)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        max_new_tokens: int = 128,
        memory_size: int = 1024,
        do_sample: bool = False,
        top_p: float = 1.0,
        top_k: int = 50,
        temperature: float = 1.0,
        eos_token_id=None,
        pad_token_id=None,
        **kwargs,
    ) -> torch.LongTensor:
        """Autoregressive generation for inference."""
        from transformers.cache_utils import DynamicCache

        text_ids = kwargs.pop("text_ids", None)
        text_attention_mask = kwargs.pop("text_attention_mask", None)
        stream_chunk_size = int(kwargs.pop("stream_chunk_size", 0) or 0)
        memory_mode = kwargs.pop("memory_mode", "default")
        tokenizer = kwargs.pop("tokenizer", None)
        memory_init_text = kwargs.pop("memory_init_text", "No previous memory")
        memory_update_template = kwargs.pop(
            "memory_update_template",
"""
The above is a newly provided section, which may or may not contain information relevant to answering the problem. The previously summarized memory is stored within the <previous_memory> tags, and the problem is specified within the <problem> tags. Please read the provided section carefully and update the memory with any new information that helps address the problem. Be sure to retain all relevant details from the previous memory while incorporating any new, useful information.

<problem> 
{prompt}
</problem>

<previous_memory>
{memory}
</previous_memory>

Updated memory:
""",
        )
        final_answer_template = kwargs.pop(
            "final_answer_template",
            """You are presented with a problem and a previous memory.You do not need to update the memory, directly answer the problem based on the previous memory.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
""",
        )

        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        elif eos_token_id is None:
            eos_token_id = []
        past_key_values = self._normalize_to_dynamic_cache(past_key_values)

        def _extract_valid_tokens(ids_2d: torch.Tensor, mask_2d: torch.Tensor, idx: int) -> torch.LongTensor:
            valid = mask_2d[idx].to(dtype=torch.bool)
            if valid.any():
                return ids_2d[idx][valid].detach().cpu().to(dtype=torch.long)
            return torch.empty(0, dtype=torch.long)

        def _concat_generated(
            prompt_ids: torch.Tensor, prompt_mask: torch.Tensor, generated: torch.Tensor
        ) -> torch.Tensor:
            out_rows = []
            bsz = prompt_ids.size(0)
            for row_idx in range(bsz):
                p = _extract_valid_tokens(prompt_ids, prompt_mask, row_idx).to(generated.device)
                g = generated[row_idx]
                out_rows.append(torch.cat([p, g], dim=0))
            from utils import pad_tensor_list_to_length

            max_len = max(x.numel() for x in out_rows) if out_rows else 0
            if max_len == 0:
                return torch.empty((bsz, 0), dtype=prompt_ids.dtype, device=generated.device)
            padded = pad_tensor_list_to_length(
                out_rows,
                pad_token_id=pad_token_id,
                max_length=max_len,
                left_pad=False,
            )
            return padded.to(device=generated.device, dtype=prompt_ids.dtype)


        def _generate_vanilla(
            in_ids: torch.Tensor,
            in_attention: Optional[torch.Tensor],
            in_past_key_values,
            vanilla_max_tokens,
        ) -> torch.Tensor:
            device = in_ids.device
            batch_size = in_ids.shape[0]
            cur_attention = in_attention
            cur_past_key_values = self._normalize_to_dynamic_cache(in_past_key_values)

            if cur_past_key_values is not None and isinstance(cur_past_key_values, DynamicCache):
                kv_seq_len = cur_past_key_values.get_seq_length()
                real_input_ids = in_ids[:, kv_seq_len:]
                if cur_attention is None:
                    cur_attention = torch.ones(
                        batch_size,
                        kv_seq_len + real_input_ids.shape[1],
                        dtype=torch.long,
                        device=device,
                    )
            else:
                if cur_attention is None:
                    cur_attention = torch.ones_like(in_ids)

                seq_len = in_ids.shape[1]
                lychee_memory_window = self.config.lychee_memory_window

                if seq_len <= lychee_memory_window:
                    cur_past_key_values = DynamicCache()
                    real_input_ids = in_ids
                else:
                    kv_cache, kv_mask = self.aggregate_text(in_ids, cur_attention)
                    cur_past_key_values = kv_cache
                    real_input_ids = in_ids[:, -1:]
                    cur_attention = torch.cat(
                        [
                            kv_mask.to(device),
                            torch.ones(batch_size, 1, dtype=torch.long, device=device),
                        ],
                        dim=1,
                    )

            outputs = self.generate_forward(
                input_ids=real_input_ids,
                attention_mask=cur_attention,
                past_key_values=cur_past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            next_token_logits = outputs.logits[:, -1, :]
            cur_past_key_values = outputs.past_key_values

            generated_ids = in_ids
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            decode_no_mask = False
            if cur_attention is None:
                decode_no_mask = True
            elif isinstance(cur_attention, torch.Tensor):
                decode_no_mask = bool((cur_attention != 0).all().item())

            for _ in range(vanilla_max_tokens):
                logits = next_token_logits.float()

                if do_sample:
                    if temperature > 0 and temperature != 1.0:
                        logits = logits / temperature
                    if top_k > 0:
                        k = min(top_k, logits.size(-1))
                        kth = torch.topk(logits, k, dim=-1)[0][:, -1:]
                        logits = logits.masked_fill(logits < kth, float("-inf"))
                    if 0 < top_p < 1.0:
                        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                        probs_sorted = F.softmax(sorted_logits, dim=-1)
                        cum_probs = torch.cumsum(probs_sorted, dim=-1)
                        remove = (cum_probs - probs_sorted) >= top_p
                        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                        logits = logits.scatter(1, sorted_idx, sorted_logits)
                    probs = F.softmax(logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    next_tokens = logits.argmax(dim=-1)

                for eos_id in eos_token_id:
                    finished = finished | (next_tokens == eos_id)
                if pad_token_id is not None:
                    next_tokens = next_tokens.masked_fill(finished, pad_token_id)

                generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(1)], dim=1)

                if finished.all():
                    break

                decode_attention = None
                if not decode_no_mask:
                    cur_attention = torch.cat(
                        [
                            cur_attention,
                            torch.ones(batch_size, 1, dtype=torch.long, device=device),
                        ],
                        dim=1,
                    )
                    decode_attention = cur_attention
                outputs = self.generate_forward(
                    input_ids=next_tokens.unsqueeze(1),
                    attention_mask=decode_attention,
                    past_key_values=cur_past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
                next_token_logits = outputs.logits[:, -1, :]
                cur_past_key_values = outputs.past_key_values

            return generated_ids

        if text_ids is not None and memory_mode == "recurrent":
            if input_ids is None:
                raise ValueError("When passing text_ids, input_ids (question/prompt) must also be provided.")
            if past_key_values is not None:
                raise ValueError("text_ids and past_key_values cannot be used together in recurrent mode.")
            if tokenizer is None:
                raise ValueError("recurrent memory mode requires `tokenizer` in generate kwargs.")

            from utils import TokenTemplate, chat_template, create_attention_mask, pad_tensor_list_to_length

            if text_attention_mask is None:
                text_attention_mask = torch.ones_like(text_ids)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            text_ids = text_ids.to(input_ids.device)
            text_attention_mask = text_attention_mask.to(input_ids.device)
            attention_mask = attention_mask.to(input_ids.device)
            question_ids = input_ids
            question_attention = attention_mask

            if stream_chunk_size <= 0:
                stream_chunk_size = int(getattr(self.config, "lychee_memory_window", 2048))

            chat_tpl = chat_template(tokenizer)
            update_tpl = TokenTemplate(chat_tpl.format(message=memory_update_template), tokenizer)
            final_tpl = TokenTemplate(chat_tpl.format(message=final_answer_template), tokenizer)
            no_memory_tokens = tokenizer.encode(memory_init_text, add_special_tokens=False)

            bsz = question_ids.size(0)
            current_memory_ids: List[torch.LongTensor] = [
                torch.tensor(no_memory_tokens, dtype=torch.long) for _ in range(bsz)
            ]
            question_valid_ids: List[torch.LongTensor] = [
                _extract_valid_tokens(question_ids, question_attention, i) for i in range(bsz)
            ]

            total_len = text_ids.size(1)
            for start in range(0, total_len, stream_chunk_size):
                end = min(start + stream_chunk_size, total_len)
                chunk_ids = text_ids[:, start:end]
                chunk_attention = text_attention_mask[:, start:end]
                active_mask = chunk_attention.sum(dim=1) > 0
                if not active_mask.any():
                    continue
                active_indices = torch.where(active_mask)[0].tolist()
                chunk_ids_active = chunk_ids[active_mask]
                chunk_attention_active = chunk_attention[active_mask]

                if stream_chunk_size > 0 and chunk_ids_active.size(1) < stream_chunk_size:
                    pad_len = stream_chunk_size - chunk_ids_active.size(1)
                    pad_ids = torch.full(
                        (chunk_ids_active.size(0), pad_len),
                        fill_value=pad_token_id,
                        dtype=chunk_ids_active.dtype,
                        device=chunk_ids_active.device,
                    )
                    pad_mask = torch.zeros(
                        (chunk_attention_active.size(0), pad_len),
                        dtype=chunk_attention_active.dtype,
                        device=chunk_attention_active.device,
                    )
                    chunk_ids_active = torch.cat([chunk_ids_active, pad_ids], dim=1)
                    chunk_attention_active = torch.cat([chunk_attention_active, pad_mask], dim=1)

                kv_cache_active, kv_mask_active = self.aggregate_text(
                    text_ids=chunk_ids_active,
                    text_attention_mask=chunk_attention_active,
                    stream_chunk_size=0,
                )

                update_messages: List[torch.LongTensor] = []
                for i in active_indices:
                    prompt_i = question_valid_ids[i]
                    memory_i = current_memory_ids[i]
                    update_message_i = update_tpl.format(
                        prompt=prompt_i,
                        memory=memory_i,
                    )
                    update_messages.append(update_message_i.to(dtype=torch.long))

                if not update_messages:
                    continue

                update_input_ids = pad_tensor_list_to_length(
                    update_messages,
                    pad_token_id=pad_token_id,
                    left_pad=True,
                ).to(question_ids.device)
                update_attention = create_attention_mask(update_input_ids, pad_token_id=pad_token_id).to(
                    question_ids.device
                )

                selected_kv_mask = kv_mask_active.to(question_ids.device)
                kv_seq_len = selected_kv_mask.size(1)
                fake_ids = torch.ones(
                    update_input_ids.size(0),
                    kv_seq_len,
                    dtype=update_input_ids.dtype,
                    device=question_ids.device,
                )
                full_input_ids = torch.cat([fake_ids, update_input_ids], dim=1)
                full_attention = torch.cat([selected_kv_mask, update_attention], dim=1)

                update_out = _generate_vanilla(
                    in_ids=full_input_ids,
                    in_attention=full_attention,
                    in_past_key_values=kv_cache_active,
                    vanilla_max_tokens=memory_size,
                )
                update_prompt_len = full_input_ids.shape[1]
                new_memory_ids = update_out[:, update_prompt_len:]

                for local_idx, sample_idx in enumerate(active_indices):
                    mem_ids = new_memory_ids[local_idx]
                    if pad_token_id is not None:
                        mem_ids = mem_ids[mem_ids != pad_token_id]
                    for _eos in eos_token_id:
                        mem_ids = mem_ids[mem_ids != _eos]
                    if memory_size > 0 and mem_ids.numel() > memory_size:
                        mem_ids = mem_ids[-memory_size:]
                    current_memory_ids[sample_idx] = mem_ids.detach().cpu().to(dtype=torch.long)

            final_messages: List[torch.LongTensor] = []
            for i in range(bsz):
                prompt_i = question_valid_ids[i]
                memory_i = current_memory_ids[i]
                final_message_i = final_tpl.format(
                    prompt=prompt_i,
                    memory=memory_i,
                )
                final_messages.append(final_message_i.to(dtype=torch.long))

            final_input_ids = pad_tensor_list_to_length(
                final_messages,
                pad_token_id=pad_token_id,
                left_pad=True,
            ).to(question_ids.device)
            final_attention = create_attention_mask(final_input_ids, pad_token_id=pad_token_id).to(question_ids.device)

            final_out = _generate_vanilla(
                in_ids=final_input_ids,
                in_attention=final_attention,
                in_past_key_values=None,
                vanilla_max_tokens=max_new_tokens,
            )
            final_prompt_len = final_input_ids.shape[1]
            final_answer_ids = final_out[:, final_prompt_len:]
            return _concat_generated(question_ids, question_attention, final_answer_ids)

        strip_virtual_prefix = 0
        if text_ids is not None:
            if input_ids is None:
                raise ValueError("When passing text_ids, input_ids (question/prompt) must also be provided.")
            if past_key_values is not None:
                raise ValueError("text_ids and past_key_values cannot be used together.")

            if text_attention_mask is None:
                text_attention_mask = torch.ones_like(text_ids)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            text_ids = text_ids.to(input_ids.device)
            text_attention_mask = text_attention_mask.to(input_ids.device)
            attention_mask = attention_mask.to(input_ids.device)

            # Non-recurrent mode: fall back to plain generation by prepending text_ids
            # to prompt input, without building/feeding any KV cache.
            strip_virtual_prefix = text_ids.size(1)
            input_ids = torch.cat([text_ids, input_ids], dim=1)
            attention_mask = torch.cat([text_attention_mask, attention_mask], dim=1)
            past_key_values = None

        if input_ids is None:
            raise ValueError("input_ids must be provided for generate().")
        generated_ids = _generate_vanilla(
            in_ids=input_ids,
            in_attention=attention_mask,
            in_past_key_values=past_key_values,
            vanilla_max_tokens=max_new_tokens,
        )

        if strip_virtual_prefix > 0:
            return generated_ids[:, strip_virtual_prefix:]
        return generated_ids

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
