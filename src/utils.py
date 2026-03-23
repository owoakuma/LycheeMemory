# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import re
import torch
import numpy as np
from transformers import PreTrainedTokenizer
from tensordict import TensorDict # this will initilize CUDA! make sure your CUDA_VISIBLE_DEVICES is set!
from typing import List
import datetime
def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class TokenTemplate:
    """
    format string, but in token_ids, use torch.LongTensor as data type.
    Input value can also be nunpy.ndarray or list[int].

    usage:
    ```
    TEMPLATE = "Here is a problem: {problem}"
    "Given this section: {section}"
    "Please answer it."

    processor = TokenTemplate(TEMPLATE, tokenizer)

    kwarg_text = dict(
        problem="What is the capital of France?",
        section="Here is a introduction to France. France is a country in Western Europe. Its capital is Paris.",
    )
    kwargs_token_ids = {
        k: tokenizer.encode(v, add_special_tokens=False) for k, v in kwarg_text.items()
    }

    print(tokenizer.decode(processor.format(**kwargs_token_ids)))

    # just as a text format string.
    assert TEMPLATE.format(**kwarg_text) == tokenizer.decode(processor.format(**kwargs_token_ids))
    ```
    """

    def __init__(self, template: str, tokenizer: PreTrainedTokenizer=None):
        self.template = template
        self.initialized = False
        if tokenizer:
            self.init(tokenizer)
        
    def init(self, tokenizer):
        self.keywords: list[str] = []  # Store extracted {keywords}
        self.token_sections: list[torch.LongTensor] = []  # Store tokenized text sections as LongTensors
        self.last_section: torch.LongTensor = None  # Last section as LongTensor
        
        # Match all {keywords}
        pattern = r'\{([a-zA-Z]+)\}'        
        parts = re.split(pattern, self.template)
        
        # Split text: even indices are non-{} parts, odd indices are {} keywords
        for i, part in enumerate(parts[:-1]):
            if i % 2 == 0:  # Even index, non-{} part
                tokens = tokenizer.encode(part, add_special_tokens=False)
                self.token_sections.append(torch.tensor(tokens, dtype=torch.long))
            else:  # Odd index, {} keyword
                self.keywords.append(part)
        self.last_section = torch.tensor(tokenizer.encode(parts[-1], add_special_tokens=False), dtype=torch.long)
        
        assert len(self.keywords) == len(self.token_sections), \
            f"{self.keywords} and {self.token_sections} should have the same length"
        self.initialized = type(tokenizer)

    @property
    def length(self) -> int:
        """
        Length of the template in token numbers
        """
        total = sum(section.numel() for section in self.token_sections)
        total += self.last_section.numel()
        return total

    def format(self, **kwargs: dict[str, torch.LongTensor | list[int] | np.ndarray]) -> torch.LongTensor:
        """
        Format the template with provided token ids
        
        Args:
            **kwargs: Dictionary of keyword to token ids (as LongTensor)
            
        Returns:
            Concatenated token ids as LongTensor
        """
        # Initialize with first section if exists
        formatted_parts = []
        
        # Reconstruct template by interleaving sections and keyword tokens
        for i, k in enumerate(self.keywords):
            if isinstance(kwargs[k], list):
                kwargs[k] = torch.tensor(kwargs[k], dtype=torch.long)
            elif isinstance(kwargs[k], np.ndarray):
                kwargs[k] = torch.from_numpy(kwargs[k]).to(torch.long)
            formatted_parts.append(self.token_sections[i])
            formatted_parts.append(kwargs[k])
        formatted_parts.append(self.last_section)
        
        return torch.cat(formatted_parts)

def chat_template(tokenizer, system=False) -> str:
    if system:
        return tokenizer.apply_chat_template([{'role':'system','content':'{system}'},
                                              {'role':'user','content':'{message}'}],
                                                    add_generation_prompt=True,
                                                    tokenize=False)
    else:
        return tokenizer.apply_chat_template([{'role':'user','content':'{message}'}],
                                                    add_generation_prompt=True, 
                                                    tokenize=False)


import re
from typing import Dict, Any

def validate_structured_response(response: str) -> Dict[str, Any]:
    """
    根据设定的规则验证响应字符串的结构。

    规则：
    1. 设置 is_valid=False 作为初始状态。
    2. 首先查找文本中是否有中文，如果有，立即返回验证失败。
    3. 如果没有中文，则进行正则匹配，必须严格符合以下四种合法格式之一：
       - <thinking>任意内容</thinking><skip>
       - <thinking>任意内容</thinking><new_memory>任意内容</new_memory><need_more_info>
       - <thinking>任意内容</thinking><new_memory>任意内容</new_memory><finish>
       - <thinking>任意内容</thinking><new_memory>任意内容</new_memory><finish>任意内容</finish>
    4. "任意内容"区域内不能包含任何标签（即不能包含 '<' 或 '>' 字符）。
    """
    # 初始化返回结果，is_valid 默认为 False
    result = {
        'is_valid': True,
        'reasons': [],
        'has_thinking': False,
        'has_new_memory': False,
        'has_skip': False,
        'has_need_more_info': False,
        'has_finish': False,
    }

    # text = response or ""

    # # 1. 检查是否包含中文字符
    # if re.search(r'[\u4e00-\u9fa5]', text):
    #     result['reasons'].append("响应中包含中文字符")
    #     result['is_valid'] = False
        
    # 2. 检查是否包含<answer></answer>以外的标签
    # if re.search(r'<(?!/?answer\b)[^>]+>', text):
    #     result['reasons'].append("响应中包含<answer></answer>以外的标签")
    #     result['is_valid'] = False
        
    return result

    # # 定义“任意内容”的正则表达式，不允许包含任何标签
    # # [^<] 表示匹配除 '<' 之外的任何单个字符
    # # content_pattern = r"[^<]*"
    # content_pattern = r"[\s\S]*"

    # # 2. 定义四种合法格式的正则表达式
    # # 使用 (?:thinking|think) 来匹配 <thinking> 或 <think>
    # # 使用 re.IGNORECASE 标志来忽略大小写
    # # 使用 ^ 和 $ 来确保整个字符串都必须匹配
    # patterns = {
    #     'skip': re.compile(
    #         rf"^\s*<(?:thinking|think)>{content_pattern}</(?:thinking|think)>\s*<skip>\s*$", 
    #         re.IGNORECASE
    #     ),
    #     'need_more_info': re.compile(
    #         rf"^\s*<(?:thinking|think)>{content_pattern}</(?:thinking|think)>\s*<new_memory>{content_pattern}</new_memory>\s*<need_more_info>\s*$",
    #         re.IGNORECASE
    #     ),
    #     'finish_simple': re.compile(
    #         rf"^\s*<(?:thinking|think)>{content_pattern}</(?:thinking|think)>\s*<new_memory>{content_pattern}</new_memory>\s*<finish>\s*$",
    #         re.IGNORECASE
    #     ),
    #     'finish_with_content': re.compile(
    #         rf"^\s*<(?:thinking|think)>{content_pattern}</(?:thinking|think)>\s*<new_memory>{content_pattern}</new_memory>\s*<finish>{content_pattern}</finish>\s*$",
    #         re.IGNORECASE
    #     ),
    # }

    # # 3. 遍历并尝试匹配每一种格式
    # match_found = False
    # if re.match(patterns['skip'], text):
    #     match_found = True
    #     result['has_thinking'] = True
    #     result['has_skip'] = True
    # elif re.match(patterns['need_more_info'], text):
    #     match_found = True
    #     result['has_thinking'] = True
    #     result['has_new_memory'] = True
    #     result['has_need_more_info'] = True
    # elif re.match(patterns['finish_simple'], text):
    #     match_found = True
    #     result['has_thinking'] = True
    #     result['has_new_memory'] = True
    #     result['has_finish'] = True
    # elif re.match(patterns['finish_with_content'], text):
    #     match_found = True
    #     result['has_thinking'] = True
    #     result['has_new_memory'] = True
    #     result['has_finish'] = True

    # # 4. 根据匹配结果更新最终的 result
    # if match_found:
    #     result['is_valid'] = True
    # else:
    #     result['reasons'].append("格式不符合四种合法格式中的任何一种")

    # return result

def r2l_pad(tensor: torch.Tensor, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将右填充的2D张量转换为左填充，并返回对应的attention_mask。
    
    Args:
        tensor: 形状为 (batch_size, seq_len) 的2D张量，假设使用右填充
        pad_token_id: 填充token的ID
    
    Returns:
        tuple: (左填充的张量, attention_mask)
            - 左填充张量: 形状与输入相同
            - attention_mask: 形状为 (batch_size, seq_len)，1表示真实token，0表示填充token
    """
    batch_size, seq_len = tensor.shape
    
    # 创建结果张量，初始化为pad_token_id
    result = torch.full_like(tensor, pad_token_id)
    # 创建attention_mask，初始化为0（填充位置）
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.int, device=tensor.device)
    
    for i in range(batch_size):
        # 找到每行中非填充token的位置
        non_pad_mask = tensor[i] != pad_token_id
        
        # 如果整行都是填充token，跳过
        if not non_pad_mask.any():
            continue
            
        # 获取非填充的tokens
        non_pad_tokens = tensor[i][non_pad_mask]
        num_non_pad = len(non_pad_tokens)
        
        # 计算左填充的起始位置
        start_pos = seq_len - num_non_pad
        
        # 将非填充tokens放到右侧
        result[i, start_pos:] = non_pad_tokens
        # 设置对应位置的attention_mask为1
        attention_mask[i, start_pos:] = 1
    
    return result, attention_mask

def graceful_padding(bsz: int, group_nums: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates an index mapping tensor that handles padding for grouped batches.
    
    The pattern is:
    - First group has no padding
    - Subsequent groups have 1 padding element
    - Padding elements are mapped to -1 (will be concatenated at the end)
    - Non-padding elements maintain their original order
    
    Example pattern for bsz=7, group_nums=3:
    no_padding_mask: [1, 1, 1, 0, 1, 1, 0, 1, 1]
    padding_index:   [0, 1, 2, -1, 3, 4, -1, 5, 6]
    
    Args:
        bsz: Batch size
        group_nums: Number of groups to split the batch into
        
    Returns:
        A tensor containing the index mapping with padding elements marked as -1
    """
    group_size = bsz // group_nums + 1
    reminder = bsz % group_nums
    if not reminder:
        return torch.arange(bsz), torch.ones(bsz, dtype=torch.bool)
    
    # Create mask where 1 = no padding, 0 = padding
    no_padding_mask = torch.tensor(
        [1 if i // group_size < reminder or i % group_size else 0 
         for i in range(group_nums * group_size)],
        dtype=torch.int
    )
    
    # Create cumulative index (shifted by -1)
    padding_index = torch.cumsum(no_padding_mask, dim=0) - 1
    
    # Mark padding elements with -1
    padding_index[~no_padding_mask.bool()] = -1
    
    return padding_index, no_padding_mask.bool()


# General torch utils
def pad_tensor_list_to_length(response: List[torch.LongTensor], pad_token_id, max_length=None, left_pad=True, return_mask=False):
    """
    similar to verl.utils.torch_functional.pad_2d_list_to_length 
    but 1. support left_pad 2. accept list[torch.Tensor] as input
    20x faster than pad_2d_list_to_length:
        - if use 2d list, simply create a tensor with shape(8192, 8192) will take 15s
        - if use list of 1d tensor, the whole process to pad(~8000->16384), concat, and stack will take only ~1s.
    """
    response_length = max(len(sub_list) for sub_list in response)
    if max_length is not None and max_length > response_length:
        target_length = max_length
    else:
        target_length = response_length
    full_long = lambda len, v: torch.full(
        (len,), fill_value=v, dtype=response[0].dtype, device=response[0].device
    )
    if left_pad:
        padded_response = [torch.cat([full_long(target_length - len(sub_tensor), pad_token_id),
                                      sub_tensor]) for sub_tensor in response]     
    else:
        padded_response = [torch.cat([sub_tensor,
                                      full_long(target_length - len(sub_tensor), pad_token_id)
                                      ]) for sub_tensor in response]    
    padded_response = torch.stack(padded_response)
    if return_mask:
        mask = torch.full(padded_response.shape, True, dtype=torch.bool)
        if left_pad:
            [mask[i, :target_length - len(sub_tensor)].fill_(False) for i, sub_tensor in enumerate(response)]
        else:
            # [-0 : ] will be the whole tensor instead of empty tensor
            [mask[i, -(target_length - len(sub_tensor)):].fill_(False) 
             for i, sub_tensor in enumerate(response) if target_length - len(sub_tensor) > 0]
        return padded_response, mask
    else:
        return padded_response

def unpad(tokenizer, tensor: torch.Tensor, remove_eos: bool = False) -> np.ndarray:
    """Unpad tensor. Remove eos if specified"""
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(0) 
    attention_mask = ~(tensor == tokenizer.pad_token_id)
    if remove_eos:
        attention_mask &= ~(tensor == tokenizer.eos_token_id)
    
    # Force object array format to avoid numpy's automatic conversion
    # when all tensors have the same length after unpadding
    result = np.empty(tensor.shape[0], dtype=object)
    for i in range(tensor.shape[0]):
        result[i] = tensor[i][attention_mask[i]]
    
    return result

def create_attention_mask(input_ids: torch.Tensor, pad_token_id) -> torch.Tensor:
    """Create attention mask from input ids."""
    return (input_ids != pad_token_id).to(torch.long)

def create_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Create position ids from attention mask."""
    return torch.clamp_min(torch.cumsum(attention_mask, dim=1) - 1, min=0)


def td_split(td: TensorDict, sections: int) -> list[TensorDict]:
    """
    split TensorDict in dim0, allows different sections, like torch.tensor_split and np.array_split
    used in workers/dp_actor to support variable length of batch size
    """
    if len(td) < sections:
        print(f"error occurred when trying to split {td}")
        raise ValueError(f"len(proto)={len(td)} < sections={sections}")        
    
    tensors_splitted = {k: torch.tensor_split(v, sections) for k, v in td.items()}
    return [TensorDict.from_dict({k: v[i] for k, v in tensors_splitted.items()}) for i in range(sections)]

def reverse_indices(tensor):
    """
    Return the unique elements of a tensor and their indices.
    https://discuss.pytorch.org/t/reverse-inverse-indices-torch-unique/114521/6
    return the reverse indices of an array
    ```py
    t[reverse_indices(t)] = unique(t)
    # e.g. [1, 4, 3, 2, 0] -> [4, 0, 3, 2, 1], 
    torch.tensor([1, 4, 3, 2, 0])[
          torch.tensor([4, 0, 3, 2, 1])
    ] == [0, 1, 2, 3, 4]
    ```

    Used in final_batch
    """
    unique, inverse_indices = torch.unique(tensor, return_inverse=True)
    assert len(unique) == len(tensor), f"Your input tensor has duplicated elements."
    indices = torch.scatter_reduce(
        torch.zeros_like(unique, dtype=torch.long, device=tensor.device), 
        dim=0,
        index=inverse_indices,
        src=torch.arange(tensor.size(0), device=tensor.device),
        reduce="amin",
        include_self=False,
    )
    return indices


def clip_long_string(string, max_length=2000):
    """Clip long string to a maximum length."""
    # assert max_length > 50, "max_length must be greater than 50"
    if not len(string) > max_length:
        return string
    target_len = max_length - len('\n\n...(truncated)\n\n')
    return string[:target_len//2] + '\n\n...(truncated)\n\n' + string[-target_len//2:]

def log_step(logger, step, conversation):
    logger.info("="*30 + f"STEP {step}" + "="*30)
    for i, msg in enumerate(conversation):
        logger.info(f"[{msg['role']}]:")
        logger.info(f"{clip_long_string(msg['content'])}")
        logger.info("-"*50)

from openai.types.chat.chat_completion import Choice
def msg(choice: Choice):
    if isinstance(choice.stop_reason, str):
        stop_suffix = choice.stop_reason
    else:
        # Here is some possible stop_reason:
        # 1. None if eos_token is generated
        # 2. 151643 if pad_token is generated
        stop_suffix = ""
    return {
        "role": choice.message.role,
        "content": choice.message.content + stop_suffix,
        "finished": choice.finish_reason == "stop"
    }