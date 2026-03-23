import torch
import numpy as np
from transformers.utils import logging
from itertools import cycle

logger = logging.get_logger(__name__)


class LycheeMemory(torch.nn.Module):
    def __init__(
        self, 
        model_config, 
        k_seq_dim:int=2, 
        v_seq_dim:int=2, 
    ):
        """Setup necessary attributes."""
        super().__init__()

        self.config = model_config

        # initialize necessary parameters
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.rng = np.random.default_rng(42)
        self.retain_kv = []
        self.kv_mask = None
        self.lychee_memory_token = torch.tensor([self.config.vocab_size])
        self.use_lychee_memory = True

        self._post_validation()
        self.reset()

    def _post_validation(self, verbose=True):
        assert self.config.lychee_memory_window >= self.config.lychee_memory_stride, f"Make sure the lychee_memory_window {self.config.lychee_memory_window} >= lychee_memory_stride {self.config.lychee_memory_stride}!"
        for ratio in self.config.lychee_memory_ratio:
            assert ratio >= 0, f"Make sure all lychee_memory ratios are greater than or equal to 0, found {self.config.lychee_memory_ratio}!"
        assert self.config.lychee_memory_attn in ["segmentation", "step-expansion", "full-coverage"], f"lychee_memory_attn {self.config.lychee_memory_attn} not implemented!"
        assert self.config.lychee_memory_ratio_mix in ["instance-random", "step-random", "sequence"] or "adapt-" in self.config.lychee_memory_ratio_mix, f"lychee_memory_ratio_mix {self.config.lychee_memory_ratio_mix} not implemented!"
        # assert self.config.lychee_memory_pos in ["append", "interleave"], f"lychee_memory_pos {self.config.lychee_memory_pos} not implemented!"
        if self.config.lychee_memory_pos == "interleave":
            assert self.config.lychee_memory_window == self.config.lychee_memory_stride, f"Make sure the lychee_memory_window equals to lychee_memory_stride when using interleaving mode."
        if self.config.lychee_memory_parallel_window > 1:
            assert self.config._attn_implementation != "flash_attention_2", f"Currently parallel window does not support flash_attention_2!"

        self._cpu = torch.device("cpu")

        if verbose:
            info = f"applying activation lychee_memory on {self.config.lychee_memory_param} (the lychee_memory embedding is initialized from {'bos' if self.config.lychee_memory_embed_init == 'bos' else 'eos'} embedding, the lychee_memory tokens are positioned with '{self.config.lychee_memory_pos}' method), with window size {self.config.lychee_memory_window}, stride {self.config.lychee_memory_stride}, {self.config.lychee_memory_attn} attention{' (attending to previous lychee_memorys)' if self.config.lychee_memory_attend_prev else ' (no attending to previous lychee_memorys)'}, sink size {self.config.lychee_memory_sink_size}, compression ratio {self.config.lychee_memory_ratio} (mixed by {self.config.lychee_memory_ratio_mix})..."
            logger.info(info)

    def set(self, verbose=True, **kwargs):
        """
        Set attributes out of the constructor.
        """
        for k, v in kwargs.items():
            setattr(self.config, k, v)
        self._post_validation(verbose=verbose)

    def reset(self):
        """Initialize attributes for a new sequence."""
        # the cursor pointing to the start of the current window
        self._start_idx = 0
        # the cursor pointing to the end of the current window
        self._end_idx = 0
        # the lychee_memory sizes of all strides
        self._all_lychee_memory_sizes = []
        # the loss per batch
        self._batch_loss = None
        # the valid token number per batch
        self._valid_token_num = None
        # the step index for processing the input_ids
        self._step_idx = 0
        # used in set_compression_ratio
        self._compression_ratio = None
        # the previous inputs is a full window or not, defaults to True
        self._is_full_window = True
        # the number of raw activations to preserve in update_memory (only useful when lychee_memory_stride < lychee_memory_window)
        self._raw_size_to_cache = 0

        # the number of tokens in previous stride that should be compressed by the upcoming lychee_memory
        self._interleave_remainder = 0
        # compression ratio for the unfinished window
        self._interleave_compression_ratio = None
        self._lychee_memory_indices = None

        self.all_input_ids = None
        self.all_attention_mask = None
        self.all_labels = None

        # NOTE: will be reset in prepare()
        self.lychee_memory_skip_first = None
        self.lychee_memory_skip_last = None

        # the raw activations of recent tokens
        self.raw_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]
        # the attention sink activations
        self.sink_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]
        # the lychee_memory activations
        self.lychee_memory_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]
        
        self.generate_past_key_values = None
        
        # self.retain_kv = []

    @property
    def all_sequence_length(self):
        if self.all_input_ids is None:
            return 0
        else:
            return self.all_input_ids.shape[1]

    @property
    def batch_size(self):
        if self.all_input_ids is None:
            return 0
        else:
            return self.all_input_ids.shape[0]

    @property
    def finish(self):
        is_finish = self._end_idx == self.all_sequence_length
        return is_finish

    @property
    def dtype(self):
        return self.config.torch_dtype

    @property
    def min_value(self):
        return torch.finfo(self.dtype).min

    @property
    def max_position_embeddings(self):
        max_position_embeddings = self.config.max_position_embeddings
        if getattr(self.config, "rope_scaling", None) is not None:
            scaling_factor = self.config.rope_scaling["factor"]
            max_position_embeddings = max_position_embeddings * scaling_factor
        return max_position_embeddings
 
    def get_memory_size(self):
        """
        Sink memory size, lychee_memory memory size and raw memory size.
        """
        sink_memory_size = 0
        lychee_memory_memory_size = 0
        raw_memory_size = 0
        if self.sink_activations[0][0] is not None:
            sink_memory_size += self.sink_activations[0][0].shape[self.k_seq_dim]
        if self.lychee_memory_activations[0][0] is not None:
            lychee_memory_memory_size += self.lychee_memory_activations[0][0].shape[self.k_seq_dim]
        if self.raw_activations[0][0] is not None:
            raw_memory_size += self.raw_activations[0][0].shape[self.k_seq_dim]
        return sink_memory_size, lychee_memory_memory_size, raw_memory_size

    def export_lychee_memory_cache(self):
        """Export current lychee_memory activations as a DynamicCache."""
        from transformers.cache_utils import DynamicCache

        if not self.lychee_memory_activations:
            return DynamicCache()

        legacy_cache = []
        for key, value in self.lychee_memory_activations:
            if key is None or value is None:
                return DynamicCache()
            legacy_cache.append((key, value))
        return DynamicCache.from_legacy_cache(legacy_cache)

    def build_kv_mask_from_attention(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Build compressed kv_mask from the raw text attention mask."""
        if attention_mask is None:
            raise ValueError("attention_mask must not be None when building kv_mask.")
        if attention_mask.ndim != 2:
            raise ValueError(f"Expected 2D attention_mask, got shape {tuple(attention_mask.shape)}.")

        comp_ratio = int(self.config.lychee_memory_ratio[0])
        if comp_ratio <= 0:
            raise ValueError(
                f"lychee_memory_ratio[0] must be > 0 to build kv_mask, got {self.config.lychee_memory_ratio[0]}."
            )

        kv_mask = []
        for i in range(0, attention_mask.shape[1], comp_ratio):
            chunk = attention_mask[:, i : i + comp_ratio]
            mask_val = (chunk.sum(dim=1, keepdim=True) > 0).to(dtype=torch.long)
            kv_mask.append(mask_val)
        return torch.cat(kv_mask, dim=1)

    def prepare(self, input_ids, attention_mask, labels, skip_first=None, skip_last=None):
        """
        Prepare inputs for the model. These inputs belong to the same sequence.
        """
        # assert input_ids.shape[0] == 1, "Make sure the batch size is 1!"
        # assert attention_mask is None or (attention_mask == 1).all(), "Make sure there is no padding!"

        self._device = input_ids.device

        # accumulate input_ids
        if self.all_input_ids is None:
            self.all_input_ids = input_ids.cpu()
        else:
            self.all_input_ids = torch.cat([self.all_input_ids, input_ids.cpu()], dim=1)

        # accumulate attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=torch.device("cpu"))
        if self.all_attention_mask is None:
            self.all_attention_mask = attention_mask.cpu()
        else:
            self.all_attention_mask = torch.cat([self.all_attention_mask, attention_mask.cpu()], dim=1)

        # accumulate labels if exisits
        if labels is not None:
            # rotate labels in advance so that the loss of the last token is not ignored in every window
            labels = torch.cat([labels[:, 1:].cpu(), torch.tensor([-100]).expand(labels.shape[0], 1)], dim=1)
            if self.all_labels is None:
                self.all_labels = labels.cpu()
            else:
                self.all_labels = torch.cat([self.all_labels, labels], dim=1)
            assert self.all_input_ids.shape[1] == self.all_labels.shape[1], f"Found inconsistent all_input_ids {self.all_input_ids.shape} and all_labels {self.all_labels.shape}!"
        
        # how many tokens to skip at the beginning of the sequence? (They will be packed in a single chunk and processed by the model, after which their activations will be cached in sink_activations.)
        if skip_first is not None:
            assert self.config.lychee_memory_parallel_window == 1, f"Make sure the parallel window is set to 1 when using lychee_memory_skip!"
            assert self.config.lychee_memory_window == self.config.lychee_memory_stride, f"Make sure the lychee_memory_window equals to lychee_memory_stride when using lychee_memory_skip."
            assert self.config.lychee_memory_sink_size == 0, f"Make sure the lychee_memory_sink_size is set to 0 when using lychee_memory_skip!"
        # stop compression after how many tokens
        if skip_last is not None:
            skip_first = skip_first if skip_first is not None else 0
            assert (skip_last - skip_first) % self.config.lychee_memory_window == 0, f"skip_last ({skip_last}) - skip_first ({skip_first}) = {skip_last - skip_first} is not divisible by window size {self.config.lychee_memory_window}"
            assert self.config.lychee_memory_sink_size == 0, "Make sure the lychee_memory_sink_size is zero when using skip_last!"
            
        self.lychee_memory_skip_first = skip_first
        self.lychee_memory_skip_last = skip_last

    def set_compression_ratio(self, start_idx, end_idx):
        """Choose a condensing ratio from self.config.lychee_memory_ratio"""
        def filter_ratio(ratios, stride):
            valid_ratios = []
            for ratio in ratios:
                # stride must be bigger than condensing ratio because we there must be at least one lychee_memory
                if stride < ratio:
                    continue
                # the stride must be evenly divisible by condensing ratio
                if ratio > 0 and (stride % ratio) != 0:
                    continue
                # when training, ratio=0 is valid if previous windows contain lychee_memory or later windows contain lychee_memory
                if ratio == 0 and self.training:
                    previous_has_zero = -1 in self._all_lychee_memory_sizes
                    following_has_nonzero = (start_idx + stride + self.config.lychee_memory_window) <= self.all_sequence_length
                    if previous_has_zero or (not following_has_nonzero):
                        continue
                valid_ratios.append(ratio)
            assert len(valid_ratios), f"Cannot find valid condensing ratio (among {ratios}) for stride {stride}!"
            return valid_ratios

        def get_max_length(ratios):
            max_lengths = []
            for compression_ratio in ratios:
                if compression_ratio > 0:
                    # NOTE: here we must use the scaled position embeddings
                    max_lengths.append((self.max_position_embeddings - self.config.lychee_memory_window) * compression_ratio + self.config.lychee_memory_window)
                else:
                    max_lengths.append(self.max_position_embeddings)
            return max_lengths

        if len(self.config.lychee_memory_ratio) == 1:
            return self.config.lychee_memory_ratio[0]

        ratio_mix = self.config.lychee_memory_ratio_mix

        lychee_memory_ratio = filter_ratio(self.config.lychee_memory_ratio, self.config.lychee_memory_stride)

        if ratio_mix == "instance-random":
            if self._compression_ratio is None:
                lychee_memory_ratio = self.rng.choice(lychee_memory_ratio).tolist()
                self._compression_ratio = lychee_memory_ratio
            else:
                lychee_memory_ratio = self._compression_ratio

        elif ratio_mix == "step-random":
            lychee_memory_ratio = self.rng.choice(lychee_memory_ratio).tolist()
        
        elif ratio_mix == "sequence":
            if self._compression_ratio is None:
                self._compression_ratio = cycle(lychee_memory_ratio)
            lychee_memory_ratio = next(self._compression_ratio)

        elif "adapt" in ratio_mix:
            if self._compression_ratio is None:
                future_length = int(ratio_mix.split("-")[1])
                sequence_length = self.all_input_ids.shape[1] + future_length
                max_lengths = get_max_length(lychee_memory_ratio)
                # ascendingly sort the max lengths
                valid_max_lengths_and_indices = [x for x in enumerate(max_lengths) if x[1] >= sequence_length]
                if len(valid_max_lengths_and_indices):
                    minimum_length_index = min(valid_max_lengths_and_indices, key=lambda x: x[1])[0]
                    # use the minimal possible length for this sequence (the smallest fold ratio)
                    lychee_memory_ratio = lychee_memory_ratio[minimum_length_index]
                else:
                    lychee_memory_ratio = max(lychee_memory_ratio)
                    # logger.warning(f"Failed to find valid fold window and size for sequence length {sequence_length}, as the maximum theoretical length is {max(max_lengths)}. Fall back to use the maximum one: {lychee_memory_ratio}.")
                self._compression_ratio = lychee_memory_ratio
            else:
                lychee_memory_ratio = self._compression_ratio

        return lychee_memory_ratio
    
    def step(self):
        # parallel does not support stride < window
        # parallel does not support non-compression
        # the input_ids is not long enough for parallel
        if \
        (self.config.lychee_memory_parallel_window > 1) and \
        (self.config.lychee_memory_stride == self.config.lychee_memory_window) and \
        (0 not in self.config.lychee_memory_ratio) and \
        (self.all_input_ids[:, self._end_idx:].shape[1] >= self.config.lychee_memory_parallel_window * self.config.lychee_memory_window):
            input_ids_list = []
            attention_mask_list = []
            position_ids_list = []
            labels_list = []

            lychee_memory_size_list = []
            lychee_memory_indices_list = []

            for i in range(self.config.lychee_memory_parallel_window):
                if i == 0:
                    _input_ids, _attention_mask, _position_ids, _past_key_values, _labels = self._step()
                else:
                    _input_ids, _attention_mask, _position_ids, _past_key_values, _labels = self._step(ignore_memory=True)

                input_ids_list.append(_input_ids)
                attention_mask_list.append(_attention_mask)
                position_ids_list.append(_position_ids)
                labels_list.append(_labels)
                lychee_memory_size_list.append(_past_key_values[0][2])
                lychee_memory_indices_list.append(_past_key_values[0][3])

                if i == 0:
                    past_key_values = _past_key_values
                    if past_key_values[0][0] is None:
                        mem_size = 0
                    else:
                        mem_size = past_key_values[0][0].shape[self.k_seq_dim]

                else:
                    # no memory
                    assert _past_key_values[0][0] is None
            
            batch_size = self.all_input_ids.shape[0]
            # NOTE: we do not need to repliace lychee_memory tokens for the last window
            seq_len = sum(x.shape[1] for x in input_ids_list) + sum(lychee_memory_size_list) - lychee_memory_size_list[-1]

            input_ids = _input_ids.new_zeros((batch_size, seq_len)) + self.lychee_memory_token.to(_input_ids.device)
            # all 0
            attention_mask = _attention_mask.new_zeros((batch_size, 1, seq_len, mem_size + seq_len)) + self.min_value
            position_ids = torch.arange(mem_size + seq_len, device=self._device).expand(batch_size, mem_size + seq_len)
            # 2 indicates the lychee_memory token is used for replication
            lychee_memory_indices = lychee_memory_indices_list[0].new_zeros(seq_len) + 2
            if _labels is not None:
                # -100 because no loss on lychee_memory tokens
                labels = _labels.new_zeros((batch_size, seq_len)) - 100
            else:
                labels = None

            start_idx = 0
            position_offset = mem_size
            for i in range(self.config.lychee_memory_parallel_window):
                lychee_memory_size = lychee_memory_size_list[i]

                # populate input_ids
                _input_ids = input_ids_list[i]
                cur_seq_len = _input_ids.shape[1]
                input_ids[:, start_idx: start_idx + cur_seq_len] = _input_ids
                
                # populate attention_mask and position_ids
                _attention_mask = attention_mask_list[i]
                _position_ids = position_ids_list[i]
                # the attention mask in the first window contains the mask for memory, which is redundant here
                if i == 0:
                    _attention_mask = _attention_mask[:, :, :, mem_size:]
                    _position_ids = _position_ids[:, mem_size:] - mem_size

                attention_mask[:, :, start_idx: start_idx + cur_seq_len, mem_size + start_idx: mem_size + start_idx + cur_seq_len] = _attention_mask
                position_ids[:, mem_size + start_idx: mem_size + start_idx + cur_seq_len] = _position_ids + position_offset

                # populate lychee_memory_indices
                _lychee_memory_indices = lychee_memory_indices_list[i]
                lychee_memory_indices[start_idx: start_idx + cur_seq_len] = _lychee_memory_indices

                # populate labels
                if labels is not None:
                    # populate labels
                    _labels = labels_list[i]
                    labels[:, start_idx: start_idx + cur_seq_len] = _labels

                # NOTE: when there is sink activations, we need to bias the position_ids for the first window
                if i == 0 and self.config.lychee_memory_sink_size > 0 and self.sink_activations[0][0] is None:
                    position_offset += 1

                # modify the attention and position for replicated lychee_memory tokens
                if i != self.config.lychee_memory_parallel_window - 1:
                    replicate_lychee_memory_row_start = start_idx + cur_seq_len
                    replicate_lychee_memory_col_start = mem_size + start_idx + cur_seq_len
                    # NOTE: any attention mask is okay for replicated lychee_memory tokens, but for convenience we use the causal mask
                    attention_mask[:, :, replicate_lychee_memory_row_start: replicate_lychee_memory_row_start + lychee_memory_size, replicate_lychee_memory_col_start: replicate_lychee_memory_col_start + lychee_memory_size] = _attention_mask.new_full((lychee_memory_size, lychee_memory_size), self.min_value).triu(1)
                    # NOTE: all future tokens can attend to the replicated lychee_memory tokens
                    attention_mask[:, :, replicate_lychee_memory_row_start + lychee_memory_size:, replicate_lychee_memory_col_start: replicate_lychee_memory_col_start + lychee_memory_size] = 0
                    # NOTE: the position of replicated lychee_memory tokens start from 0
                    position_ids[:, mem_size + start_idx + cur_seq_len: mem_size + start_idx + cur_seq_len + lychee_memory_size] = torch.arange(position_offset, position_offset + lychee_memory_size, device=_input_ids.device)[None:]

                start_idx += cur_seq_len + lychee_memory_size
                position_offset += lychee_memory_size

            # the memory is visible to all subsequent tokens
            attention_mask[:, :, :, :max(mem_size, self.config.lychee_memory_sink_size)] = 0

            # NOTE: modify lychee_memory_indices
            for i, (key, value, _, _) in enumerate(past_key_values):
                past_key_values[i] = (key, value, sum(lychee_memory_size_list), lychee_memory_indices)

            # NOTE: update _lychee_memory_indices so that the next-token logits can be properly sliced out in self.output()
            self._lychee_memory_indices = lychee_memory_indices
            
            return input_ids, attention_mask, position_ids, past_key_values, labels

        else:
            return self._step()

    def _step(self, ignore_memory=False):
        """
        Yield inputs for the current sliding window, including the input_ids, attention_mask, position_ids, and past_key_values.
        """
        #============================================#
        # Check whether the inputs fulfills a window.
        #============================================#

        # the starting position of the current window w.r.t. the start of the current input sequence
        start_idx = self._start_idx
        # the end position of the current window w.r.t. the start of the current input sequence
        end_idx = start_idx + self.config.lychee_memory_window
        # indicates if the current window is completely filled by raw activations and new tokens
        # we only append lychee_memory tokens for full windows
        if end_idx > self.all_sequence_length:
            # the input is shorter than the initial window size
            end_idx = self.all_sequence_length
            is_full_window = False
        else:
            is_full_window = True

        # NOTE: in training, the entire sequence is input to the model at once
        # In the last window, we do not need to append lychee_memorys because they will not be used at all
        if (self.training and end_idx == self.all_sequence_length) or (not self.use_lychee_memory):
            next_start_idx = start_idx
            is_full_window = False
            raw_size_to_cache = -1
            lychee_memory_size = 0
            compression_ratio = -1
        
        elif self._step_idx == 0 and self.lychee_memory_skip_first is not None:
            end_idx = start_idx + self.lychee_memory_skip_first
            assert end_idx < self.all_sequence_length
            next_start_idx = end_idx
            is_full_window = True
            raw_size_to_cache = -1
            lychee_memory_size = 0
            compression_ratio = -1
        
        elif self.lychee_memory_skip_last is not None and start_idx >= self.lychee_memory_skip_last:
            end_idx = min(start_idx + self.config.lychee_memory_window, self.all_sequence_length)
            next_start_idx = end_idx
            is_full_window = False
            raw_size_to_cache = -1
            lychee_memory_size = 0
            compression_ratio = -1

        else:
            #============================================#
            # Set compression ratio
            #============================================#
            if self.config.lychee_memory_pos == "append":
                if is_full_window:
                    # determine compression ratio for the current window
                    lychee_memory_stride = self.config.lychee_memory_stride
                    compression_ratio = self.set_compression_ratio(start_idx=start_idx, end_idx=end_idx)

                    if compression_ratio > 0:
                        # the stride must be evenly divisible by compression_ratio
                        lychee_memory_size = lychee_memory_stride // compression_ratio
                    else:
                        # the raw activations are used as lychee_memory activations
                        lychee_memory_size = -1

                    # forward start_idx and end_idx
                    next_start_idx = start_idx + lychee_memory_stride
                    # how many raw activations to save
                    raw_size_to_cache = end_idx - next_start_idx
                else:
                    # no stride because the sequence has finished
                    next_start_idx = start_idx
                    # cache all raw activations
                    raw_size_to_cache = -1
                    lychee_memory_size = 0
                    compression_ratio = 0

            elif self.config.lychee_memory_pos == "interleave":
                # the number of raw tokens in the input_ids
                input_size = end_idx - self._end_idx
                # set compression ratio once the previous window has finished, otherwise, reuse the interleave_compression_ratio if the input belongs to an unfinished window
                if self._is_full_window:
                    compression_ratio = self.set_compression_ratio(start_idx=start_idx, end_idx=end_idx)
                    self._interleave_compression_ratio = compression_ratio
                else:
                    compression_ratio = self._interleave_compression_ratio

                # the lychee_memory size is non-zero even if the window is not full
                if compression_ratio > 0:
                    # this number of lychee_memory tokens will be inserted among the raw tokens
                    lychee_memory_size = (input_size + self._interleave_remainder) // compression_ratio
                else:
                    # the raw activations are used as lychee_memory activations
                    lychee_memory_size = -1

                if is_full_window:
                    # move forward one window
                    next_start_idx = start_idx + self.config.lychee_memory_stride
                    # no save raw activations
                    raw_size_to_cache = 0
                else:
                    # no stride because the sequence has not finished
                    next_start_idx = start_idx
                    # cache all recent raw activations to be used in the next window
                    raw_size_to_cache = -1

        #============================================#
        # Slice out input_ids (raw tokens in the current window)
        #============================================#
        input_ids = self.all_input_ids[:, self._end_idx: end_idx].to(self._device)
        attention_mask = self.all_attention_mask[:, self._end_idx: end_idx].to(self._device)
        if self.all_labels is not None:
            labels = self.all_labels[:, self._end_idx: end_idx].to(self._device)
        else:
            labels = None
        batch_size = input_ids.shape[0]

        #============================================#
        # Insert lychee_memory tokens if necessary.
        #============================================#
        # t1 = time.time()

        if self.config.lychee_memory_pos == "append":
            # append lychee_memorys if necessary
            if is_full_window and lychee_memory_size > 0:
                input_ids = torch.cat([input_ids, self.lychee_memory_token.expand(batch_size, lychee_memory_size).to(input_ids.device, dtype=input_ids.dtype)], dim=1)
                # NOTE: prepend 1 to attention_mask because we have past_key_values
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones(batch_size, lychee_memory_size)], dim=1)
                if labels is not None:
                    labels = torch.cat([labels, labels.new_zeros(batch_size, lychee_memory_size) - 100], dim=1)

        elif self.config.lychee_memory_pos == "interleave":
            input_len = input_ids.shape[1]
            if lychee_memory_size > 0:
                # insert lychee_memory tokens in between raw tokens
                input_ids_with_lychee_memorys = input_ids.new_full((input_ids.shape[0], input_len + lychee_memory_size), self.lychee_memory_token.item())
                raw_token_indices = torch.arange(input_ids_with_lychee_memorys.shape[1], device=input_ids.device)
                interleave_start_idx = compression_ratio - self._interleave_remainder
                raw_token_indices = raw_token_indices[raw_token_indices % (compression_ratio + 1) != interleave_start_idx].unsqueeze(0).expand_as(input_ids)
                input_ids_with_lychee_memorys = input_ids_with_lychee_memorys.scatter(dim=1, index=raw_token_indices, src=input_ids)
                input_ids = input_ids_with_lychee_memorys
                # attention mask
                attention_mask_with_lychee_memorys = attention_mask.new_full((attention_mask.shape[0], attention_mask.shape[1] + lychee_memory_size), 1)
                attention_mask_with_lychee_memorys = attention_mask_with_lychee_memorys.scatter(dim=1, index=raw_token_indices, src=attention_mask)
                attention_mask = attention_mask_with_lychee_memorys
                # labels
                if labels is not None:
                    labels_with_lychee_memorys = labels.new_full((labels.shape[0], labels.shape[1] + lychee_memory_size), -100)
                    labels_with_lychee_memorys = labels_with_lychee_memorys.scatter(dim=1, index=raw_token_indices, src=labels)
                    labels = labels_with_lychee_memorys

            if compression_ratio > 0:
                # update the reminder
                self._interleave_remainder = (input_len + self._interleave_remainder) % compression_ratio

        # NOTE: skip computing loss in the very first window because the lychee_memory tokens will be used in the next window
        if self.training and self._step_idx == 0 and not (self.config.lychee_memory_pos == 'interleave' and self.config.lychee_memory_attn == 'full-coverage'):
            labels[:] = -100
        
        # t2 = time.time()

        #============================================#
        # Prepare lychee_memory_indices for interleave lychee_memory_pos, a boolean mask where True indicates the lychee_memory tokens.
        # The mask is applied on the inputs of the entire window, including the cached activations and the input_ids.
        #============================================#
        lychee_memory_indices = (input_ids[0] == self.lychee_memory_token.item()).long()
        if self._is_full_window:
            self._lychee_memory_indices = torch.tensor([], dtype=torch.long, device=input_ids.device)
        # the lychee_memory_indices always tracks the lychee_memory tokens in both the cached activations and the input_ids
        lychee_memory_indices = torch.cat([self._lychee_memory_indices, lychee_memory_indices])
        # record the lychee_memory_indices for the next window
        self._lychee_memory_indices = lychee_memory_indices
        if is_full_window and lychee_memory_size == -1:
            # NOTE: the first lychee_memory_stride raw tokens serve as lychee_memory tokens
            # we use -1 to indicate these raw tokens, so that the attention mask and position ids will not be modified
            lychee_memory_indices[:self.config.lychee_memory_stride] = -1

        # t3 = time.time()

        #============================================#
        # Prepare past_key_values.
            # lychee_memory_size: how many lychee_memory tokens are there in the input_ids
            # lychee_memory_indices: the boolean mask for the entire window where True indicates the lychee_memory tokens (for append, the lychee_memory_indices corresponds to input_ids, while for 'interleave', the lychee_memory_indices corresponds to the entire window including both the input_ids and the cached activations)
        #============================================#
        past_key_values = []
        for layer_idx in range(self.config.num_hidden_layers):
            if ignore_memory:
                key, value = None, None
            else:
                sink_key, sink_value = self.sink_activations[layer_idx]
                lychee_memory_key, lychee_memory_value = self.lychee_memory_activations[layer_idx]
                raw_key, raw_value = self.raw_activations[layer_idx]

                key = cat_tensor([
                    sink_key, lychee_memory_key, raw_key,
                ], dim=self.k_seq_dim)
                value = cat_tensor([
                    sink_value, lychee_memory_value, raw_value,
                ], dim=self.v_seq_dim)

            layer_past_key_values = (key, value, lychee_memory_size, lychee_memory_indices)
            past_key_values.append(layer_past_key_values)

        # t4 = time.time()
        
        #============================================#
        # Prepare attention_mask and position_ids.
        #============================================#
        first_key = past_key_values[0][0]
        mem_size = first_key.shape[self.k_seq_dim] if first_key is not None else 0
        if mem_size > 0:
            attention_mask = torch.cat([attention_mask.new_ones(batch_size, mem_size), attention_mask], dim=1)

        input_length = input_ids.shape[1]
        position_ids = torch.arange(attention_mask.shape[-1], dtype=torch.long, device=self._device).repeat(batch_size, 1)

        if self.config._attn_implementation == "flash_attention_2":
            assert self.config.lychee_memory_attn == "full-coverage", f"Make sure to set lychee_memory_attn='full-coverage' when using flash attention! Found {self.config.lychee_memory_attn}."
            if 0 in attention_mask:
                pass
            else:
                attention_mask = None
        elif self.config._attn_implementation == "sdpa" and self.config.lychee_memory_pos == "append" and lychee_memory_size <= 0 and (input_length == 1 or mem_size == 0):
            attention_mask = None
        else:
            attention_mask, position_ids = self._make_4d_attention_mask_and_position_ids(
                attention_mask, 
                position_ids, 
                mem_size, 
                lychee_memory_size, 
                compression_ratio, 
            )

        # t5 = time.time()

        # print(f"prepare inputs {t2-t1}, prepare indices {t3-t2}, prepare memory {t4-t3}, prepare attention mask {t5-t4}")

        #============================================#
        # Update necessary attributes.
        #============================================#
        # keep track of whether the current inputs is a full_window
        self._is_full_window = is_full_window
        # keep track of the raw_size_to_cache
        self._raw_size_to_cache = raw_size_to_cache
        # involked in self.output()
        self._all_lychee_memory_sizes.append(lychee_memory_size)
        # update end_idx
        self._start_idx = next_start_idx
        self._end_idx = end_idx
        self._step_idx += 1

        # print(f"start_idx:          {start_idx}")
        # print(f"next_start_idx:     {next_start_idx}")
        # print(f"lychee_memory_size:        {lychee_memory_size}")
        # print(f"raw_size_to_cache:  {raw_size_to_cache}")
        # print(f"interleave_remainder:{self._interleave_remainder}")
        # print(f"input_ids:          {input_ids}")
        # print(f"lychee_memory_indices:     {lychee_memory_indices}")
        # print(f"position_ids:       {position_ids}")
        # print(f"attention_mask:\n{attention_mask == 0}")
        # x = input()
        # if x == "s":
        #     return

        return input_ids, attention_mask, position_ids, past_key_values, labels

    def update_memory(self, past_key_values):
        """
        Accumulate lychee_memory activations and raw activations.
        """
        for layer_idx, (key, value, lychee_memory_size, lychee_memory_indices) in enumerate(past_key_values):
            # NOTE: the past_key_values are incrementally returned (only the new keys and values are returned)
            previous_raw_key, previous_raw_value = self.raw_activations[layer_idx]

            if self.lychee_memory_skip_first is not None and self.sink_activations[layer_idx][0] is None:
                assert key.shape[self.k_seq_dim] == self.lychee_memory_skip_first
                assert value.shape[self.k_seq_dim] == self.lychee_memory_skip_first
                self.sink_activations[layer_idx] = [
                    key,
                    value,
                ]
                # NOTE: no need to update raw activations and lychee_memory activations as all activations are kept as sink activations
                continue

            if self.lychee_memory_activations[layer_idx][0] is None and self.config.lychee_memory_sink_size > 0:
                # save the sink activations
                # NOTE: we do not slice the key/value activations, which may cause duplication when lychee_memory_ratio=-1 for the first window, but it's okay
                self.sink_activations[layer_idx] = [
                    slice_tensor(key, end=self.config.lychee_memory_sink_size, dim=self.k_seq_dim),
                    slice_tensor(value, end=self.config.lychee_memory_sink_size, dim=self.v_seq_dim),
                ]

            if not self._is_full_window:
                # this means the current input does not fulfill a window
                # thus, the key and value are all raw activations, and we accumulate them until the window is fulfilled
                assert self._raw_size_to_cache == -1
                raw_key = cat_tensor([
                    previous_raw_key,
                    key
                ], dim=self.k_seq_dim)
                raw_value = cat_tensor([
                    previous_raw_value, 
                    value
                ], dim=self.v_seq_dim)
                self.raw_activations[layer_idx] = (raw_key, raw_value)

            else:
                # NOTE: use the correct previous_lychee_memory_key and value!
                previous_lychee_memory_key, previous_lychee_memory_value = self.lychee_memory_activations[layer_idx]
                
                lychee_memory_key, lychee_memory_value, raw_key, raw_value = self._extract_lychee_memory_and_raw_memory(
                    key, 
                    value, 
                    previous_lychee_memory_key, 
                    previous_lychee_memory_value, 
                    previous_raw_key, 
                    previous_raw_value,     
                    lychee_memory_indices,
                )

                self.lychee_memory_activations[layer_idx] = (lychee_memory_key, lychee_memory_value)
                self.raw_activations[layer_idx] = (raw_key, raw_value)
    
    

    def update_loss(self, batch_loss, valid_token_num):
        """
        Accumulate loss for later perplexity computation and backward pass.
        """
        if self._batch_loss is None:
            # NOTE: multiply valid_token_num because batch_loss is divided by it in advance
            self._batch_loss = batch_loss * valid_token_num
            self._valid_token_num = valid_token_num
        else:
            # NOTE: avoid in-place operations, otherwise there will be gradient errors in training
            self._batch_loss = self._batch_loss + batch_loss * valid_token_num
            self._valid_token_num = self._valid_token_num + valid_token_num

    def output(self, model_outputs):
        """
        Override loss with accumulated loss. Update the next-token logits.
        """
        # override loss
        if self._batch_loss is not None:
            # here the batch_loss is the summation of all token losses in each element
            loss = self._batch_loss.sum() / self._valid_token_num.sum()

            # NOTE: prevent nan
            batch_loss = self._batch_loss / self._valid_token_num
            if (self._valid_token_num == 0).any():
                batch_loss = batch_loss.masked_fill(self._valid_token_num == 0, 0.)

            # NOTE: we must use dict to override values, otherwise trainer cannot find loss
            model_outputs["loss"] = loss
            model_outputs["batch_loss"] = batch_loss

        # override last_hidden_states (used in generation)
        lychee_memory_size = self._all_lychee_memory_sizes[-1]
        # remove logits corresponding to lychee_memory tokens
        if lychee_memory_size > 0:
            logits = model_outputs["logits"]
            lychee_memory_indices = self._lychee_memory_indices[-logits.shape[1]:]
            model_outputs["logits"] = logits[:, lychee_memory_indices == 0]

        return model_outputs

    def _make_4d_attention_mask_and_position_ids(
        self, 
        attention_mask, 
        position_ids,
        mem_size, 
        lychee_memory_size, 
        compression_ratio, 
    ):
        """
        Convert attention_mask into causal 4D attention_mask (batch_size, head_num, query_len, key_len).
        """
        tgt_size = attention_mask.size(-1) - mem_size
        dtype = self.dtype
        min_value = self.min_value
        device = self._device
        batch_size, src_size = attention_mask.size()

        # 🔧 优化：预分配完整大小的mask，避免concat产生临时张量
        # square for memory, and lower triangular for input_ids
        causal_mask = torch.full((tgt_size, src_size), min_value, device=device, dtype=dtype)
        # memory部分（前mem_size列）全部可见，置0
        if mem_size > 0:
            causal_mask[:, :mem_size] = 0
        # input_ids部分（后tgt_size列）使用lower triangular
        mask_cond = torch.arange(tgt_size, device=device)
        causal_mask[:, mem_size:].masked_fill_(
            mask_cond < (mask_cond + 1).view(tgt_size, -1), 0
        )
        causal_mask = causal_mask[None, None, ...].expand(batch_size, 1, tgt_size, src_size)
        # 1 for non-padding tokens
        expand_mask = attention_mask[:, None, None, :].expand(batch_size, 1, tgt_size, src_size)
        invert_mask = 1.0 - expand_mask
        invert_mask.masked_fill_(invert_mask.bool(), min_value)

        attention_mask = causal_mask.masked_fill(invert_mask.bool(), min_value)

        if self.config.lychee_memory_attn == "step-expansion":
            # each lychee_memory can attend to one more sub-interval than its predecessor

            if self.config.lychee_memory_pos == "append" and lychee_memory_size > 0:
                window_size = self.config.lychee_memory_window
                window_size_with_lychee_memory = window_size + lychee_memory_size
                lychee_memory_start_idx = -lychee_memory_size
                # batch_size, head_num, window_size
                reference_attention_mask = attention_mask[..., -lychee_memory_size - 1, -window_size_with_lychee_memory: -lychee_memory_size]

                # compression_ratio, 2 * compression_ratio, ..., lychee_memory_size * compression_ratio
                lychee_memory_arange = torch.arange(1, lychee_memory_size + 1, device=device) * compression_ratio
                # 0, 1, 2, ..., window_size - 1
                ordinal_arange = torch.arange(window_size, device=device)
                # lychee_memory_size, window_size
                valid_pos = ordinal_arange.expand(lychee_memory_size, window_size) < lychee_memory_arange.unsqueeze(-1)
                # lychee_memory_size, window_size
                ordinal_attention_mask = torch.where(valid_pos, 0, min_value)
                # NOTE: add reference attention_mask so that padding tokens are considered
                ordinal_attention_mask = ordinal_attention_mask[None, None, ...] + reference_attention_mask.unsqueeze(-2)

                if self.config.lychee_memory_attend_prev:
                    lychee_memory_attention_mask = attention_mask.new_full((lychee_memory_size, lychee_memory_size), min_value).triu(1)
                    # the lychee_memory token is next to the last ordinal token it attends to
                    ordinal_position_ids = position_ids[:, -window_size_with_lychee_memory: -lychee_memory_size]
                    lychee_memory_position_ids = ordinal_position_ids[:, compression_ratio - 1::compression_ratio] + torch.arange(1, lychee_memory_size + 1, device=device)[None]
                    position_ids[:, lychee_memory_start_idx:] = lychee_memory_position_ids
                else:
                    lychee_memory_attention_mask = attention_mask.new_full((lychee_memory_size, lychee_memory_size), min_value).fill_diagonal_(0)
                    # the lychee_memory token is next to the last ordinal token it attends to
                    ordinal_position_ids = position_ids[:, -window_size_with_lychee_memory: -lychee_memory_size]
                    lychee_memory_position_ids = ordinal_position_ids[:, compression_ratio - 1::compression_ratio] + 1
                    position_ids[:, lychee_memory_start_idx:] = lychee_memory_position_ids

                attention_mask[..., lychee_memory_start_idx:, -window_size_with_lychee_memory: -lychee_memory_size] = ordinal_attention_mask
                attention_mask[..., lychee_memory_start_idx:, lychee_memory_start_idx:] = lychee_memory_attention_mask

            # NOTE: the attention mask should be modified when there is lychee_memory token within the window, not in the input_ids
            elif self.config.lychee_memory_pos == "interleave" and (self._lychee_memory_indices == 1).any():
                assert self.config.lychee_memory_attend_prev == False, f"Make sure lychee_memory_attend_prev is False if using 'interleave' lychee_memory pos!"

                lychee_memory_indices = self._lychee_memory_indices

                cur_position_ids = position_ids[:, -len(lychee_memory_indices):]
                base_position = cur_position_ids[:, 0] - 1
                # NOTE: alternate position so that the position of raw tokens are consistent
                position_template = cur_position_ids.new_ones(cur_position_ids.shape)
                position_template[:, compression_ratio + 1::compression_ratio + 1] = 0
                cur_position_ids = base_position + position_template.cumsum(-1)
                position_ids[:, -len(lychee_memory_indices):] = cur_position_ids

                cur_input_length = len(lychee_memory_indices)
                cur_attention_mask = attention_mask[..., -cur_input_length:, -cur_input_length:]
                # mask all lychee_memory columns
                cur_attention_mask[..., lychee_memory_indices] = min_value
                # lychee_memory tokens can attend to themselves
                input_ids_attention_mask = cur_attention_mask[..., -tgt_size:, -tgt_size:]
                input_ids_attention_mask[..., range(tgt_size), range(tgt_size)] = 0

        elif self.config.lychee_memory_attn == "segmentation":
            # each lychee_memory can attend to its corresponding sub-interval

            if self.config.lychee_memory_pos == "append" and lychee_memory_size > 0:
                window_size = self.config.lychee_memory_window
                window_size_with_lychee_memory = window_size + lychee_memory_size
                lychee_memory_start_idx = -lychee_memory_size
                # batch_size, head_num, window_size
                reference_attention_mask = attention_mask[..., -lychee_memory_size - 1, -window_size_with_lychee_memory: -lychee_memory_size]

                # lychee_memory_size, compression_ratio
                indices = torch.arange(compression_ratio * lychee_memory_size, device=device).view(lychee_memory_size, -1)
                # lychee_memory_size, window_size
                ordinal_attention_mask = attention_mask.new_full((lychee_memory_size, window_size), min_value)
                ordinal_attention_mask.scatter_(dim=-1, index=indices, value=0)

                # NOTE: add reference attention_mask so that padding tokens are considered
                ordinal_attention_mask = ordinal_attention_mask[None, None, ...] + reference_attention_mask.unsqueeze(-2)

                if self.config.lychee_memory_attend_prev:
                    lychee_memory_attention_mask = attention_mask.new_full((lychee_memory_size, lychee_memory_size), min_value).triu(1)
                    # the lychee_memory token is next to the last ordinal token it attends to
                    lychee_memory_position_ids = position_ids.new_full(lychee_memory_size, fill_value=compression_ratio + mem_size)
                    lychee_memory_position_ids = lychee_memory_position_ids + torch.arange(lychee_memory_size)
                    position_ids[:, lychee_memory_start_idx:] = lychee_memory_position_ids
                else:
                    lychee_memory_attention_mask = attention_mask.new_full((lychee_memory_size, lychee_memory_size), min_value).fill_diagonal_(0)
                    # the lychee_memory token is next to the last ordinal token it attends to
                    lychee_memory_position_ids = position_ids.new_full(lychee_memory_size, fill_value=compression_ratio + mem_size)
                    position_ids[:, lychee_memory_start_idx:] = lychee_memory_position_ids

                attention_mask[..., lychee_memory_start_idx:, -window_size_with_lychee_memory: -lychee_memory_size] = ordinal_attention_mask
                attention_mask[..., lychee_memory_start_idx:, lychee_memory_start_idx:] = lychee_memory_attention_mask
                # lychee_memorys of different ratios are blind to others
                attention_mask[..., lychee_memory_start_idx:, -lychee_memory_size: lychee_memory_start_idx] = min_value

            elif self.config.lychee_memory_pos == "interleave":
                raise NotImplementedError

        elif self.config.lychee_memory_attn == "full-coverage":
            pass

        return attention_mask, position_ids

    def _extract_lychee_memory_and_raw_memory(
        self, 
        key, 
        value, 
        previous_lychee_memory_key, 
        previous_lychee_memory_value, 
        previous_raw_key, 
        previous_raw_value, 
        lychee_memory_indices,
    ):
        """Extract lychee_memory and raw memory from the returned key and value when the window is full."""
        key = cat_tensor([
            previous_raw_key, 
            key
        ], dim=self.k_seq_dim)
        value = cat_tensor([
            previous_raw_value, 
            value
        ], dim=self.v_seq_dim)

        # NOTE: we use magic slice instead of boolean index here for efficiency
        lychee_memory_key = slice_tensor(key, index=torch.logical_or(lychee_memory_indices == 1, lychee_memory_indices == -1), dim=self.k_seq_dim)
        lychee_memory_key = cat_tensor([previous_lychee_memory_key, lychee_memory_key], dim=self.k_seq_dim)
        lychee_memory_value = slice_tensor(value, index=torch.logical_or(lychee_memory_indices == 1, lychee_memory_indices == -1), dim=self.v_seq_dim)
        lychee_memory_value = cat_tensor([previous_lychee_memory_value, lychee_memory_value], dim=self.v_seq_dim)

        if self._raw_size_to_cache > 0:
            raw_key = slice_tensor(key, index=lychee_memory_indices == 0, dim=self.k_seq_dim)
            raw_key = slice_tensor(raw_key, start=-raw_size_to_cache, dim=self.k_seq_dim)

            raw_value = slice_tensor(value, index=lychee_memory_indices == 0, dim=self.v_seq_dim)
            raw_value = slice_tensor(raw_value, start=-raw_size_to_cache, dim=self.v_seq_dim)        

        else:
            raw_key = None
            raw_value = None

        return lychee_memory_key, lychee_memory_value, raw_key, raw_value


def slice_tensor(x, start=None, end=None, step=None, index=None, dim=2):
    if x is None:
        return None
    if end == 0:
        return None
    if start == x.shape[dim]:
        return None
    if start is not None and start == end:
        return None
    if dim == 2:
        if index is not None:
            return x[:, :, index]
        elif start is None and end is not None:
            if step is None:
                return x[:, :, :end, ...]
            else:
                return x[:, :, :end:step, ...]
        elif start is not None and end is None:
            if step is None:
                return x[:, :, start:, ...]
            else:
                return x[:, :, start::step, ...]
        elif start is not None and end is not None:
            if step is None:
                return x[:, :, start:end, ...]
            else:
                return x[:, :, start:end:step, ...]
    elif dim == 1:
        if index is not None:
            return x[:, :, index]
        elif start is None and end is not None:
            if step is None:
                return x[:, :end, ...]
            else:
                return x[:, :end:step, ...]
        elif start is not None and end is None:
            if step is None:
                return x[:, start:, ...]
            else:
                return x[:, start::step, ...]
        elif start is not None and end is not None:
            if step is None:
                return x[:, start:end, ...]
            else:
                return x[:, start:end:step, ...]
    else:
        raise NotImplementedError

def cat_tensor(list_of_tensors, dim=-1):
    list_of_tensors = [t for t in list_of_tensors if t is not None]
    if len(list_of_tensors) > 1:
        result = torch.cat(list_of_tensors, dim=dim)
    elif len(list_of_tensors) == 1:
        result = list_of_tensors[0]
    else:
        result = None
    return result

def slice_activations(activations, start=None, end=None, k_seq_dim=2, v_seq_dim=2):
    new_activations = []
    for key, value in activations:
        new_key = slice_tensor(key, start=start, end=end, dim=k_seq_dim)
        new_value = slice_tensor(value, start=start, end=end, dim=v_seq_dim)
        new_activations.append([new_key, new_value])
    return new_activations

def cat_activations(list_of_activations, k_seq_dim=2, v_seq_dim=2):
    assert all(len(x) == len(list_of_activations[0]) for x in list_of_activations), f"Make sure all activations have the same number of layers! Found {[len(x) for x in list_of_activations]}."

    new_activations = []
    for layer_idx in range(len(list_of_activations[0])):
        keys = [x[layer_idx][0] for x in list_of_activations]
        values = [x[layer_idx][1] for x in list_of_activations]

        new_key = cat_tensor(keys, dim=k_seq_dim)
        new_value = cat_tensor(values, dim=v_seq_dim)
        new_activations.append([new_key, new_value])
    return new_activations

def interleave_activations(main_activations, augment_activations, main_spans, augment_spans, k_seq_dim=2, v_seq_dim=2, device=torch.device("cuda")):
    """ Interleave main_activations and augment_activations according to main_span and augment_span.

    Args:
        main_span: a list of tuples (start_idx, end_idx). when start_idx and end_idx is None, the augment_activations will be plugged in.
        augment_span: a list of tuples (start_idx, end_idx)
    """
    assert len(main_activations) == len(augment_activations) , f"Make sure main and augment activations have the same number of layers! Found {len(main_activations)} and {len(augment_activations)}!"
    assert sum(x[0] is None and x[1] is None for x in main_spans) == len(augment_spans), f"Make sure the number of slots for augmentation (start_idx=None and end_idx=None in main_spans) matches the number of augmentations. Found {sum(x for x in main_spans if x[0] is None and x[1] is None)} slots but {len(augment_spans)} augmentations!"

    new_activations = []
    for layer_idx in range(len(main_activations)):
        main_key, main_value = main_activations[layer_idx]
        augment_key, augment_value = augment_activations[layer_idx]

        sliced_keys = []
        sliced_values = []

        augment_idx = 0
        for start, end in main_spans:
            if start is None and end is None:
                # this means the augment key/value should be plugged in
                augment_start, augment_end = augment_spans[augment_idx]
                sliced_key = slice_tensor(
                    augment_key, 
                    start=augment_start, 
                    end=augment_end,
                    dim=k_seq_dim
                ).to(device)
                sliced_value = slice_tensor(
                    augment_value, 
                    start=augment_start, 
                    end=augment_end,
                    dim=v_seq_dim
                ).to(device)

            else:
                sliced_key = slice_tensor(
                    main_key, 
                    start=start, 
                    end=end,
                    dim=k_seq_dim
                )
                sliced_value = slice_tensor(
                    main_value, 
                    start=start, 
                    end=end,
                    dim=v_seq_dim
                )

            sliced_keys.append(sliced_key)
            sliced_values.append(sliced_value)

        new_key = cat_tensor(sliced_keys, dim=k_seq_dim)
        new_value = cat_tensor(sliced_values, dim=v_seq_dim)
        new_activations.append([new_key, new_value])

    return new_activations

def softmax(x:np.ndarray, axis=-1, temperature=1):
    if isinstance(x, list):
        x = np.array(x)
    x = x / temperature
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def l1_norm(x):
    sum_x = sum(x)
    x = [y/sum_x for y in x]
    return x

