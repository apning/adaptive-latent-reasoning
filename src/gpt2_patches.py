from typing import Callable, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils.deprecation import deprecate_kwarg

from src.cache import LatentThinkingCheckpointingCache

from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2Block,
    GPT2Model,
    logger,
    eager_attention_forward,
    _prepare_4d_attention_mask_for_sdpa,
)

from src.patches import get_patch_functions


@deprecate_kwarg("layer_past", new_name="past_key_value", version="4.53.0", raise_if_both_names=True)
def _GPT2Attention_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = False,
    **kwargs,
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
    is_cross_attention = encoder_hidden_states is not None
    if is_cross_attention:
        if not hasattr(self, "q_attn"):
            raise ValueError(
                "If class is used as cross attention, the weights `q_attn` have to be defined. "
                "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
            )

        query_states = self.q_attn(hidden_states)
        key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
        attention_mask = encoder_attention_mask
    else:
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

    shape_q = (*query_states.shape[:-1], -1, self.head_dim)
    shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

    query_states = query_states.view(shape_q).transpose(1, 2)
    key_states = key_states.view(shape_kv).transpose(1, 2)
    value_states = value_states.view(shape_kv).transpose(1, 2)

    if past_key_value is not None:
        if isinstance(past_key_value, EncoderDecoderCache):
            if is_cross_attention:
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache
        cache_kwargs = {"cache_position": cache_position}

        ### EDIT START
        ## Edited so that if past_key_value is an instance of LatentThinkingCheckpointingCache, it calls .get_new_kv instead of .update
        if isinstance(past_key_value, LatentThinkingCheckpointingCache):
            cache_update_kwargs = {
                "key_states": key_states,
                "value_states": value_states,
                "layer_idx": self.layer_idx,
                "cache_kwargs": cache_kwargs,
            }
            key_states, value_states = past_key_value.get_new_kv(**cache_update_kwargs)
        else:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs=cache_kwargs
            )
        ### EDIT END

    is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

    using_eager = self.config._attn_implementation == "eager"
    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):
            using_eager = True
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            # Attention functions are consistent with previous equivalent attention classes, however they do not support some options
            # (e.g. layer scaling, head mask) that eager supports. These implementations are thus equivalent to previous code, but
            # not necessarily to eager (if mentioned options are provided).
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    if using_eager and self.reorder_and_upcast_attn:
        attn_output, attn_weights = self._upcast_and_reordered_attn(
            query_states, key_states, value_states, attention_mask, head_mask
        )
    else:
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            head_mask=head_mask,
            dropout=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
            **kwargs,
        )

    attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    ### EDIT START
    ## If past_key_value was an instance of LatentThinkingCheckpointingCache, return the cache input kwargs too
    if isinstance(past_key_value, LatentThinkingCheckpointingCache):
        return attn_output, attn_weights, cache_update_kwargs
    ### EDIT END

    return attn_output, attn_weights


@deprecate_kwarg("layer_past", new_name="past_key_value", version="4.53.0", raise_if_both_names=True)
def _GPT2Block_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    **kwargs,
) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)

    ### EDIT START
    ## Collects cache_update_kwargs from attn if past_key_value is LatentThinkingCheckpointingCache instance

    _attn_outputs = self.attn(
        hidden_states,
        past_key_value=past_key_value,
        cache_position=cache_position,
        attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        **kwargs,
    )

    if isinstance(past_key_value, LatentThinkingCheckpointingCache):
        attn_output, self_attn_weights, cache_update_kwargs = _attn_outputs
    else:
        attn_output, self_attn_weights = _attn_outputs

    ### EDIT END

    # residual connection
    hidden_states = attn_output + residual

    if encoder_hidden_states is not None:
        # add one self-attention block for cross-attention
        if not hasattr(self, "crossattention"):
            raise ValueError(
                f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                "cross-attention layers by setting `config.add_cross_attention=True`"
            )
        residual = hidden_states
        hidden_states = self.ln_cross_attn(hidden_states)
        cross_attn_output, cross_attn_weights = self.crossattention(
            hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        # residual connection
        hidden_states = residual + cross_attn_output

    residual = hidden_states
    hidden_states = self.ln_2(hidden_states)
    feed_forward_hidden_states = self.mlp(hidden_states)
    # residual connection
    hidden_states = residual + feed_forward_hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
        if encoder_hidden_states is not None:
            outputs += (cross_attn_weights,)

    ### EDIT START
    ## If past_key_value was an instance of LatentThinkingCheckpointingCache, return the cache_update_kwargs too
    if isinstance(past_key_value, LatentThinkingCheckpointingCache):
        return outputs, cache_update_kwargs
    ### EDIT END

    return outputs


def _GPT2Model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Tuple[Tuple[torch.Tensor]], Cache]] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
    r"""
    input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
        `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
        `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
        sequence tokens in the vocabulary.

        If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
        `input_ids`.

        Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
        [`PreTrainedTokenizer.__call__`] for details.

        [What are input IDs?](../glossary#input-ids)
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])

    ### EDIT START
    ## The below was edited. use_cache is not longer incompatible with gradient checkpointing if the cache being used is an instance of LatentThinkingCheckpointingCache

    if not isinstance(past_key_values, LatentThinkingCheckpointingCache):
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing WHEN past_key_values IS NOT AN INSTANCE OF LatentThinkingCheckpointingCache. Setting `use_cache=False`..."
            )
            use_cache = False

    ### EDIT END

    # based on pattern from src/transformers/models/whisper/modeling_whisper.py::WhisperDecoder
    return_legacy_cache = False
    if use_cache:
        if past_key_values is None:
            return_legacy_cache = True
            past_key_values = DynamicCache()
        elif not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            logger.warning_once(
                "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.53.0. "
                "You should pass an instance of `Cache` instead, e.g. "
                "`past_key_values=DynamicCache.from_legacy_cache(past_key_values)`."
            )
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if self.config.add_cross_attention and not isinstance(past_key_values, EncoderDecoderCache):
            past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    position_embeds = self.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

    # Attention mask.
    # ._update_causal_mask() and ._prepare_4d_causal_attention_mask_with_cache_position() copied from LlamaModel
    if attention_mask is not None and attention_mask.ndim < 4:
        attention_mask = attention_mask.view(batch_size, -1)
    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
    if self.config.add_cross_attention and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        if _use_sdpa:
            encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
            )
        elif not self._attn_implementation == "flash_attention_2":
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # head_mask has shape n_layer x batch x n_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    if token_type_ids is not None:
        token_type_embeds = self.wte(token_type_ids)
        hidden_states = hidden_states + token_type_embeds

    hidden_states = self.drop(hidden_states)

    output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
    all_hidden_states = () if output_hidden_states else None
    for i, block in enumerate(self.h):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(hidden_states.device)
            # Ensure that attention_mask is always on the same device as hidden_states
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)
            if isinstance(head_mask, torch.Tensor):
                head_mask = head_mask.to(hidden_states.device)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            outputs = self._gradient_checkpointing_func(
                block.__call__,
                hidden_states,
                past_key_values,
                cache_position,
                causal_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
                use_cache,
                output_attentions,
            )

        else:
            outputs = block(
                hidden_states,
                past_key_value=past_key_values,
                cache_position=cache_position,
                attention_mask=causal_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

        ### EDIT START
        ## If past_key_values is an instance of LatentThinkingCheckpointingCache, extract cache_update_kwargs from the outputs and use it to copy_add past_key_values
        if isinstance(past_key_values, LatentThinkingCheckpointingCache):
            outputs, cache_update_kwargs = outputs
            # Don't do the update if use_cache is False...
            # If gradient_checkpointing is enabled, this prevents new KV values from being saved in memory, which would otherwise occur because the gradient checkpointing func will hold onto all inputs
            if use_cache:
                past_key_values = past_key_values.copy_add(**cache_update_kwargs)
        ### EDIT END

        hidden_states = outputs[0]

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (outputs[2],)

        # Model Parallel: If it's the last layer for that device, put things on the next device
        if self.model_parallel:
            for k, v in self.device_map.items():
                if i == v[-1] and "cuda:" + str(k) != self.last_device:
                    hidden_states = hidden_states.to("cuda:" + str(k + 1))

    hidden_states = self.ln_f(hidden_states)

    hidden_states = hidden_states.view(output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    past_key_values = past_key_values if use_cache else None
    if return_legacy_cache:
        past_key_values = (
            past_key_values.self_attention_cache.to_legacy_cache()
            if self.config.add_cross_attention
            else past_key_values.to_legacy_cache()
        )
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, past_key_values, all_hidden_states, all_self_attentions, all_cross_attentions]
            if v is not None
        )

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )


patch_GPT2Model, unpatch_GPT2Model = get_patch_functions(
    model_class=GPT2Model,
    block_class=GPT2Block,
    attn_class=GPT2Attention,
    model_forward_patch=_GPT2Model_forward,
    block_forward_patch=_GPT2Block_forward,
    attn_forward_patch=_GPT2Attention_forward,
)
