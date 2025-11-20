from typing import Callable, Optional, Tuple, Unpack

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils.generic import can_return_tuple

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    logger,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


from src.cache import LatentThinkingCheckpointingCache
from src.patches import get_patch_functions


def _LlamaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

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

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    ### EDIT START
    ## If past_key_value was an instance of LatentThinkingCheckpointingCache, return the cache input kwargs too
    if isinstance(past_key_value, LatentThinkingCheckpointingCache):
        return attn_output, attn_weights, cache_update_kwargs
    ### EDIT END

    return attn_output, attn_weights


def _LlamaDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    ### EDIT START
    ## Collects cache_update_kwargs from attn if past_key_value is LatentThinkingCheckpointingCache instance

    # Self Attention
    _attn_outputs = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )

    if isinstance(past_key_value, LatentThinkingCheckpointingCache):
        hidden_states, self_attn_weights, cache_update_kwargs = _attn_outputs
    else:
        hidden_states, self_attn_weights = _attn_outputs

    ### EDIT END

    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)

    ### EDIT START
    ## If past_key_value was an instance of LatentThinkingCheckpointingCache, return the cache_update_kwargs too
    if isinstance(past_key_value, LatentThinkingCheckpointingCache):
        return outputs, cache_update_kwargs
    ### EDIT END

    return outputs


@can_return_tuple
def _LlamaModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
) -> BaseModelOutputWithPast:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    ### EDIT START
    ## The below was edited. use_cache is not longer incompatible with gradient checkpointing if the cache being used is an instance of LatentThinkingCheckpointingCache

    if not isinstance(past_key_values, LatentThinkingCheckpointingCache):
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing WHEN past_key_values IS NOT AN INSTANCE OF LatentThinkingCheckpointingCache. Setting `use_cache=False`..."
            )
            use_cache = False

    ### EDIT END

    # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **flash_attn_kwargs,
        )

        ### EDIT START
        ## If past_key_values is an instance of LatentThinkingCheckpointingCache, extract cache_update_kwargs from the outputs and use it to copy_add past_key_values
        if isinstance(past_key_values, LatentThinkingCheckpointingCache):
            layer_outputs, cache_update_kwargs = layer_outputs
            # Don't do the update if use_cache is False...
            # If gradient_checkpointing is enabled, this prevents new KV values from being saved in memory, which would otherwise occur because the gradient checkpointing func will hold onto all inputs
            if use_cache:
                past_key_values = past_key_values.copy_add(**cache_update_kwargs)
        ### EDIT END

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


patch_LlamaModel, unpatch_LlamaModel = get_patch_functions(
    model_class=LlamaModel,
    block_class=LlamaDecoderLayer,
    attn_class=LlamaAttention,
    model_forward_patch=_LlamaModel_forward,
    block_forward_patch=_LlamaDecoderLayer_forward,
    attn_forward_patch=_LlamaAttention_forward,
)
