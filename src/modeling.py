####-------------------------
"""
Notes:

* LatentThinkingCheckpointingCache from src.cache and the patching methods/functions here and in other code in src was originally designed to support gradient checkpointing on LatentThinkingModel during latent thinking. However, although it functioned, it did not result in the expected benefit to memory usage and has thus been disabled.





"""
####-------------------------

import math
import os
from typing import Callable, Optional, Union
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from transformers import LlamaForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

from src.cache import LatentThinkingCheckpointingCache

# from src.gpt2_patches import patch_GPT2Model, unpatch_GPT2Model
# from src.llama_patches import patch_LlamaModel, unpatch_LlamaModel
from src.modeling_utils import (
    RecurrentFilterLinear,
    RecurrentFilterLinearLN,
    RecurrentFilterMLP,
)
from src.settings import _LatentThinkingModelConfig, LatentThinkingModelSettings
from src.utils import FloatingTensor


class LatentThinkingModel(PreTrainedModel, GenerationMixin):
    """
    A wrapper model to implement latent thinking.

    Args:
        config (PretrainedConfig | None): A Hugging Face config for the base model. If None, a base model must be provided.
        base_model (PreTrainedModel | None): An initialized Hugging Face model that supports Causal LM. Must be a supported model type. If None, a config must be provided.
        settings (LatentThinkingModelSettings | None): A settings object for the latent thinking model. If None, a settings object will be instantiated from default values.
        verbose (bool): Whether to print verbose output.


    Note: In the current implementation, <START> and <END> tokens ARE "considered" as part of the model's vocabulary while <|CONTINUE|> and <|STOP|> tokens are not. This is strictly respected throughout the implementation.

    However, the <START> and <END> tokens do not necessarily need to be part of the model's output vocabulary; the current conceptualization of latent thinking only values them as inputs.

    Therefore, under a stricter implementation, only the input embeddings would require the <START> and <END> tokens while only the output embeddings would need the <|CONTINUE|> and <|STOP|> tokens. If the base_model uses weight-tying, this wouldn't matter; both input and output embeddings would contain all four special tokens. But, for models not using weight tying between input and output embeddings, it could be conceived that the <START> and <|CONTINUE|> tokens could be merged, as could the <END> and <|STOP|> tokens.

    However, for now, they are not, and thus it is required that all four tokens have different token ids to alleviate ambiguity during input parsing.

    """

    config_class = _LatentThinkingModelConfig

    # We can't call it base_model because of the base_model @property in PreTrainedModel, which will cause infinite recursion
    base_model_prefix = "_base_model"
    main_input_name = "input_ids"
    _skip_keys_device_placement = ["past_key_values"]
    is_parallelizable = False
    supports_gradient_checkpointing = False  # 9/19/2025: Turned False as gradient checkpointing doesn't seem to help much. Disabled all patching logic for now
    _supports_cache_class = True
    _supports_static_cache = True

    # So... this part is kind of iffy. Basically because these should just belong to the base_model, but super().__init__(...) does checks of the class attributes, so there's no way to forward the values easily
    # Of the two models that are currently supported, Llama supports all of the below and GPT2 supports all of the below except for flex_attn
    # I did a cursory lookthrough and that might not be an issue? HF might handle composite models nicely?
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _supports_flex_attn = True

    SUPPORTED_MODELS = {GPT2LMHeadModel, LlamaForCausalLM}

    resize_embeddings_padding_value = 128

    def __init__(
        self,
        config: PretrainedConfig | None = None,
        base_model: PreTrainedModel | None = None,
        settings: LatentThinkingModelSettings | None = None,
        verbose: bool = True,
    ):
        """
        Args:
            base_model: A Hugging Face causal LM model
        """

        # Only one of config or base_model can be specified, and the one that is not specified is derived from the one that is

        if not ((config is None) ^ (base_model is None)):
            raise ValueError("Exactly one of config or base_model must be specified")

        if config is None:
            config = base_model.config

        super().__init__(config)

        if base_model is None:
            base_model_cls = MODEL_FOR_CAUSAL_LM_MAPPING[type(config)]
            base_model = base_model_cls(config)

        self._base_model = base_model

        if not isinstance(self._base_model, tuple(self.SUPPORTED_MODELS)):
            raise ValueError(
                f"Unsupported model type: {type(self._base_model)}. Currently supported models: {self.SUPPORTED_MODELS}"
            )

        """ Get the LatentThinkingModelSettings """

        if settings is None:
            settings = LatentThinkingModelSettings.pop_from_config(self.config)
            if settings is None:
                warnings.warn(
                    "A LatentThinkingModelSettings instance was not passed as an argument (as settings), nor was it able to be extracted from the config. One will be instaniated from default values"
                )
                settings = LatentThinkingModelSettings()
        else:
            LatentThinkingModelSettings.remove_from_config(self.config)

        settings.validate()

        self.settings = settings

        """ Validate and handle special latent tokens """

        # Check if any special token IDs are None - throw error if so
        unset_tokens = [k for k in LatentThinkingModelSettings.SPECIAL_TOKEN_KEYS if getattr(self.settings, k) is None]

        if unset_tokens:
            raise ValueError(
                f"The following special token IDs are not set in settings: {unset_tokens}. "
                f"All special tokens must be explicitly provided."
            )

        # Get all special token IDs
        special_token_ids = [getattr(self.settings, k) for k in LatentThinkingModelSettings.SPECIAL_TOKEN_KEYS]
        max_token_id = max(special_token_ids)

        if verbose:
            print(f"Special token IDs: {dict(zip(LatentThinkingModelSettings.SPECIAL_TOKEN_KEYS, special_token_ids))}")

        # Check if vocab is large enough to contain all special token IDs
        current_vocab_size = self.vocab_size
        if max_token_id >= current_vocab_size:
            old_vocab_size = current_vocab_size
            min_required_size = max_token_id + 1

            if verbose:
                print(
                    f"Resizing vocab from {old_vocab_size} to at least {min_required_size} to accommodate token IDs up to {max_token_id} with embedding padding of {self.resize_embeddings_padding_value}"
                )

            self.resize_token_embeddings(
                min_required_size,
                pad_to_multiple_of=self.resize_embeddings_padding_value,
                mean_resizing=True,
            )

            # Add all new token IDs to unused_token_ids
            new_token_ids = set(range(old_vocab_size, self.vocab_size))
            self.settings.unused_token_ids.update(new_token_ids)

            # Check for non-sequential token IDs and warn if there are gaps
            if min(special_token_ids) < old_vocab_size:
                # Some special tokens are in the original vocab, check for gaps in the new range
                new_special_tokens = [tid for tid in special_token_ids if tid >= old_vocab_size]
                if new_special_tokens:
                    expected_sequential = list(range(old_vocab_size, old_vocab_size + len(new_special_tokens)))
                    if sorted(new_special_tokens) != expected_sequential:
                        warnings.warn(
                            f"Special token IDs {sorted(new_special_tokens)} are not sequential from vocab size {old_vocab_size}. "
                            f"This will create gaps in the token space."
                        )
            else:
                # All special tokens are new, check if they're sequential from old vocab size
                if sorted(special_token_ids) != list(range(old_vocab_size, old_vocab_size + len(special_token_ids))):
                    warnings.warn(
                        f"Special token IDs {sorted(special_token_ids)} are not sequential from vocab size {old_vocab_size}. "
                        f"This will create gaps in the token space."
                    )

        # Remove special token IDs from unused_token_ids if they're present
        for token_id in special_token_ids:
            if token_id in self.settings.unused_token_ids:
                self.settings.unused_token_ids.remove(token_id)

        """ Instantiate latent thinking modules """

        if self.recurrent_filter_mode is None:
            self.recurrent_filter = None
        elif self.recurrent_filter_mode in ("linear", "linear_ECF"):
            self.recurrent_filter = RecurrentFilterLinear(self.config.hidden_size)
        elif self.recurrent_filter_mode == "linear_ln":
            self.recurrent_filter = RecurrentFilterLinearLN(self.config.hidden_size)
        elif self.recurrent_filter_mode == "SRE":
            self.recurrent_filter = self.softmax_reembedding
        elif self.recurrent_filter_mode == "MLP":
            self.recurrent_filter = RecurrentFilterMLP(self.config.hidden_size)
        else:
            self.settings.validate()
            raise ValueError(
                f"Recurrent filter mode {self.recurrent_filter_mode} is not yet supported in .__init__(...)!"
            )

        self.gradient_checkpointing = False
        self.base_model.gradient_checkpointing_disable()

        """ Enable LoRA mode if specified """

        if self.lora_mode:
            self.lora_mode_enable(ignore_lora_mode_on=True)

        """ Monkey patch model """

        # self.patch_model()

        """ Misc """
        self._generation_mode = False

        """ Post-init """

        self.post_init()

    @property
    def recurrent_filter_mode(self):
        return self.settings.recurrent_filter_mode

    @property
    def start_token_id(self):
        return self.settings.start_token_id

    @property
    def continue_token_id(self):
        return self.settings.continue_token_id

    @property
    def end_token_id(self):
        return self.settings.end_token_id

    @property
    def stop_token_id(self):
        return self.settings.stop_token_id

    @property
    def lm_head(self):
        return self.get_output_embeddings()

    @property
    def input_embeddings(self):
        return self.get_input_embeddings()

    @property
    def vocab_size(self):
        return self.config.vocab_size

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def eos_token_id(self):
        return self.config.eos_token_id

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self.gradient_checkpointing

    @property
    def input_embeddings_device(self):
        return self.input_embeddings.weight.device

    @property
    def input_embeddings_dtype(self):
        return self.input_embeddings.weight.dtype

    @property
    def lm_head_device(self):
        return self.lm_head.weight.device

    @property
    def lm_head_dtype(self):
        return self.lm_head.weight.dtype

    @property
    def _model(self):
        """
        The body of the transformer from the base_model
        """
        if type(self.base_model) is GPT2LMHeadModel:
            return self.base_model.transformer
        elif type(self.base_model) is LlamaForCausalLM:
            return self.base_model.model
        else:
            raise TypeError(
                f"Unsupported model type: {type(self.base_model)}. Currently supported models: ", self.SUPPORTED_MODELS
            )

    def get_input_embeddings(self) -> nn.Module:
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.base_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.base_model.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        return self.base_model.resize_token_embeddings(
            new_num_tokens=new_num_tokens, pad_to_multiple_of=pad_to_multiple_of, mean_resizing=mean_resizing
        )

    def tie_weights(self):
        return self.base_model.tie_weights()

    def _get_name(self):
        return self.base_model._get_name()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        raise NotImplementedError("Gradient checkpointing is not supported on LatentThinkingModel!")

        ## Although gradient checkpointing during latent thinking does work when combined with LatentThinkingCheckpointingCache, I haven't seen the expected benefit from it. So it has been disabled

        # if gradient_checkpointing_kwargs is None:
        #     gradient_checkpointing_kwargs = {"use_reentrant": False}

        # self.gradient_checkpointing = True
        # self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self.base_model.gradient_checkpointing_disable()

    def _generation_mode_enable(self):
        self._generation_mode = True

    def _generation_mode_disable(self):
        self._generation_mode = False

    @property
    def tie_word_embeddings(self) -> bool:
        return self.config.tie_word_embeddings

    @property
    def lora_mode(self) -> bool:
        return self.settings.lora_mode

    def lora_mode_enable(self, ignore_lora_mode_on: bool = False) -> bool:
        """
        Copy the weights from the embeddings over to separate modules, then zero-out the original embeddings.

        The zeroing may be useful as later on we can just add in the missing parts into the embeddings/logits without having to do masking first, as the output will already be 0. Without doing masking, however, we RELY on the embeddings being frozen. So lora_mode CANNOT work if the embeddings have grads on.

        """

        if not ignore_lora_mode_on and self.lora_mode:
            warnings.warn("Lora mode was already enabled! Nothing will be done...")
            return

        if self.tie_word_embeddings:
            self._start_embed = nn.Linear(
                self.hidden_size, 1, bias=False, device=self.input_embeddings_device, dtype=self.input_embeddings_dtype
            )
            self._end_embed = nn.Linear(
                self.hidden_size, 1, bias=False, device=self.input_embeddings_device, dtype=self.input_embeddings_dtype
            )
            self._continue_embed = nn.Linear(
                self.hidden_size, 1, bias=False, device=self.input_embeddings_device, dtype=self.input_embeddings_dtype
            )
            self._stop_embed = nn.Linear(
                self.hidden_size, 1, bias=False, device=self.input_embeddings_device, dtype=self.input_embeddings_dtype
            )

            with torch.no_grad():
                ## Copy weights over
                self._start_embed.weight[0].copy_(self.input_embeddings.weight[self.start_token_id])
                self._end_embed.weight[0].copy_(self.input_embeddings.weight[self.end_token_id])
                self._continue_embed.weight[0].copy_(self.input_embeddings.weight[self.continue_token_id])
                self._stop_embed.weight[0].copy_(self.input_embeddings.weight[self.stop_token_id])

                ## Set weights in embeddings to 0
                self.input_embeddings.weight[self.start_token_id].zero_()
                self.input_embeddings.weight[self.end_token_id].zero_()
                self.input_embeddings.weight[self.continue_token_id].zero_()
                self.input_embeddings.weight[self.stop_token_id].zero_()

        else:
            # Input embeddings
            self._start_embed_in = nn.Linear(
                self.hidden_size, 1, bias=False, device=self.input_embeddings_device, dtype=self.input_embeddings_dtype
            )
            self._end_embed_in = nn.Linear(
                self.hidden_size, 1, bias=False, device=self.input_embeddings_device, dtype=self.input_embeddings_dtype
            )

            # Output embeddings
            whether_outputs_use_bias = self.lm_head.bias is not None
            self._start_embed_out = nn.Linear(
                self.hidden_size, 1, bias=whether_outputs_use_bias, device=self.lm_head_device, dtype=self.lm_head_dtype
            )
            self._continue_embed_out = nn.Linear(
                self.hidden_size, 1, bias=whether_outputs_use_bias, device=self.lm_head_device, dtype=self.lm_head_dtype
            )
            self._stop_embed_out = nn.Linear(
                self.hidden_size, 1, bias=whether_outputs_use_bias, device=self.lm_head_device, dtype=self.lm_head_dtype
            )

            with torch.no_grad():
                ## Copy weights over

                # Input embeddings
                self._start_embed_in.weight[0].copy_(self.input_embeddings.weight[self.start_token_id])
                self._end_embed_in.weight[0].copy_(self.input_embeddings.weight[self.end_token_id])

                # Output embeddings
                self._start_embed_out.weight[0].copy_(self.lm_head.weight[self.start_token_id])
                self._continue_embed_out.weight[0].copy_(self.lm_head.weight[self.continue_token_id])
                self._stop_embed_out.weight[0].copy_(self.lm_head.weight[self.stop_token_id])

                if whether_outputs_use_bias:
                    self._start_embed_out.bias[0].copy_(self.lm_head.bias[self.start_token_id])
                    self._continue_embed_out.bias[0].copy_(self.lm_head.bias[self.continue_token_id])
                    self._stop_embed_out.bias[0].copy_(self.lm_head.bias[self.stop_token_id])

                ## Zero out parameters in actual embeddings
                self.input_embeddings.weight[self.start_token_id].zero_()
                self.input_embeddings.weight[self.end_token_id].zero_()

                self.lm_head.weight[self.start_token_id].zero_()
                self.lm_head.weight[self.continue_token_id].zero_()
                self.lm_head.weight[self.stop_token_id].zero_()

                if whether_outputs_use_bias:
                    self.lm_head.bias[self.start_token_id].zero_()
                    self.lm_head.bias[self.continue_token_id].zero_()
                    self.lm_head.bias[self.stop_token_id].zero_()

        self.settings.lora_mode = True

    def lora_mode_disable(self) -> bool:
        """
        Undo self.lora_mode_enable(...)
        """

        if not self.lora_mode:
            warnings.warn("lora_mode was already off! Nothing to disable.")
            return

        if self.tie_word_embeddings:
            with torch.no_grad():
                ## Copy weights over
                self.input_embeddings.weight[self.start_token_id].copy_(self._start_embed.weight[0])
                self.input_embeddings.weight[self.end_token_id].copy_(self._end_embed.weight[0])
                self.input_embeddings.weight[self.continue_token_id].copy_(self._continue_embed.weight[0])
                self.input_embeddings.weight[self.stop_token_id].copy_(self._stop_embed.weight[0])

            # Delete the extra modules
            del self._start_embed
            del self._end_embed
            del self._continue_embed
            del self._stop_embed

        else:
            whether_outputs_use_bias = self.lm_head.bias is not None

            with torch.no_grad():
                ## Copy weights over

                # Input embeddings
                self.input_embeddings.weight[self.start_token_id].copy_(self._start_embed_in.weight[0])
                self.input_embeddings.weight[self.end_token_id].copy_(self._end_embed_in.weight[0])

                # Output embeddings
                self.lm_head.weight[self.start_token_id].copy_(self._start_embed_out.weight[0])
                self.lm_head.weight[self.continue_token_id].copy_(self._continue_embed_out.weight[0])
                self.lm_head.weight[self.stop_token_id].copy_(self._stop_embed_out.weight[0])

                if whether_outputs_use_bias:
                    self.lm_head.bias[self.start_token_id].copy_(self._start_embed_out.bias[0])
                    self.lm_head.bias[self.continue_token_id].copy_(self._continue_embed_out.bias[0])
                    self.lm_head.bias[self.stop_token_id].copy_(self._stop_embed_out.bias[0])

            del self._start_embed_in
            del self._end_embed_in

            del self._start_embed_out
            del self._continue_embed_out
            del self._stop_embed_out

        self.settings.lora_mode = False

    @property
    def start_embed_in(self) -> nn.Linear:
        if self.tie_word_embeddings:
            return self._start_embed
        else:
            return self._start_embed_in

    @property
    def end_embed_in(self) -> nn.Linear:
        if self.tie_word_embeddings:
            return self._end_embed
        else:
            return self._end_embed_in

    @property
    def start_embed_out(self) -> nn.Linear:
        if self.tie_word_embeddings:
            return self._start_embed
        else:
            return self._start_embed_out

    @property
    def continue_embed_out(self) -> nn.Linear:
        if self.tie_word_embeddings:
            return self._continue_embed
        else:
            return self._continue_embed_out

    @property
    def stop_embed_out(self) -> nn.Linear:
        if self.tie_word_embeddings:
            return self._stop_embed
        else:
            return self._stop_embed_out

    @property
    def binary_head_temp(self) -> float | None:
        return self.settings.binary_head_temp

    @binary_head_temp.setter
    def binary_head_temp(self, temp: float | None):
        self.settings.binary_head_temp = temp
        self.settings.validate()

    def input_embed_forward(self, input_ids: torch.LongTensor) -> FloatingTensor:
        """
        If not self.lora_mode, just does a forward pass on the input embeddings.

        If self.lora_mode is True, does a forward pass and then replaces any instances of the start token or end token with the separate embeddings.
        """

        input_embeds = self.input_embeddings(input_ids)

        if self.lora_mode:
            if self.input_embeddings.weight.requires_grad:
                raise ValueError("If lora_mode is ON the input embeddings CANNOT have requires_grad=True!")

            start_map = input_ids == self.start_token_id
            if start_map.any():
                # Make sure all start embeddings are 0
                with torch.no_grad():
                    start_embeds_mean = input_embeds[start_map].abs().mean()
                    if start_embeds_mean > 1e-8:
                        raise ValueError(
                            f"The average absolute value of starting embeds before replacement with the actual embedding under lora mode was {start_embeds_mean}. It should be 0!"
                        )

                input_embeds = torch.where(start_map.unsqueeze(-1), self.start_embed_in.weight[0], input_embeds)

            end_map = input_ids == self.end_token_id
            if end_map.any():
                # Make sure all end embeddings are 0
                with torch.no_grad():
                    end_embeds_mean = input_embeds[end_map].abs().mean()
                    if end_embeds_mean > 1e-8:
                        raise ValueError(
                            f"The average absolute value of ending embeds before replacement with the actual embedding under lora mode was {end_embeds_mean}. It should be 0!"
                        )

                input_embeds = torch.where(end_map.unsqueeze(-1), self.end_embed_in.weight[0], input_embeds)

        return input_embeds

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        self.settings.insert_in_config(self.config)

        return_value = super().save_pretrained(
            save_directory,
            is_main_process,
            state_dict,
            save_function,
            push_to_hub,
            max_shard_size,
            safe_serialization,
            variant,
            token,
            save_peft_format,
            **kwargs,
        )

        LatentThinkingModelSettings.remove_from_config(self.config)

        return return_value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[FloatingTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if output_attentions is not None:
            raise NotImplementedError("output_attentions is not yet supported in the LatentThinkingModel.forward().")

        if not ((input_ids is None) ^ (inputs_embeds is None)):
            raise ValueError("Exactly one of input_ids or inputs_embeds must be defined")

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if self.settings.disable_input_past_key_values and past_key_values is not None:
            raise ValueError(
                f"disable_input_past_key_values was True but past_key_values was not None! It was of type: {type(past_key_values)}"
            )

        if (
            not isinstance(past_key_values, (LatentThinkingCheckpointingCache, type(None)))
            and use_cache
            and self.is_gradient_checkpointing
        ):
            warnings.warn(
                f"use_cache is not compatible with gradient_checkpointing unless past_key_values is either None or an instance of LatentThinkingCheckpointingCache. But past_key_values was type {type(past_key_values)}. Setting use_cache to False"
            )
            use_cache = False

        # If inputs_embeds are provided, then we do not do latent thinking. Instead, we do a normal forward pass w/ base model
        # This will result in logits that are not properly masked, as we can't tell which embeds require binary head logits and which require actual vocab logits
        if inputs_embeds is not None:
            if self.settings.disable_forward_input_embeds:
                raise ValueError("inputs_embeds are provided but forward with inputs_embeds is disabled in settings")
            return self.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                cache_position=cache_position,
                position_ids=position_ids,
                labels=labels,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        """ Extract previous iteration hidden states from past_key_values if self._generation_mode is True """
        # Refers to the final sequence position of the last_hidden_state tensor of transformer body's output
        final_hidden_state = None
        unfinished_thinking_map = None
        if self._generation_mode:
            if not isinstance(past_key_values, Cache):
                raise ValueError(
                    f"During generation mode past_key_values must be either an instance of Cache class. But got {type(past_key_values)}"
                )
            if past_key_values is not None and past_key_values.get_seq_length():
                if not hasattr(past_key_values, "_LatentThinkingModel_final_hidden_state"):
                    raise ValueError(
                        "self._generation_mode was True, past_key_values was not None AND was not empty, yet the attribute _LatentThinkingModel_final_hidden_state did not exist"
                    )
                if not hasattr(past_key_values, "_LatentThinkingModel_unfinished_thinking_map"):
                    raise ValueError(
                        "self._generation_mode was True, past_key_values was not None AND was not empty, yet the attribute _LatentThinkingModel_unfinished_thinking_map did not exist"
                    )
                final_hidden_state = past_key_values._LatentThinkingModel_final_hidden_state
                unfinished_thinking_map = past_key_values._LatentThinkingModel_unfinished_thinking_map

                if unfinished_thinking_map is None:
                    raise ValueError("unfinished_thinking_map is None")
                if unfinished_thinking_map.any():
                    if final_hidden_state is None:
                        raise ValueError("unfinished_thinking_map.any() but final_hidden_state is None")
                else:
                    final_hidden_state = None

                if final_hidden_state is not None and final_hidden_state.device != input_ids.device:
                    final_hidden_state = final_hidden_state.to(input_ids.device)
                if unfinished_thinking_map.device != input_ids.device:
                    unfinished_thinking_map = unfinished_thinking_map.to(input_ids.device)

        """ Processing input_ids"""

        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D tensor, got {input_ids.dim()}D tensor of shape {input_ids.shape}")

        if input_ids.shape[1] == 0:
            raise ValueError(f"input_ids had a sequence length of zero! input_ids shape: {input_ids.shape}")

        batch_size, input_ids_length = input_ids.shape

        input_ids, binary_map, thinking_map, stop_map, next_unfinished_thinking_map = self._process_input_ids_forward(
            input_ids, unfinished_thinking_map
        )

        """ Post Token Id Processing and Checks """

        if (input_ids == self.stop_token_id).any():
            raise ValueError(
                "Processing complete but input_ids still has <|STOP|> tokens! These should have been replaced with <END> tokens during processing! Something has gone wrong."
            )

        if (unfinished_thinking_map is None or not unfinished_thinking_map.any()) and thinking_map[:, 0].any():
            raise ValueError(
                "thinking_map indicated True at first sequence position yet there was no previous unfinished thinking. This should be impossible!"
            )

        # Basically we sum across the batch dimension and indicate whether at any position there are any elements of the batch doing latent thinking
        # It is a 1d boolean tensor of length input_ids_length
        pos_thinking_map = thinking_map.any(dim=0)

        """ Use LatentThinkingCheckpointingCache if necessary """

        # LatentThinkingCheckpointingCache is necessary when
        #   1) Gradient checkpointing is enabled
        #   2) There will be latent thinking OR use_cache is True
        # Technically it may be okay to not use it when gradient checkpointing is enabled but torch is under no_grad and the model is set to eval model. But this is also fine and much simpler.

        if self.is_gradient_checkpointing and (thinking_map.any() or use_cache):
            if past_key_values is None:
                past_key_values = LatentThinkingCheckpointingCache(
                    disable_update=self.settings.disable_checkpointing_cache_update
                )

            elif not isinstance(past_key_values, LatentThinkingCheckpointingCache):
                raise ValueError(
                    f"The conditions to require LatentThinkingCheckpointingCache were met but past_key_values was not None or LatentThinkingCheckpointingCache type. Instead got {type(past_key_values)}. These conditions are:"
                    "\n\t1) Gradient checkpointing is enabled"
                    "\n\t2) There will be latent thinking OR use_cache is True"
                )

        if (
            self.is_gradient_checkpointing
            and torch.is_grad_enabled()
            and not isinstance(past_key_values, (type(None), LatentThinkingCheckpointingCache))
        ):
            raise ValueError(
                f"When gradient checkpointing is enabled, unless under no_grad the inputed past_key_values must be either None or LatentThinkingCheckpointingCache type, as otherwise a memory leak out of the gradient checkpointed function could result. Got type {type(past_key_values)} for past_key_values"
            )

        """ Dynamic Forward Pass Iterations """

        all_hidden_states = [] if output_hidden_states else None

        final_hidden_states = []  # Will hold tensors of shape [batch_size, _n_positions, hidden_size], where _n_positions may vary between elements of the list. Later they will be concantenated into a single tensor of shape [batch_size, input_ids_length, hidden_size]

        # Sequence position
        position = 0

        while position < input_ids_length:
            # On each iteration, we do a forward pass for at least one sequence position; the current one. We also strive to do more if subsequent positions do not involve any latent thinking

            if position > 0 and past_key_values is None:
                raise ValueError(
                    "In the middle of iterating through input and past_key_values is None, which means previous positions won't be attended to!"
                )

            # Because GPT2 is annoying
            if isinstance(self._model, GPT2Model) and isinstance(past_key_values, tuple):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

            """ Preparing Input """

            # If the current position is latent thinking, we will prepare it and then check if subsequent positions are non-latent thinking
            # If the current position is NOT latent thinking, we will check if subsequent positions are non-latent thinking and then prepare the current position alongside any subsequent ones which are not latent thinking

            is_current_pos_thinking = pos_thinking_map[position]

            # Insert latent thinking hidden states into next input
            if is_current_pos_thinking:
                current_inputs_embeds = self.input_embed_forward(input_ids[:, position : position + 1])

                # 1d tensor binary mask of length batch_size representing batches doing latent thinking
                current_pos_thinking_mask = thinking_map[:, position]
                # 1d tensor of batch indices doing latent thinking
                thinking_indices = current_pos_thinking_mask.nonzero(as_tuple=True)[0]
                # the rows of final_hidden_state that will be used again as latent input
                thinking_input_states = final_hidden_state[current_pos_thinking_mask]

                # Apply recurrent filter
                if self.recurrent_filter is not None:
                    thinking_input_states = self.recurrent_filter(thinking_input_states)

                # Add <END> token embedding to any which represent <|STOP|> sequence positions
                if self.settings.add_latent_to_end and any(stop_map[:, position]):
                    # This map represents the batch elements of thinking_input_states which need to have <END> added to them (ie. they are on <|STOP|> positions)
                    thinking_input_states_stop_map = stop_map[:, position][current_pos_thinking_mask]
                    thinking_input_states_stop_indices = thinking_input_states_stop_map.nonzero(as_tuple=True)[0]
                    end_embed_to_add = self.input_embed_forward(
                        torch.tensor(self.end_token_id, dtype=torch.long, device=input_ids.device)
                    ).expand(len(thinking_input_states_stop_indices), 1, -1)
                    thinking_input_states = thinking_input_states.index_add(
                        0, thinking_input_states_stop_indices, end_embed_to_add
                    )

                current_inputs_embeds = current_inputs_embeds.index_copy(
                    dim=0, index=thinking_indices, source=thinking_input_states.to(current_inputs_embeds.dtype)
                )

            # Check how many following positions do not do latent thinking
            last_non_thinking_pos = position
            while last_non_thinking_pos + 1 < input_ids_length and not pos_thinking_map[last_non_thinking_pos + 1]:
                last_non_thinking_pos += 1

            if is_current_pos_thinking and last_non_thinking_pos > position:
                # We already prepared the current inputs_embeds. Since there are more subsequent non-thinking positions, get their inputs_embeds and concatenate them on
                additional_input_ids = input_ids[:, position + 1 : last_non_thinking_pos + 1]
                additional_inputs_embeds = self.input_embed_forward(additional_input_ids)
                current_inputs_embeds = torch.cat((current_inputs_embeds, additional_inputs_embeds), dim=1)
            elif not is_current_pos_thinking:
                # We did not already make inputs_embeds for the current position, so make them from the current position up to and including last_non_thinking_pos
                current_input_ids = input_ids[:, position : last_non_thinking_pos + 1]
                current_inputs_embeds = self.input_embed_forward(current_input_ids)

            assert current_inputs_embeds.shape[1] == last_non_thinking_pos + 1 - position, (
                f"current_inputs_embeds did not have a sequence length of (last_non_thinking_pos - position + 1). Expected {last_non_thinking_pos - position + 1}, got {current_inputs_embeds.shape[1]}. This should be impossible!"
            )

            # Why not attention_mask[:, position : last_non_thinking_pos + 1] instead? Because this is supposed to be as long as the past_key_values PLUS the current input
            past_kv_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            current_attention_mask = (
                attention_mask[:, : past_kv_len + current_inputs_embeds.shape[1]]
                if attention_mask is not None
                else None
            )

            current_cache_position = None
            if cache_position is not None:
                current_cache_position = cache_position[position : last_non_thinking_pos + 1]

            current_position_ids = None
            if position_ids is not None:
                current_position_ids = position_ids[:, position : last_non_thinking_pos + 1]

            """ Forward Pass """

            is_last_iteration = last_non_thinking_pos + 1 == input_ids_length

            current_output = self._model(
                inputs_embeds=current_inputs_embeds,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=(not is_last_iteration) or use_cache,  # Must use cache if not last iteration
                output_attentions=False,
                output_hidden_states=output_hidden_states,
                cache_position=current_cache_position,
                position_ids=current_position_ids,
            )

            final_hidden_state = current_output.last_hidden_state[:, -1:]
            past_key_values = current_output.past_key_values

            final_hidden_states.append(current_output.last_hidden_state)

            if output_hidden_states:
                all_hidden_states.append(current_output.hidden_states)

            position = last_non_thinking_pos + 1

        """ Process final_hidden_states into logits """

        final_hidden_states = torch.cat(final_hidden_states, dim=1)

        assert final_hidden_states.shape[1] == input_ids_length, (
            f"After forward pass iterations final_hidden_states did not have a sequence length of input_ids_length. Expected: {input_ids_length}\tGot: {final_hidden_states.shape[1]}. This should be impossible!"
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        # Convert final_hidden_states and binary_map to sliced versions
        sliced_final_hidden_states = final_hidden_states[:, slice_indices]
        sliced_binary_map = binary_map[:, slice_indices]

        sliced_logits_length = sliced_final_hidden_states.shape[1]

        # Create logits tensor where we will assemble everything
        logits = torch.zeros(
            (batch_size, sliced_logits_length, self.vocab_size),
            dtype=sliced_final_hidden_states.dtype,
            device=sliced_final_hidden_states.device,
        )

        logits[sliced_binary_map] = self._unembed_binary_head(sliced_final_hidden_states[sliced_binary_map])

        logits[~sliced_binary_map] = self._unembed_actual_vocab(sliced_final_hidden_states[~sliced_binary_map])

        """ Wrap it up """

        # Calculate Loss if Required
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs)
            if not thinking_map.any():
                ## What is this? Well, in the case of DDP, there could be trouble if some parameters are never used.
                # find_unused_parameters=True is A solution to this, but 1) it introduces a good bit of overhead and 2) there are issues when combined with gradient accumulation when the parameter MAY or MAY NOT be used on any given micro batch (https://github.com/pytorch/pytorch/issues/69031)
                # So a cheap solution is to just include the parameters of all potentially unused modules into the computational graph like so
                # This is specifically done here and not in the training loop so that it is wrapped by DDP's __call__
                loss = loss + self.unused_parameter_dummy_loss()

        if output_hidden_states:
            # all_hidden_states is currently a LIST of the hidden_states outputs from each forawrd pass
            # each of these hidden_states is a tuple of tensors of shape (batch, seq, hidden_size). One for each layer in the transformer
            # We unpack then zip all_hidden_states, which makes it an iterable of tuples, each of which contains all of the tensors for a single layer across the forward pass iterations
            # Then we concatenate this across dim 1, which is the seq dim
            # Except if all_hidden_states only contains 1 element (when there was no latent thinking). In this case, we simply extract the one element from all_hidden_states, as using torch.cat on 1 tensor makes an unnecessary copy (thus more memory allocated)
            if len(all_hidden_states) == 1:
                all_hidden_states = all_hidden_states[0]
            else:
                all_hidden_states = tuple(torch.cat(layer_states, dim=1) for layer_states in zip(*all_hidden_states))

        if self._generation_mode and past_key_values is not None:
            past_key_values._LatentThinkingModel_final_hidden_state = final_hidden_state
            past_key_values._LatentThinkingModel_unfinished_thinking_map = next_unfinished_thinking_map

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )

    def _process_input_ids_forward(
        self,
        input_ids: torch.LongTensor,
        unfinished_thinking_map: torch.BoolTensor | None,
    ) -> tuple[torch.LongTensor, torch.BoolTensor, torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
        """
        Process input_ids. Accomplishes the following (for each row of input_ids):

            * Check whether a latent thinking sequence exists, and ensure that the form of the sequence is correct.
                * Generally, latent thinking sequences take the form: [other tokens, <START>, <|CONTINUE|>, <|CONTINUE|>, ..., <|CONTINUE|>, <|STOP|>, other tokens] where the other tokens can be of length 0
                * [other tokens, <START>, <|CONTINUE|>, <|CONTINUE|>, ..., <|CONTINUE|>] is also valid, representing a latent thinking chain that did not end before cutoff
                * [<|CONTINUE|>, <|CONTINUE|>, ..., <|CONTINUE|>, <|STOP|>, other tokens] is also valid IF unfinished_thinking_map is True for the respective row, since this indicates a continuation of the previously started latent thinking process
                    * As a natural consequence, [<|CONTINUE|>, <|CONTINUE|>, ..., <|CONTINUE|>] is also valid IF unfinished_thinking_map is True for the respective row
                * <START>, <|CONTINUE|>, and <|STOP|> tokens may ONLY appear within valid latent thinking sequences and in their expected position(s)
                * <END> should not appear at all
            * Replaces the <|STOP|> token with an <END> token (because although we want the model to predict <|STOP|>, we want the next input to be an <END>)

        Args:
            input_ids (torch.LongTensor): 2d tensor of input_ids of shape [batch_size, input_ids_length]
            unfinished_thinking_map (torch.BoolTensor | None): 1 tensor of length batch_size, indicating whether that row was mid-latent thinking. Intended to be used with past_key_values. None is the same as passing in a tensor of all False

        Returns:
            input_ids (torch.LongTensor: The modified input_ids.

            binary_map (torch.BoolTensor): A 2d boolean tensor with same shape as input_ids which indicates which tokens require binary head unembedding (_unembed_binary_head) as opposed to actual vocab unembedding (_unembed_actual_vocab). Equivalent to the positions of all <START> and <|CONTINUE|> tokens.

            thinking_map (torch.BoolTensor): A 2d boolean tensor with same shape as input_ids which indicates which tokens will require latent input instead of normal input from the input embeddings. If self.settings.add_latent_to_end is True, then it is equivalent to the positions of all <|CONTINUE|> and <|STOP|> (later converted to <END>) tokens. If False, then it is just equivalent to the positions of all the <|CONTINUE|> tokens

            stop_map (torch.BoolTensor): A 2D boolean tensor with same shape as input_ids which indicates the position of all <|STOP|> (later converted to <END>) tokens. Useful because if self.settings.add_latent_to_end is True, then at the <END> tokens at the end of latent thinking sequences, the latent state is summed with the <END> token before input.

            new_unfinished_thinking_map (torch.BoolTensor): 1 tensor of length batch_size, indicating whether each row of the inputs was mid-latent thinking at the end of the sequence. A row is considered to be mid-latent thinking if its last sequence input was either a <START> or a <|CONTINUE|>

        """

        device = input_ids.device

        batch_size, input_ids_length = input_ids.shape

        # Check for no <END> tokens
        if (input_ids == self.end_token_id).any():
            raise ValueError(f"input_ids contains {(input_ids == self.end_token_id).sum().item()} <END> tokens")

        """ Create maps """

        if unfinished_thinking_map is None:
            unfinished_thinking_map = torch.zeros(batch_size, dtype=bool, device=device)
        else:
            if unfinished_thinking_map.dim() != 1:
                raise ValueError(f"unfinished_thinking_map must be 1D, got {unfinished_thinking_map.dim()}D")
            if len(unfinished_thinking_map) != batch_size:
                raise ValueError(
                    f"unfinished_thinking_map must be of length {batch_size}, got {len(unfinished_thinking_map)}"
                )
            unfinished_thinking_map = unfinished_thinking_map.to(device=device, dtype=torch.bool)

        start_map = input_ids == self.start_token_id
        continue_map = input_ids == self.continue_token_id
        stop_map = input_ids == self.stop_token_id
        continue_or_stop_map = continue_map | stop_map

        """ Iterate through sequence positions to check for correct form """

        is_thinking = unfinished_thinking_map
        for pos in range(input_ids_length):
            # Check if all thinking sequences have a <|CONTINUE|> or <|STOP|> next
            # AND if all non-thinking sequences do NOT have a <|CONTINUE|> or <|STOP|> next
            if not torch.equal(is_thinking, continue_or_stop_map[:, pos]):
                mismatch = is_thinking ^ continue_or_stop_map[:, pos]
                if mismatch.any():
                    # thinking but not <|CONTINUE|> or <|STOP|>
                    bad_thinking = is_thinking & ~continue_or_stop_map[:, pos]
                    # not thinking but <|CONTINUE|> or <|STOP|>
                    bad_non_thinking = ~is_thinking & continue_or_stop_map[:, pos]
                    raise ValueError(
                        f"Invalid tokens at position {pos}: "
                        f"thinking but not <|CONTINUE|> or <|STOP|> = {bad_thinking.nonzero(as_tuple=True)[0].tolist()}, "
                        f"not thinking but <|CONTINUE|> or <|STOP|> = {bad_non_thinking.nonzero(as_tuple=True)[0].tolist()}"
                    )
            # Thinking sequences that have reached a <|STOP|> are no longer thinking sequences
            # Similarly, non-thinking sequences that have reached a <START> now ARE thinking sequences
            is_thinking = (is_thinking & ~stop_map[:, pos]) | start_map[:, pos]

        assert torch.equal(is_thinking, start_map[:, -1] | continue_map[:, -1]), "is_thinking does not match expected"

        """ Replace <|STOP|> tokens with <END> tokens """
        input_ids = torch.where(stop_map, self.end_token_id, input_ids)

        """ Create return maps """

        binary_map = start_map | continue_map

        if self.settings.add_latent_to_end:
            thinking_map = continue_map | stop_map
        else:
            thinking_map = continue_map

        return input_ids, binary_map, thinking_map, stop_map, is_thinking

    def _unembed_binary_head(self, x: torch.Tensor, dtype=None) -> torch.Tensor:
        """
        Performs a forward pass on the binary head, then inserts the binary head logits into logits for the entire output vocabulary with all other logits set to masked values

        Args:
            x (torch.Tensor): An input tensor representing final hidden states of the model with shape [..., hidden_size]

            dtype (torch.dtype): dtype of the resulting operations. If None, will use the dtype of x

        Returns:
            torch.Tensor: A tensor of shape [..., vocab_size], representing the logits with everything but binary head masked.
        """

        binary_logits = self.binary_head(x)

        logits_shape = list(x.shape)
        logits_shape[-1] = self.vocab_size

        logits = torch.full(logits_shape, -float("inf"), dtype=dtype if dtype is not None else x.dtype, device=x.device)

        if binary_logits.dtype != logits.dtype:
            binary_logits = binary_logits.to(logits.dtype)

        binary_head_indices = (self.continue_token_id, self.stop_token_id)

        logits[..., binary_head_indices] = binary_logits

        return logits

    def _unembed_actual_vocab(self, x: torch.Tensor, mask_unused_tokens: bool = False) -> torch.Tensor:
        """
        Performs a forward pass on the lm head, then masks out the binary head tokens. If self.lora_mode, will additionally replace the logit corresponding to the start token with the logit calculated from the actual embedding.

        Args:
            x (torch.Tensor): An input tensor representing final hidden states of the model with shape [..., hidden_size]
            mask_unused_tokens (bool): If True, the unused tokens will be masked out

        Returns:
            torch.Tensor: A tensor of shape [..., vocab_size], representing the logits with the binary head tokens masked. And start token logit replacement if self.lora_mode.
        """

        logits = self.lm_head(x)

        if self.lora_mode:
            if self.lm_head.weight.requires_grad:
                raise ValueError("If lora_mode is ON the output embeddings CANNOT have requires_grad=True!")

            # Make sure the current logit value for the start token is 0
            with torch.no_grad():
                start_logit_mean = logits[..., self.start_token_id].abs().mean()
                if start_logit_mean > 1e-8:
                    raise ValueError(
                        f"The average absolute value of start token logits before replacement with the actual logit under lora mode was {start_logit_mean}. It should be 0!"
                    )

            # Create a mask the same shape as logits. Then replace the logit for the start logit with the real value from the actual start unembedding
            mask = torch.zeros_like(logits)
            mask[..., self.start_token_id : self.start_token_id + 1] = self.start_embed_out(x)
        else:
            # mask with broadcasting
            mask = torch.zeros(self.vocab_size, dtype=logits.dtype, device=x.device)

        # Mask out the <|CONTINUE|> and <|STOP|> tokens
        mask[..., self.continue_token_id] = -float("inf")
        mask[..., self.stop_token_id] = -float("inf")
        # Also mask <END> since model is not supposed to generate it
        mask[..., self.end_token_id] = -float("inf")

        # Mask out unused tokens if specified
        if mask_unused_tokens and self.settings.unused_token_ids:
            mask[..., list(self.settings.unused_token_ids)] = -float("inf")

        logits = logits + mask

        # Note: do not sanitize here; temperature/warpers are applied outside the model

        return logits

    def binary_head(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the binary head using the weights from the output embeddings for the <|CONTINUE|> and <|STOP|> tokens

        Args:
            x (torch.Tensor): A hidden state of shape [..., hidden_size]

        Returns:
            torch.Tensor: A tensor of shape [..., 2], representing the logits of the binary head (in order <|CONTINUE|>, <|STOP|>). These logits are scaled by self.binary_head_temp if it is set

        """

        if self.settings.detach_binary_head_inputs:
            x = x.detach()

        if self.lora_mode:
            continue_logits = self.continue_embed_out(x)
            stop_logits = self.stop_embed_out(x)
            binary_logits = torch.cat([continue_logits, stop_logits], dim=-1)
        else:
            binary_head_indices = [self.continue_token_id, self.stop_token_id]
            binary_weight = self.lm_head.weight[binary_head_indices]
            binary_bias = self.lm_head.bias[binary_head_indices] if self.lm_head.bias is not None else None

            binary_logits = F.linear(x, binary_weight, binary_bias)

        if self.binary_head_temp is not None and self.binary_head_temp != 1.0:
            binary_logits = binary_logits / self.binary_head_temp

        return binary_logits

    @torch.no_grad()
    def _init_weights(self, module):
        """
        PreTrainedModel.initialize_weights(self) defines a 'smart_apply' function to replace the usual 'apply' of torch. Basically, if it finds a sub-module of any module that is also a PreTrainedModel, it will use the _initialize_weights(...) (and therefore _init_weights(...)) method belonging to that submodule, so the base_model will still correctly have its _init_weights applied.

        So anything here only needs to target parameters belonging to LatentThinkingModel but not its base_model
        """

        if isinstance(self.recurrent_filter, nn.Module) and any(module is m for m in self.recurrent_filter.modules()):
            if self.recurrent_filter_mode == "linear_ECF":
                if isinstance(module, nn.Linear):
                    W = self._get_input_embedding_data_without_unused_tokens()
                    V, _ = W.shape

                    module.weight.copy_((W.t() @ W) / V)
                    if module.bias is not None:
                        module.bias.zero_()
                elif isinstance(module, nn.RMSNorm):
                    W = self._get_input_embedding_data_without_unused_tokens()
                    _, d = W.shape

                    # L is average embedding length
                    L = W.norm(dim=1).mean().item()

                    module.weight.fill_(L / math.sqrt(d))
                elif isinstance(module, RecurrentFilterLinear):
                    module.bias.zero_()
                else:
                    raise ValueError(
                        f"module of type {type(module)} does not seem to be supported for linear_ECF initialization"
                    )
            elif self.recurrent_filter_mode == "MLP":
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        module.bias.zero_()
            elif self.recurrent_filter_mode in ["SRE", "linear", "linear_ln"]:
                pass
                # No init needed
            else:
                self.settings.validate()
                raise ValueError(
                    f"Recurrent filter mode {self.recurrent_filter_mode} not yet supported in ._init_weights(...)"
                )

    def _get_input_embedding_data_without_unused_tokens(self):
        # Get embedding weight
        # Shape (vocab, hidden_size)
        W = self.input_embeddings.weight.data

        # Take a slice to exclude all parts of the embedding corresponding to unused tokens
        used_token_mask = torch.ones(W.size(0), dtype=torch.bool, device=W.device)
        used_token_mask[list(self.settings.unused_token_ids)] = False

        W = W[used_token_mask]

        return W

    # def patch_model(self, strict: bool = True):
    #     if isinstance(self._model, GPT2Model):
    #         patch_GPT2Model(self._model, strict=strict)
    #     elif isinstance(self._model, LlamaModel):
    #         patch_LlamaModel(self._model, strict=strict)
    #     else:
    #         raise ValueError(f"Model of type {type(self._model)} is not supported!")

    # def unpatch_model(self, strict: bool = True):
    #     if isinstance(self._model, GPT2Model):
    #         unpatch_GPT2Model(self._model, strict=strict)
    #     elif isinstance(self._model, LlamaModel):
    #         unpatch_LlamaModel(self._model, strict=strict)
    #     else:
    #         raise ValueError(f"Model of type {type(self._model)} is not supported!")

    def softmax_reembedding(self, x: torch.Tensor) -> torch.Tensor:
        ## NOTICE: In the case where LoRA is used to fine-tune the input embeddings via an adapter, this will not work properly, as the .weight will NOT include the changes borne by the adapter

        logits = self._unembed_actual_vocab(x, mask_unused_tokens=True)

        with torch.autocast(device_type=x.device.type, enabled=False):
            # Compute softmax in fp32 for stability under autocast
            probs_fp32 = torch.softmax(logits, dim=-1, dtype=torch.float32)
            # Reembed the probs back to hidden space
            # probs shape: [..., vocab_size]
            # input_embeddings.weight shape: [vocab_size, hidden_size]
            reembedded_fp32 = probs_fp32 @ self.input_embeddings.weight.float()
        # cast back to original dtype
        return reembedded_fp32.to(x.dtype)

    def generate(self, *args, **kwargs):
        was_generation_mode = self._generation_mode
        if not was_generation_mode:
            self._generation_mode_enable()
        try:
            return super().generate(*args, **kwargs)
        finally:
            if not was_generation_mode:
                self._generation_mode_disable()

    def unused_parameter_dummy_loss(self) -> FloatingTensor:
        dummy_loss = 0

        dummy_input = torch.zeros(
            (1, self.hidden_size), dtype=self.input_embeddings_dtype, device=self.input_embeddings_device
        )

        if isinstance(self.recurrent_filter, nn.Module):
            dummy_loss = dummy_loss + self.recurrent_filter(dummy_input).sum()

        if self.lora_mode:
            for k in ["start_embed_in", "end_embed_in", "start_embed_out", "continue_embed_out", "stop_embed_out"]:
                module = getattr(self, k)
                dummy_loss = dummy_loss + module(dummy_input).sum()

        dummy_loss = 0 * dummy_loss

        return dummy_loss
