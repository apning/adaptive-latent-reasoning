from dataclasses import asdict, dataclass
from typing import Union
import warnings

from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


from src.modeling import LatentThinkingModel

import os

from src.modeling_utils import (
    add_special_latent_tokens_to_tokenizer,
    ensure_tokenizer_has_latent_tokens,
    get_special_tokens_mapping_from_tokenizer,
)
from src.settings import LatentThinkingModelSettings
from src.utils import JsonMixin, STR_TO_TORCH_DTYPE


def hf_repo_exists(repo_id: str) -> bool:
    api = HfApi()
    return api.repo_exists(repo_id)


@dataclass
class ModelCreationConfig(JsonMixin):
    """
    Contains everything needed to get a model and its tokenizer
    """

    path_or_repo: str = None
    dtype: str | None = None

    is_latent: bool = False
    latent_settings_dict: dict | None = None

    is_lora: bool = False
    lora_config_dict: dict | None = None
    lora_path_or_repo: str | None = None

    def __post_init__(self):
        if isinstance(self.lora_config_dict, LoraConfig):
            self.lora_config_dict = asdict(self.lora_config_dict)

        self.validate()

    def validate(self):
        # require path_or_repo
        if self.path_or_repo is None:
            raise ValueError("path_or_repo is required")

        # Make sure dtype is valid if specified
        if self.dtype is not None and self.dtype not in STR_TO_TORCH_DTYPE.keys():
            raise ValueError(f"dtype must be one of {STR_TO_TORCH_DTYPE.keys()}, got {self.dtype}")

        # only specify latent_settings_dict if is_latent
        if not self.is_latent and self.latent_settings_dict is not None:
            raise ValueError(
                f"latent_settings_dict must be None if is_latent is False. Got latent_settings_dict = {self.latent_settings_dict}"
            )

        # Only specify lora_path_or_repo or lora_config_dict if is_lora
        if not self.is_lora and (self.lora_path_or_repo is not None or self.lora_config_dict is not None):
            raise ValueError(
                f"lora_path_or_repo and lora_config_dict must be None if is_lora is False. Got lora_path_or_repo = {self.lora_path_or_repo}, lora_config_dict = {self.lora_config_dict}"
            )
        # If is_lora is True, exactly one of lora_path_or_repo or lora_config_dict must be specified
        if self.is_lora and not ((self.lora_path_or_repo is not None) ^ (self.lora_config_dict is not None)):
            raise ValueError(
                f"If is_lora is True, exactly one of lora_path_or_repo or lora_config_dict must be specified. Got lora_path_or_repo = {self.lora_path_or_repo}, lora_config_dict = {self.lora_config_dict}"
            )

    def get_model_and_tokenizer(self) -> tuple[PreTrainedModel | PeftModel, PreTrainedTokenizerBase]:
        model = automodelforcausallm_from_pretrained_latent(self.path_or_repo)
        tokenizer = AutoTokenizer.from_pretrained(self.path_or_repo)

        """ Make the model latent if necessary """

        if self.is_latent:
            if not isinstance(model, LatentThinkingModel):
                """ Create latent thinking settings """

                if self.latent_settings_dict is None:
                    raise ValueError(
                        "latent_settings_dict is required if is_latent is True and loaded model is not a LatentThinkingModel"
                    )
                latent_thinking_settings = LatentThinkingModelSettings(**self.latent_settings_dict)
                # Clear unused token ids
                latent_thinking_settings.unused_token_ids = set()

                """ Add special tokens to tokenizer and get special tokens to id mapping """

                if ensure_tokenizer_has_latent_tokens(tokenizer, raise_error=False):
                    special_tokens_mapping = get_special_tokens_mapping_from_tokenizer(tokenizer)
                else:
                    special_tokens_mapping = add_special_latent_tokens_to_tokenizer(tokenizer)

                """ If latent_thinking_settings already had special token ids set, make sure they are equal to the ids in the tokenizer. If they were not already set, add them to settings """

                if latent_thinking_settings.are_special_tokens_set(strict=True):
                    assert latent_thinking_settings.get_special_tokens_to_ids_dict() == special_tokens_mapping, (
                        f"Special tokens to ids dict from tokenizer does not match special tokens to ids dict from latent thinking settings. Got: {latent_thinking_settings.get_special_tokens_to_ids_dict()} != {special_tokens_mapping}"
                    )
                else:
                    latent_thinking_settings.add_special_tokens(special_tokens_mapping)

                """ Initialize LatentThinkingModel """

                model = LatentThinkingModel(base_model=model, settings=latent_thinking_settings)

            else:
                if self.latent_settings_dict is not None:
                    raise ValueError(
                        f"latent_settings_dict must not be specified if loaded model is already a LatentThinkingModel. But got latent_settings_dict = {self.latent_settings_dict}"
                    )
                if not ensure_tokenizer_has_latent_tokens(tokenizer, raise_error=False):
                    raise ValueError(
                        "Loaded model was already a LatentThinkingModel, but tokenizer did not have latent tokens"
                    )

        """ Send to dtype if specified """

        if self.dtype is not None:
            model.to(STR_TO_TORCH_DTYPE[self.dtype])

        """ Wrap in PEFT if necessary """

        if self.is_lora:
            if isinstance(model, LatentThinkingModel) and not model.lora_mode:
                warnings.warn(
                    "Model was LatentThinkingModel and lora_mode was False, but we are doing lora training. Enabling lora mode!"
                )
                model.lora_mode_enable()

            if self.lora_path_or_repo is not None:
                model = PeftModel.from_pretrained(model, self.lora_path_or_repo, is_trainable=True)
            else:
                lora_config = LoraConfig(**self.lora_config_dict)
                model = get_peft_model(model, lora_config)
        else:
            if isinstance(model, LatentThinkingModel) and model.lora_mode:
                warnings.warn(
                    "Model was LatentThinkingModel and lora_mode was True, but we are not doing lora training. Disabling lora mode!"
                )
                model.lora_mode_disable()

        return model, tokenizer


def automodelforcausallm_from_pretrained_latent(
    pretrained_model_name_or_path: Union[str, os.PathLike[str]], *args, **kwargs
) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

    if hasattr(config, "_inserted_LatentThinkingModelSettings"):
        return LatentThinkingModel.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
