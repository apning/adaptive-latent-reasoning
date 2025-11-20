from dataclasses import dataclass, field
from typing import ClassVar
import warnings
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig
from src.modeling_utils import SPECIAL_TOKEN_STRS, START_TOKEN_STR, END_TOKEN_STR, CONTINUE_TOKEN_STR, STOP_TOKEN_STR
from src.utils import JsonMixin


class _LatentThinkingModelConfig:
    """
    Dummy config class. Exists to allow PreTrainedModel.from_pretrained(...) to instantiate the correct config class when it calls 'cls.config_class.from_pretrained(...)'
    """

    @staticmethod
    def from_pretrained(*args, **kwargs) -> PretrainedConfig:
        # I wonder if just forwarding to AutoConfig.from_pretrained(...) would be easier than this...
        # Oh well, this works too.
        config_dict, _ = PretrainedConfig.get_config_dict(*args, **kwargs)

        if "model_type" in config_dict:
            config_cls = type(AutoConfig.for_model(config_dict["model_type"]))
            return config_cls.from_pretrained(*args, **kwargs)
        else:
            raise ValueError(
                "config to load had no 'model_type' key. Correct config class cannot be found without this value"
            )


@dataclass
class LatentThinkingModelSettings(JsonMixin):
    """

    SETTINGS for a latent thinking model. Deliberately not named "config" to prevent mistaking as part of the Hugging Face ecosystem. It is NOT. It is not a subclass of PretrainedConfig.

    It just holds some settings for a latent thinking model

    Args:
        disable_forward_input_embeds (bool): If True, the model will not forward inputs_embeds to the base model. Instead, it will raise an error. This may be useful as the model has no mechanism to determine when latent thinking is enabled or not if inputs_embeds are provided instead of input_ids. Therefore, the model cannot correctly mask the logits accordingly.

        disable_input_past_key_values (bool): Throw an exception if past_key_values is not None on the .forward()

        disable_checkpointing_cache_update (bool): Disables .update(...) method for any LatentThinkingCheckpointingCache created by the model. Will not affect any LatentThinkingCheckpointingCache given to the model as input

        add_latent_to_end (bool): If True, the <END> token will be added to the last latent state. If False, that last latent state will simply be replaced with the <END> token, and there will be one fewer latent state.

        start_token_id, continue_token_id, end_token_id, stop_token_id (int | None): Token ids for the respective tokens. Defaults to None (unset)

        recurrent_filter_mode (str | None): What kind of recurrent filter (the module that sits between recurrent iterations) the model should use. None means none is used
        detach_binary_head_inputs (bool): If true, inputs to the binary head will be detached. This means that gradients from binary head will not flow through the rest of the model

        lora_mode (bool): When fine-tuning with LoRA, it would be preferable if the input/output embeddings were kept frozen. But then, how to train the additional special tokens? The solution: Make them separate modules and then splice them into the input/output embeddings as needed. During lora_mode, the model keeps these as separate so that only they may be trained.

        binary_head_temp (float | None): Temperature for binary heads. Has a multiplicative effect on temperature applied later.

        unused_token_ids (list[int]): The addition of the special token ids to the model may result in unused additional token ids if the resizing of the embeddings is done up to a padding value. We track the unused token ids here. They can be used in future to add additional tokens, and should also be tracked to prevent their selection during generation.

        debug_mode (bool): When True, various debugging checks, warnings, prints, and exceptions are activated within the model

    """

    disable_forward_input_embeds: bool = True
    disable_input_past_key_values: bool = False
    disable_checkpointing_cache_update: bool = True
    add_latent_to_end: bool = False

    start_token_id: int | None = None
    continue_token_id: int | None = None
    end_token_id: int | None = None
    stop_token_id: int | None = None

    recurrent_filter_mode: str | None = None
    detach_binary_head_inputs: bool = False

    lora_mode: bool = False

    binary_head_temp: float | None = None

    debug_mode: bool = False

    unused_token_ids: set[int] = field(default_factory=set)

    SPECIAL_TOKEN_KEYS: ClassVar[set[str]] = {"start_token_id", "continue_token_id", "end_token_id", "stop_token_id"}
    VALID_RECURRENT_FILTER_MODES: ClassVar[set[str]] = {"linear", "linear_ln", "linear_ECF", "SRE", "MLP"}

    def __post_init__(self):
        self.validate()

    def validate(self):
        # Could add stuff here to make sure the dict added to a config in insert_in_config will work with json.dumps(...) when the config is dumped via PretrainedConfig.save_pretrained(...)
        if (
            self.recurrent_filter_mode is not None
            and self.recurrent_filter_mode not in self.VALID_RECURRENT_FILTER_MODES
        ):
            raise ValueError(
                f"{self.recurrent_filter_mode} is not valid! Please specify either None or one of: ",
                self.VALID_RECURRENT_FILTER_MODES,
            )

        if self.binary_head_temp is None:
            pass
        elif self.binary_head_temp == 0:
            raise ValueError("binary_head_temp cannot be 0")
        elif self.binary_head_temp < 0:
            raise ValueError(f"binary_head_temp cannot be negative. Got {self.binary_head_temp}")

    def insert_in_config(self, config: PretrainedConfig):
        if hasattr(config, "_inserted_LatentThinkingModelSettings"):
            warnings.warn("config file already had a _inserted_LatentThinkingModelSettings attribute! Overwriting...")

        settings_dict = vars(self).copy()

        # Because json.dumps() can't handle sets
        settings_dict["unused_token_ids"] = list(settings_dict["unused_token_ids"])

        config._inserted_LatentThinkingModelSettings = settings_dict

    @classmethod
    def pop_from_config(cls, config: PretrainedConfig, return_if_missing: bool = True):
        if not hasattr(config, "_inserted_LatentThinkingModelSettings"):
            if return_if_missing:
                return None
            raise ValueError("config did not have a _inserted_LatentThinkingModelSettings attribute!")

        if not isinstance(config._inserted_LatentThinkingModelSettings, dict):
            raise ValueError(
                f"config._inserted_LatentThinkingModelSettings was not a dict! Instead it was of type {type(config._inserted_LatentThinkingModelSettings)}"
            )

        settings_dict = config._inserted_LatentThinkingModelSettings

        ## Legacy stuff that is no longer needed
        if "_generation_mode" in settings_dict:
            del settings_dict["_generation_mode"]

        # to find attributes that are missing from settings_dict
        all_keys = vars(cls()).keys()
        missing_keys = set(all_keys) - set(settings_dict.keys())

        if missing_keys:
            warnings.warn(
                f"Extracted keys from config did not contain the keys for the following attributes. They will retain their default values: {missing_keys}",
            )

        # When it comes back from JSON it's a list, but we want a set
        if "unused_token_ids" in settings_dict:
            settings_dict["unused_token_ids"] = set(settings_dict["unused_token_ids"])

        settings = cls(**settings_dict)

        cls.remove_from_config(config)

        return settings

    @staticmethod
    def remove_from_config(config: PretrainedConfig, must_exist: bool = False):
        if hasattr(config, "_inserted_LatentThinkingModelSettings"):
            del config._inserted_LatentThinkingModelSettings
        elif must_exist:
            raise ValueError("The attribute _inserted_LatentThinkingModelSettings did not exist in config")

    def add_special_tokens(self, special_tokens_to_ids: dict[str, int]):
        """
        Add special token ids using the dict output by src.modeling_utils.add_special_latent_tokens_to_tokenizer
        """

        if not set(SPECIAL_TOKEN_STRS).issubset(special_tokens_to_ids.keys()):
            raise ValueError(f"Special tokens to ids dict must contain the following keys: {SPECIAL_TOKEN_STRS}")

        additional_keys = set(special_tokens_to_ids.keys()) - set(SPECIAL_TOKEN_STRS)
        if additional_keys:
            warnings.warn(
                f"special_tokens_to_ids contained the following additional keys. These will not be added as they are unfamiliar: {additional_keys}"
            )

        self.start_token_id = special_tokens_to_ids[START_TOKEN_STR]
        self.end_token_id = special_tokens_to_ids[END_TOKEN_STR]
        self.continue_token_id = special_tokens_to_ids[CONTINUE_TOKEN_STR]
        self.stop_token_id = special_tokens_to_ids[STOP_TOKEN_STR]

    def are_special_tokens_set(self, strict: bool = True) -> bool:
        all_none = all(getattr(self, key) is None for key in self.SPECIAL_TOKEN_KEYS)
        all_set = all(getattr(self, key) is not None for key in self.SPECIAL_TOKEN_KEYS)

        if strict and not all_none and not all_set:
            raise ValueError("Special tokens must be either all set or all None. Got a mix of set and None.")

        return all_set

    def get_special_tokens_to_ids_dict(self) -> dict[str, int | None]:
        return {
            START_TOKEN_STR: self.start_token_id,
            END_TOKEN_STR: self.end_token_id,
            CONTINUE_TOKEN_STR: self.continue_token_id,
            STOP_TOKEN_STR: self.stop_token_id,
        }
