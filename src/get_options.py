import sys
from typing import Any

from peft import LoraConfig
from trl.trainer.grpo_config import GRPOConfig
from src.eval import EvalConfig
from src.settings import LatentThinkingModelSettings
from src.training_utils import TrainingConfig
from src.utils import get_project_root
from src.rl.training_utils import GRPOTrainingConfig

# Add root directly to Python path for imports
options_path = get_project_root()
if str(options_path) not in sys.path:
    sys.path.insert(0, str(options_path))

from options.latent_thinking_settings_options import LATENT_THINKING_SETTINGS_OPTIONS  # noqa: E402
from options.training_config_options import TRAINING_CONFIG_OPTIONS  # noqa: E402
from options.eval_config_options import EVAL_CONFIG_OPTIONS  # noqa: E402
from options.lora_config_options import LORA_CONFIG_OPTIONS  # noqa: E402
from options.grpo_config_options import GRPO_CONFIG_OPTIONS  # noqa: E402
from options.grpo_training_options import GRPO_TRAINING_OPTIONS  # noqa: E402


def split_name_into_parts(name: str) -> list[str]:
    return [part.strip() for part in name.split("/") if part.strip()]


def build_options_dict_from_name(name: str, options_dict: dict[str, Any]):
    name_parts = split_name_into_parts(name)
    options = {}
    for part in name_parts:
        options.update(options_dict[part])
    return options


def get_latent_thinking_settings_from_options(
    name: str, return_as_dict: bool = False
) -> LatentThinkingModelSettings | dict:
    options_dict = build_options_dict_from_name(name, LATENT_THINKING_SETTINGS_OPTIONS)

    if return_as_dict:
        return options_dict

    return LatentThinkingModelSettings(**options_dict)


def get_training_config_from_options(name: str, return_as_dict: bool = False) -> TrainingConfig | dict:
    options_dict = build_options_dict_from_name(name, TRAINING_CONFIG_OPTIONS)

    if return_as_dict:
        return options_dict

    return TrainingConfig(**options_dict)


def get_eval_config_from_options(name: str, return_as_dict: bool = False) -> EvalConfig | dict:
    options_dict = build_options_dict_from_name(name, EVAL_CONFIG_OPTIONS)

    if return_as_dict:
        return options_dict

    return EvalConfig(**options_dict)


def get_lora_config_from_options(name: str, return_as_dict: bool = False) -> LoraConfig | dict:
    options_dict = build_options_dict_from_name(name, LORA_CONFIG_OPTIONS)

    if return_as_dict:
        return options_dict

    return LoraConfig(**options_dict)


def get_grpo_config_from_options(name: str, return_as_dict: bool = False) -> GRPOConfig | dict:
    options_dict = build_options_dict_from_name(name, GRPO_CONFIG_OPTIONS)

    if return_as_dict:
        return options_dict

    return GRPOConfig(**options_dict)


def get_grpo_training_config_from_options(name: str, return_as_dict: bool = False) -> GRPOTrainingConfig | dict:
    options_dict = build_options_dict_from_name(name, GRPO_TRAINING_OPTIONS)

    if return_as_dict:
        return options_dict

    return GRPOTrainingConfig(**options_dict)