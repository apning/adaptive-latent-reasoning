import copy
import time
from dataclasses import dataclass
from typing import Callable

from peft import PeftModel
import torch
from transformers import TrainerCallback
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from src.rl.data_processing import get_gsm8k_aug_for_rl
from src.rl.rewards import (
    AllGensCorrectProportion,
    AllGensWrongProportion,
    RelativeLengthRewarder,
    SimpleAccuracyRewarder,
    SimpleFormatRewarder,
    SimpleLatentLengthRewarder,
    SomeGensCorrectProportion,
    gsm8k_aug_answer_verifier,
    latent_length_judge,
    latent_prefix_format_verifier,
)
from src.utils import JsonMixin


SUPPORTED_TRAINING_DATA_NAMES = {
    "gsm8k-aug",
}

SUPPORTED_VAL_DATA_NAMES = {
    "gsm8k-aug",
}


@dataclass
class GRPOTrainingConfig(JsonMixin):
    is_latent: bool = True

    ## Model
    model_path_or_repo: str = None
    binary_head_temp: float | None = None

    ## Training data
    data_name: str = "gsm8k-aug"
    append_start_token: bool = True
    max_prompt_tokens: int = 100
    slice_proportion: float | None = None

    ## Training
    batch_size: int = 64
    format_penalty_weight: float = 1.0  # It is a NON-NEGATIVE weight
    accuracy_reward_weight: float = 1.0
    relative_length_penalty: float = 1.0
    relative_length_accuracy_requirement: float | int | None = None
    relative_length_reward: None | float | str = None

    ## Validation data
    # If None then will default to data_name
    val_data_name: str | None = None

    ## Computational
    num_proc: int = 12  # For use w/ data processing

    def __post_init__(self):
        if self.val_data_name is None:
            self.val_data_name = self.data_name

        self.validate()

    def validate(self):
        if self.model_path_or_repo is None:
            raise ValueError("model_path_or_repo must be provided")

        if self.data_name not in SUPPORTED_TRAINING_DATA_NAMES:
            raise ValueError(f"data_name must be one of {SUPPORTED_TRAINING_DATA_NAMES}, got {self.data_name}")

        if self.val_data_name not in SUPPORTED_VAL_DATA_NAMES:
            raise ValueError(f"val_data_name must be one of {SUPPORTED_VAL_DATA_NAMES}, got {self.val_data_name}")

        if self.append_start_token and not self.is_latent:
            raise ValueError("append_start_token must be False if is_latent is False")

        if self.relative_length_reward and not self.relative_length_accuracy_requirement:
            raise ValueError("relative_length_reward requires relative_length_accuracy_requirement to be set")


class TimeLimitCallback(TrainerCallback):
    def __init__(self, max_seconds: int, announce: bool = True):
        self.max_seconds = max_seconds
        self.announce = announce
        self._t0 = time.monotonic()  # start clock at instantiation

    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() - self._t0 >= self.max_seconds:
            control.should_training_stop = True
            control.should_save = True
            if self.announce and state.is_world_process_zero:
                print(
                    f"[TimeLimitCallback] Reached {self.max_seconds}s at step {state.global_step}. Saving and stopping."
                )
        return control


def get_datasets_from_training_config(training_config: GRPOTrainingConfig, tokenizer: PreTrainedTokenizerBase):
    """Get train dataset"""
    if training_config.data_name == "gsm8k-aug":
        train_dataset = get_gsm8k_aug_for_rl(
            append_start_token=training_config.append_start_token,
            split="train",
            max_prompt_tokens=training_config.max_prompt_tokens,
            slice_proportion=1 - training_config.slice_proportion
            if training_config.slice_proportion is not None
            else None,
            tokenizer=tokenizer,
            num_proc=training_config.num_proc,
        )
    else:
        raise ValueError(f"Unsupported data name: {training_config.data_name}")

    """ Get val dataset """

    if training_config.val_data_name == "gsm8k-aug":
        val_dataset = get_gsm8k_aug_for_rl(
            append_start_token=training_config.append_start_token,
            split="validation",
            tokenizer=tokenizer,
            num_proc=training_config.num_proc,
        )
    else:
        raise ValueError(f"Unsupported val data name: {training_config.val_data_name}")

    return train_dataset, val_dataset


def get_reward_funcs_and_weights_from_training_config(
    training_config: GRPOTrainingConfig, grpo_config: GRPOConfig, tokenizer: PreTrainedTokenizerBase
) -> tuple[list[Callable], list[float]]:
    reward_funcs = []
    reward_weights = []
    if training_config.is_latent:
        if training_config.data_name == "gsm8k-aug":
            relative_len_rewarder = RelativeLengthRewarder(
                num_generations=grpo_config.num_generations,
                format_verifier=latent_prefix_format_verifier,
                answer_verifier=gsm8k_aug_answer_verifier,
                length_judge=latent_length_judge,
                format_penalty=-training_config.format_penalty_weight,
                answer_reward=training_config.accuracy_reward_weight,
                relative_length_penalty=training_config.relative_length_penalty,
                tokenizer=copy.deepcopy(tokenizer),
                relative_length_accuracy_requirement=training_config.relative_length_accuracy_requirement,
                relative_length_reward=training_config.relative_length_reward,
            )
            reward_funcs.append(relative_len_rewarder)
            reward_funcs.append(
                SimpleFormatRewarder(format_verifier=latent_prefix_format_verifier, tokenizer=copy.deepcopy(tokenizer))
            )
            reward_funcs.append(SimpleAccuracyRewarder(answer_verifier=gsm8k_aug_answer_verifier))
            reward_funcs.append(SimpleLatentLengthRewarder(tokenizer=copy.deepcopy(tokenizer)))
            reward_funcs.append(
                AllGensCorrectProportion(
                    answer_verifier=gsm8k_aug_answer_verifier, num_generations=grpo_config.num_generations
                )
            )
            reward_funcs.append(
                SomeGensCorrectProportion(
                    answer_verifier=gsm8k_aug_answer_verifier, num_generations=grpo_config.num_generations
                )
            )
            reward_funcs.append(
                AllGensWrongProportion(
                    answer_verifier=gsm8k_aug_answer_verifier, num_generations=grpo_config.num_generations
                )
            )
            reward_weights.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        else:
            raise ValueError(f"Unsupported data name: {training_config.data_name}")
    else:
        raise NotImplementedError("Not implemented for non-latent mode")

    return reward_funcs, reward_weights


## Made to solve https://github.com/huggingface/trl/issues/3520
# The original proposed implmentation doesn't work due to instance checking in GRPOTrainer
def fix_tokenizer_so_always_add_special_tokens(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    OriginalClass = tokenizer.__class__

    class PatchedTokenizerClass(OriginalClass):
        def __call__(self, *args, add_special_tokens=True, **kwargs):
            if not add_special_tokens:
                add_special_tokens = True
            return super().__call__(*args, add_special_tokens=add_special_tokens, **kwargs)

    # So that save_pretrained(...) will still work correctly
    PatchedTokenizerClass.__name__ = OriginalClass.__name__

    tokenizer.__class__ = PatchedTokenizerClass


class ExpandedMetricsGRPOTrainer(GRPOTrainer):
    """
    Although GRPOTrainer logs all rewards calculated using the validation set, they cannot be used as the argument for 'metric_for_best_model' as GRPOTrainer does not make them available.
    This modification makes it so that any of the evaluation metrics calculated by GRPO can also be used as the 'metric_for_best_model'.

    Can also pass a custom evaluation function. The function must only take the model, and output a dict with string keys and float/int values.
    At the moment, the custom eval function is run independently on each device. If the result is not determinisitic, this can result in diverging behavior if the result of custom eval function is used as the best metric and is used to make control flow decisions.
    If custom_eval_fn is passed, it must have a __name__ attribute.

    """

    def __init__(
        self,
        *args,
        custom_eval_fn: Callable[[PreTrainedModel | PeftModel], dict[str, float | int]] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if custom_eval_fn is not None and not hasattr(custom_eval_fn, "__name__"):
            raise ValueError("custom_eval_fn must have a __name__ attribute")

        self.custom_eval_fn = custom_eval_fn

    def evaluate(self, *args, **kwargs):
        # Run evaluation first (this computes rewards, logs them, and clears internal buffers)
        metrics = super().evaluate(*args, **kwargs)

        prefix = kwargs.get("metric_key_prefix", "eval")

        # Grab the most recent log entry that contains eval_* keys (it was just added by GRPOTrainer.log)
        last_eval = None
        for rec in reversed(self.state.log_history):
            if any(k.startswith("eval_") for k in rec.keys()):
                last_eval = rec
                break

        if last_eval is not None:
            metrics.update({k: v for k, v in last_eval.items() if k.startswith("eval_")})

        if self.custom_eval_fn is not None:
            model = self.accelerator.unwrap_model(self.model)
            model.eval()
            with torch.inference_mode():
                custom_metrics = self.custom_eval_fn(model)
            custom_eval_fn_name = self.custom_eval_fn.__name__
            custom_metrics = {prefix + "_" + custom_eval_fn_name + "/" + k: v for k, v in custom_metrics.items()}
            self.log(custom_metrics)
            metrics.update(custom_metrics)

        return metrics
