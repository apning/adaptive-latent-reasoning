import copy
from functools import partial
from peft import PeftConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from trl.trainer.grpo_config import GRPOConfig

from accelerate import Accelerator

from src.eval import EvalConfig, eval_from_config, get_tokenized_eval_dataset
from src.rl.training_utils import (
    ExpandedMetricsGRPOTrainer,
    GRPOTrainingConfig,
    TimeLimitCallback,
    fix_tokenizer_so_always_add_special_tokens,
    get_datasets_from_training_config,
    get_reward_funcs_and_weights_from_training_config,
)

accelerator = Accelerator()
IS_MAIN_PROCESS = accelerator.is_main_process


def train(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    training_config: GRPOTrainingConfig,
    grpo_config: GRPOConfig,
    lora_config: PeftConfig | None,
    val_config: EvalConfig | None,
    resuming: bool,
    time_limit: float | None,
):
    ## Start time limit callback before we do data processing
    time_limit_callback = TimeLimitCallback(max_seconds=time_limit * 3600) if time_limit is not None else None

    """ Processing args """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if IS_MAIN_PROCESS:
            print("pad_token_id is not set! Setting it to eos_token_id")

    """ Get datasets """

    train_dataset, val_dataset = get_datasets_from_training_config(training_config, tokenizer)

    """ Get reward funcs and weights """

    reward_funcs, reward_weights = get_reward_funcs_and_weights_from_training_config(
        training_config=training_config, grpo_config=grpo_config, tokenizer=tokenizer
    )

    if grpo_config.reward_weights is not None:
        raise ValueError(
            f"reward_weights must be None since it will be set later. But got: {grpo_config.reward_weights}"
        )

    grpo_config.reward_weights = reward_weights

    """ Get custom eval func """

    if val_config is not None:
        custom_val_func_dataset = get_tokenized_eval_dataset(tokenizer, val_config)
        custom_eval_fn = partial(
            eval_from_config,
            tokenizer=copy.deepcopy(tokenizer),
            eval_config=val_config,
            dataset=custom_val_func_dataset,
            batch_size=grpo_config.gradient_accumulation_steps * grpo_config.per_device_eval_batch_size,
            verbose=False,
        )
        custom_eval_fn.__name__ = "custom_eval"
    else:
        custom_eval_fn = None

    # make sure to deepcopy tokenizer
    # and assign a good __name__ attribute. Maybe just "custom_eval"

    """ Fix tokenizer """
    # Because GRPOTrainer will only tokenizer without special tokens.
    # This means leaving behind all latent thinking tokens and the <BOS>
    # https://github.com/huggingface/trl/issues/3520
    fix_tokenizer_so_always_add_special_tokens(tokenizer)

    """ Start training """

    callbacks = [time_limit_callback] if time_limit_callback is not None else None

    trainer = ExpandedMetricsGRPOTrainer(
        custom_eval_fn=custom_eval_fn,
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
        peft_config=lora_config,
    )

    trainer.train(resume_from_checkpoint=resuming)
