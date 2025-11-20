import argparse
from pathlib import Path
import warnings

import torch
from transformers import AutoTokenizer
from accelerate import Accelerator
from peft import PeftConfig
from trl.trainer.grpo_config import GRPOConfig


from src.eval import EvalConfig
from src.get_options import (
    get_eval_config_from_options,
    get_grpo_config_from_options,
    get_lora_config_from_options,
    get_grpo_training_config_from_options,
)
from src.model_creation import automodelforcausallm_from_pretrained_latent
from src.modeling import LatentThinkingModel
from src.modeling_utils import (
    ensure_tokenizer_has_latent_tokens,
)
from src.rl.training import train
from src.rl.training_utils import GRPOTrainingConfig
from src.utils import (
    get_checkpoint_dir,
    save_as_json,
    load_from_json,
    str2bool,
    str2float_w_none,
    str2int_w_none,
    str_formatted_datetime,
    str_w_none,
)

"""
Command-line interface for training models via GRPO.

"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ahoy! Parse them arghs!",
    )
    parser.add_argument(
        "--training_config_options",
        type=str_w_none,
        default=None,
        help="Slash-separated string specifying the options for the GRPOTrainingConfig",
    )
    parser.add_argument(
        "--grpo_config_options",
        type=str_w_none,
        default=None,
        help="Slash-separated string specifying the options for the GRPOConfig",
    )
    parser.add_argument(
        "--lora_config_options",
        type=str_w_none,
        default=None,
        help="Slash-separated string specifying the options for the LoraConfig. If not specified, LoRA will not be used. If you wish to use lora with all default options, use the string 'default'",
    )
    parser.add_argument(
        "--val_config_options",
        type=str_w_none,
        default=None,
        help="Slash-separated string specifying the options for the EvalConfig. If not specified, custom evaluation will not be used.",
    )
    parser.add_argument(
        "--dir_name",
        type=str_w_none,
        default=None,
        help="Name of subdirectory to create for this training run under checkpoints. Slashes in name will be separated into nested subdirectories. Subdirectory that will be created will be at grpo_checkpoints/<dir_name>/<datetime>.",
    )
    parser.add_argument(
        "--save_dir_override",
        default=None,
        type=str,
        help="If specified, subdirectory will be created at save_dir_override/grpo_checkpoints/<dir_name>/<datetime>",
    )
    parser.add_argument(
        "--resume_path",
        default=None,
        type=str_w_none,
        help="Path to the output directory of a run to resume",
    )
    parser.add_argument(
        "--time_limit",
        type=str2float_w_none,
        default=None,
        help='Time limit, in hours. If empty string "", "none", "false" (both case insenitive) then no time limit (will be set to None). Once the time limit has reached, the current mini-batched will be finished, the last checkpoint saved, and then training will be complete.',
    )
    parser.add_argument(
        "--wrap_up_time",
        nargs="?",
        const=True,  # bare flag → True
        default=False,  # flag absent → False
        type=str2bool,
        help="If True, subtracts 15 mins from the time limit if specified. This should give enough time for the current step to complete the final checkpoints saving, ensuring the whole process actually finishes under time limit",
    )
    parser.add_argument(
        "--micro_batch_size",
        default=None,
        type=str2int_w_none,
        help="Micro batch size per device. Aka per_device_train_batch_size. Gradient accumulation steps will be automatically adjusted based on this, the batch size, and the number of processes",
    )

    # Get args
    args = parser.parse_args()

    # Parse args
    training_config_options = args.training_config_options
    grpo_config_options = args.grpo_config_options
    lora_config_options = args.lora_config_options
    val_config_options = args.val_config_options
    dir_name = args.dir_name
    save_dir_override = args.save_dir_override
    resume_path = args.resume_path
    time_limit = args.time_limit
    wrap_up_time = args.wrap_up_time
    micro_batch_size = args.micro_batch_size

    accelerator = Accelerator()
    num_processes = accelerator.num_processes
    is_main_process = accelerator.is_main_process

    """ Check/Process Args """

    if resume_path and (
        training_config_options
        or grpo_config_options
        or lora_config_options
        or val_config_options
        or dir_name
        or save_dir_override
    ):
        raise ValueError(
            f"If resume_path is specified then no earlier arguments can be! Got resume_path: {resume_path}"
        )

    if not resume_path and not dir_name:
        raise ValueError("Either resume_path or dir_name must be specified")

    if micro_batch_size is None:
        raise ValueError("micro_batch_size must be specified")

    """ Get the items needed for training """

    if not resume_path:
        """ Resolve output dir """
        datetime_str = str_formatted_datetime()

        if save_dir_override:
            output_dir = get_checkpoint_dir(
                dir_name, datetime=datetime_str, base_dir=save_dir_override, checkpoints_name="grpo_checkpoints"
            )
        else:
            output_dir = get_checkpoint_dir(dir_name, datetime=datetime_str, checkpoints_name="grpo_checkpoints")

        """ Check output dir does not exist """

        if output_dir.exists():
            raise ValueError(f"output_dir ({output_dir}) already exists!")

        accelerator.wait_for_everyone()  # Ensure the check for existence executes for everyone before we might create the output dir

        """ Get configs """

        training_config = get_grpo_training_config_from_options(training_config_options)
        lora_config = get_lora_config_from_options(lora_config_options) if lora_config_options is not None else None
        grpo_config_dict = get_grpo_config_from_options(grpo_config_options, return_as_dict=True)
        val_config = get_eval_config_from_options(val_config_options) if val_config_options is not None else None

        ## Set some things in grpo_config_dict
        grpo_config_dict["do_train"] = True
        grpo_config_dict["do_eval"] = True
        grpo_config_dict["output_dir"] = str(output_dir)

        ## Checks for grpo_config_dict
        if "reward_weights" in grpo_config_dict and grpo_config_dict["reward_weights"] is not None:
            raise ValueError(
                "reward_weights must not be specified in GRPOConfig options. It will be automatically set later"
            )

        """ Save configs """

        if is_main_process:
            config_save_dir = output_dir / "saved_for_resume"
            config_save_dir.mkdir(parents=True)
            training_config.save_as_json(config_save_dir / "grpo_training_config.json")
            if lora_config is not None:
                lora_config.save_pretrained(config_save_dir)
            if val_config is not None:
                val_config.save_as_json(config_save_dir / "val_config.json")
            save_as_json(grpo_config_dict, config_save_dir / "grpo_config_dict.json")
    else:
        resume_path = Path(resume_path)

        config_save_dir = resume_path / "saved_for_resume"

        if not config_save_dir.exists():
            raise ValueError(f"config_save_dir ({config_save_dir}) does not exist!")

        training_config = GRPOTrainingConfig.load_from_json(config_save_dir / "grpo_training_config.json")

        if (config_save_dir / "adapter_config.json").exists():
            lora_config = PeftConfig.from_pretrained(config_save_dir)
        else:
            print("Resuming: No LoRA config detected! None will be used.")
            lora_config = None

        val_config_path = config_save_dir / "val_config.json"
        if val_config_path.exists():
            val_config = EvalConfig.load_from_json(val_config_path)
        else:
            val_config = None

        grpo_config_dict = load_from_json(config_save_dir / "grpo_config_dict.json")

        if Path(grpo_config_dict["output_dir"]).resolve() != resume_path.resolve():
            raise ValueError(
                f"output_dir ({grpo_config_dict['output_dir']}) does not match resume_path ({resume_path})"
            )

    """ Set gradient accumulation steps and per device batch sizes """

    batch_size = training_config.batch_size

    if batch_size % (num_processes * micro_batch_size) != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by (num_processes ({num_processes}) * micro_batch_size ({micro_batch_size}))"
        )

    gradient_accumulation_steps = batch_size // (num_processes * micro_batch_size)

    per_device_train_batch_size = micro_batch_size
    per_device_eval_batch_size = micro_batch_size

    grpo_config_dict["gradient_accumulation_steps"] = gradient_accumulation_steps
    grpo_config_dict["per_device_train_batch_size"] = per_device_train_batch_size
    grpo_config_dict["per_device_eval_batch_size"] = per_device_eval_batch_size

    # Avoid forcing a distributed backend in single-process runs
    if num_processes == 1:
        grpo_config_dict.pop("ddp_backend", None)

    grpo_config = GRPOConfig(**grpo_config_dict)

    """ Get model and tokenizer """

    model_path_or_repo = training_config.model_path_or_repo
    model = automodelforcausallm_from_pretrained_latent(model_path_or_repo)
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_repo)

    if training_config.is_latent:
        if not isinstance(model, LatentThinkingModel):
            raise ValueError("training_config.is_latent is True, but model is not a LatentThinkingModel")
        ensure_tokenizer_has_latent_tokens(tokenizer)

        if lora_config is not None:
            model.lora_mode_enable()
        else:
            model.lora_mode_disable()

    if val_config is not None and val_config.latent_thinking:
        if not isinstance(model, LatentThinkingModel):
            raise ValueError("val_config.latent_thinking is True, but model is not a LatentThinkingModel")
        ensure_tokenizer_has_latent_tokens(tokenizer)

    if grpo_config.bf16:
        model.to(torch.bfloat16)

    if isinstance(model, LatentThinkingModel):
        model.binary_head_temp = training_config.binary_head_temp
    else:
        if training_config.binary_head_temp is not None:
            raise ValueError(
                f"binary_head_temp must be None if model is not a LatentThinkingModel. But got training_config.binary_head_temp = {training_config.binary_head_temp}"
            )

    """ Misc processing/checks """

    if time_limit is not None and wrap_up_time:
        time_to_subtract = 0.25
        if time_limit <= time_to_subtract:
            warnings.warn(
                f"Wrap up time specified, but total time limit was less/equal to than {int(time_to_subtract * 60)} minutes so wrap up time is not possible. Total time: {time_limit}"
            )
        else:
            time_limit -= time_to_subtract

    train(
        model=model,
        tokenizer=tokenizer,
        training_config=training_config,
        grpo_config=grpo_config,
        lora_config=lora_config,
        val_config=val_config,
        resuming=bool(resume_path),
        time_limit=time_limit,
    )
