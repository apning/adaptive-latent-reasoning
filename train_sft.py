import argparse
from pathlib import Path
import warnings
import copy


from src.get_options import (
    get_eval_config_from_options,
    get_latent_thinking_settings_from_options,
    get_lora_config_from_options,
    get_training_config_from_options,
)
from src.model_creation import ModelCreationConfig
from src.modeling import LatentThinkingModel
from src.modeling_utils import (
    ensure_tokenizer_has_latent_tokens,
)
from src.training import train
from src.training_utils import CheckpointManager, strip_peft
from src.utils import get_checkpoint_dir, str2bool, str2float_w_none, str2int_w_none, str_formatted_datetime, str_w_none

"""
Command-line interface for training models via SFT.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ahoy! Parse them arghs!",
    )

    parser.add_argument(
        "--model_path_or_repo",
        type=str_w_none,
        default=None,
        help="String to model path or repo that can be instantiated with HF",
    )
    parser.add_argument(
        "--latent_thinking_settings_options",
        type=str_w_none,
        default=None,
        help="Slash-separated string specifying the options for the LatentThinkingModelSettings. To use default settings, specify the string 'default'",
    )
    parser.add_argument(
        "--lora_config_options",
        type=str_w_none,
        default=None,
        help="Slash-separated string specifying the options for the LoraConfig. If not specified, LoRA will not be used. If you wish to use lora with all default options, use the string 'default'",
    )
    parser.add_argument(
        "--training_config_options",
        type=str_w_none,
        default=None,
        help="Slash-separated string specifying the options for the TrainingConfig",
    )
    parser.add_argument(
        "--val_config_options",
        type=str_w_none,
        default=None,
        help="Slash-separated string specifying the options for the EvalConfig",
    )
    parser.add_argument(
        "--dir_name",
        type=str_w_none,
        default=None,
        help="Name of subdirectory to create for this training run under checkpoints. Slashes in name will be separated into nested subdirectories. Subdirectory that will be created will be at checkpoints/<dir_name>/<datetime>.",
    )
    parser.add_argument(
        "--save_dir_override",
        default=None,
        type=str,
        help="If specified, models/checkpoints (but not logging) will be saved here instead of in the usual directory. Useful if there are storage limitations. A comparable file hierarchy using --dir_name will be created under save_dir_override to house the saved models. Mutually exclusive w/ --hf_saving_username",
    )
    parser.add_argument(
        "--hf_saving_username",
        default=None,
        type=str_w_none,
        help="If specified, save to HF repo instead of local disk. The repo name will be automatically generated from --save_dir_override. Mutually exclusive w/ --save_dir_override",
    )
    parser.add_argument(
        "--resume_path",
        default=None,
        type=str_w_none,
        help="Path to a checkpoint directory with form checkpoints/<dir_name>/.../<datetime> from which to continue training. Ie. it should have been the logging directory. If specified, none of the above arguments may also be specified. The directory must contain a resume.json.",
    )
    parser.add_argument(
        "--do_not_save",
        nargs="?",
        const=True,  # bare flag → True
        default=False,  # flag absent → False
        type=str2bool,
        help="If True, wil not save any model/optimizer checkpoint files. Will still save logging and Tensorboard files",
    )
    parser.add_argument(
        "--device_criteria",
        type=str,
        default="m",
        help="Whether to use free memory or available utilization to select device. 'm' for memory-based selection, 'u' for utilization-based selection. Default 'm'. I mean, this is LLMs. What are you training, a 5M parameter model? What do you think the first L stands for? Memory is the first priority, duh.",
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
        help="If True, subtracts 15 mins (30 mins if uploading to HF) from the time limit if specified. This should give enough time for the current step to complete the final checkpoints saving, ensuring the whole process actually finishes under time limit",
    )
    parser.add_argument(
        "--micro_batch_size",
        default=None,
        type=str2int_w_none,
        help='Overrides training_config.initial_micro_batch_size. If using DDP with world size > 1, micro batch size is not adaptive to OOM, so a good value must be specified at the start. If unspecified or empty string ("") then will not override training config.',
    )
    parser.add_argument(
        "--min_micro_batch_size",
        default=None,
        type=str2int_w_none,
        help="If set, adaptive micro batching will not permanently decrease micro batch size to below this size. Only works if world size == 1.",
    )

    # Get args
    args = parser.parse_args()

    model_path_or_repo = args.model_path_or_repo
    latent_thinking_settings_options = args.latent_thinking_settings_options
    training_config_options = args.training_config_options
    val_config_options = args.val_config_options
    lora_config_options = args.lora_config_options
    dir_name = args.dir_name
    save_dir_override = args.save_dir_override
    hf_saving_username = args.hf_saving_username
    resume_path = args.resume_path
    do_not_save = args.do_not_save
    device_criteria = args.device_criteria
    time_limit = args.time_limit
    wrap_up_time = args.wrap_up_time
    micro_batch_size = args.micro_batch_size
    min_micro_batch_size = args.min_micro_batch_size

    """ Check/Process Args """

    if resume_path and (
        model_path_or_repo
        or latent_thinking_settings_options
        or training_config_options
        or val_config_options
        or lora_config_options
        or dir_name
        or save_dir_override
        or hf_saving_username
    ):
        raise ValueError(
            f"If resume_path is specified then no earlier arguments can be! Got resume_path: {resume_path}"
        )

    if save_dir_override and hf_saving_username:
        raise ValueError(
            f"Only one of save_dir_override and hf_saving_username can be specified. But got save_dir_override: {save_dir_override}, hf_saving_username: {hf_saving_username}"
        )

    if not resume_path and not dir_name:
        raise ValueError("Either resume_path or dir_name must be specified")

    """ Get the items needed for training """

    if not resume_path:
        """ Get training/val config """

        training_config = get_training_config_from_options(training_config_options)
        val_config = get_eval_config_from_options(val_config_options)

        """ Get latent thinking settings """
        if training_config.is_latent:
            if not latent_thinking_settings_options:
                raise ValueError(
                    "latent_thinking_settings_options is required if training_config.is_latent is True. For default settings, specify 'default'"
                )
            latent_settings_dict = get_latent_thinking_settings_from_options(
                latent_thinking_settings_options, return_as_dict=True
            )
        else:
            if latent_thinking_settings_options:
                raise ValueError(
                    f"latent_thinking_settings_options must not be specified if training_config.is_latent is False. But got: {latent_thinking_settings_options}"
                )
            latent_settings_dict = None

        """ Get LoRA config dict """

        is_lora = bool(lora_config_options)
        if is_lora:
            lora_config_dict = get_lora_config_from_options(lora_config_options, return_as_dict=True)
        else:
            lora_config_dict = None

        """ Get Model creation config """

        model_creation_config = ModelCreationConfig(
            path_or_repo=model_path_or_repo,
            dtype=training_config.torch_dtype_str,
            is_latent=training_config.is_latent,
            latent_settings_dict=latent_settings_dict,
            is_lora=is_lora,
            lora_config_dict=lora_config_dict,
        )

        """ Get model and tokenizer """

        model, tokenizer = model_creation_config.get_model_and_tokenizer()

        """ Get logging dir """

        datetime_str = str_formatted_datetime()

        logging_dir = get_checkpoint_dir(dir_name, datetime=datetime_str)

        if logging_dir.exists():
            raise ValueError(f"logging_dir ({logging_dir}) already exists!")

        """ Get checkpoint dir or repo """

        if save_dir_override:
            checkpoint_dir_or_repo = get_checkpoint_dir(dir_name, datetime=datetime_str, base_dir=save_dir_override)
        elif hf_saving_username:
            checkpoint_dir_or_repo = "lt-checkpoints-" + dir_name.strip("/ ") + "-" + datetime_str
        else:
            checkpoint_dir_or_repo = logging_dir / "saved"

        """ Create checkpoint manager """

        checkpoint_manager = CheckpointManager(
            checkpoint_dir_or_repo=checkpoint_dir_or_repo,
            hf_username=hf_saving_username,
            should_exist=False,
            keep_only_last_checkpoint_local=training_config.keep_only_last_checkpoint
            if not hf_saving_username
            else None,
            initial_model_creation_config=model_creation_config,
            logging_dir=logging_dir,
        )

    else:
        resume_path = Path(resume_path)

        checkpoint_manager = CheckpointManager.from_resume_json(resume_path)

        model, tokenizer, training_config, val_config, logging_dir = checkpoint_manager.get_initial_train_items()

    """ Misc processing/checks """

    if training_config.is_latent:
        if not isinstance(strip_peft(model), LatentThinkingModel):
            raise ValueError("training_config.is_latent is True, but model is not a LatentThinkingModel")
        ensure_tokenizer_has_latent_tokens(tokenizer)

    if checkpoint_manager.is_hf and not training_config.keep_only_last_checkpoint:
        warnings.warn(
            "training_config.keep_only_last_checkpoint was False but saving to HF. When saving to HF, older checkpoints will always be overwritten. Buyer beware!"
        )

    if time_limit is not None and wrap_up_time:
        time_to_subtract = 0.25 if not checkpoint_manager.is_hf else 0.5
        if time_limit <= time_to_subtract:
            warnings.warn(
                f"Wrap up time specified, but total time limit was less/equal to than {int(time_to_subtract * 60)} minutes so wrap up time is not possible. Total time: {time_limit}"
            )
        else:
            time_limit -= time_to_subtract

    if micro_batch_size is not None:
        training_config = copy.deepcopy(training_config)
        training_config.initial_micro_batch_size = micro_batch_size

    train(
        model=model,
        tokenizer=tokenizer,
        training_config=training_config,
        val_config=val_config,
        logging_dir=logging_dir,
        checkpoint_manager=checkpoint_manager,
        resuming=bool(resume_path),
        device_selection_mode=device_criteria,
        time_limit=time_limit,
        do_not_save=do_not_save,
        min_micro_batch_size=min_micro_batch_size,
    )
