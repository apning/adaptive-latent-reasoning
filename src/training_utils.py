import copy
from dataclasses import asdict, dataclass, field

from io import BytesIO
import json
import re
import gc
import os
from pathlib import Path
import shutil
from typing import Any, Callable
import warnings

from huggingface_hub import HfApi, HfFileSystem
from peft import PeftModel
import torch
from torch import nn
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from transformers import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


from src.data_processing import SUPPORTED_TRAINING_DATA_NAMES, tokenize_gsm8k_aug_for_training
from src.eval import EvalConfig
from src.model_creation import ModelCreationConfig
from src.modeling import LatentThinkingModel
from src.utils import (
    STR_TO_TORCH_DTYPE,
    JsonMixin,
    PrintLog,
    find_instances,
    get_project_root,
    is_power_of_2,
    load_from_json,
    save_as_json,
    str_w_none,
)

# This will be used when gradient checkpointing is enabled
_GRADIENT_CHECKPOINTING_KWARGS = {"use_reentrant": False}

SUPPORTED_TRAINING_DATA_MODES = {"cot", "latent", "no cot"}
REMOVE_LAST_STEP_DATA_NAMES = {"gsm8k-aug", "gsm8k-aug-nl"}

LATENT_LOSS_CALC_MODES = {"smooth_l1", "mean_l2"}
LATENT_LOSS_NORM_MODES = {"none", "layerwise_std", "layerwise_avg_l2", "std", "avg_l2"}


@dataclass
class TrainingConfig(JsonMixin):
    ## Training Data
    data_name: str = None
    data_mode: str = "cot"  # Options in SUPPORTED_TRAINING_DATA_MODES
    slice_proportion: float | None = None
    latent_thinking_factor: int | None = None
    latent_thinking_bias: int | None = None
    min_latent_thinking_steps: int | None = None
    max_latent_thinking_steps: int | None = None
    mask_latent_reasoning_labels: bool = False
    remove_last_reasoning_step: bool = False
    shift_target_token_one: bool = False
    max_training_tokens: int | None = 512
    additional_trainset_kwargs: dict = field(default_factory=dict)
    epochs: int | None = None
    steps: int | None = None
    # Batch size must NOT be changed when resuming a previous training run - it will mess up resuming the position of the dataloader
    batch_size: int = 128

    ## Lr Scheduler
    scheduler_name: str | None = None
    scheduler_num_warmup_steps: int | None = None

    ## Latent thinking loss
    # Whether to use each loss component
    # Values can be True, False, or a non-zero float. If a float, it also serves as the coefficient of the loss component. True is the same as 1.0
    codi_loss: bool | float = False
    mean_reasoning_loss: bool | float = False

    # Loss calculation / normalization modes
    # Only applicable for latent mode(s)
    latent_loss_calc_mode: str | None = None
    latent_loss_norm_mode: str | None = None

    # If None, loss will be enforced on all blocks
    # If a tuple of ints, then it defines the range of intermediate blocks that will be used. Eg. if (2, 8), then the 2nd and 8th blocks will be used. Blocks are 0-indexed and both ends are inclusive.
    intermediate_block_latent_loss_range: tuple[int, int] | list[int, int] | None = None

    ## Optimization
    lr: float = 1e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    # If True, certain parameters (such as biases, norms, and new embeddings tokens in LoRA mode) will not undergo weight decay
    no_decay_some_params: bool = False

    ## Computational
    use_gradient_checkpointing: bool = False
    use_gradient_checkpointing_if_necessary: bool = False
    # Initial micro batch size. This is PER device
    # If none, training will default to per device batch size
    # If DDP is used during training, then this is not dynamic. Otherwise, training will lower automatically if OOM
    # Reccomend setting if DDP is used. Otherwise, do not.
    initial_micro_batch_size: int | None = None
    # Number of processes used to process data
    num_proc: int = 12
    # Number of workers to use with dataloader
    num_workers: int = 4
    # Seed for rng. Currently *only* used with DistributedSampler
    seed: int = 42
    torch_dtype_str: str = "bf16"

    ## Evaluation/Saving

    # How often to evaluate and save if best val score, in steps
    val_period: int = 5000
    # How often to save a checkpoint (in steps) that can be resumed from. If None, checkpoints that can be resumed from will NOT be saved EXCEPT for the save after the training loop ends.
    save_checkpoint_period: int | None = None
    # If True, delete the last saved checkpoint once the next is saved
    keep_only_last_checkpoint: bool = True
    # TODO: implement early stopping

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validation of args"""

        if self.data_name not in SUPPORTED_TRAINING_DATA_NAMES:
            raise ValueError(
                f"Invalid data name: {self.data_name}. Please choose from: {SUPPORTED_TRAINING_DATA_NAMES}"
            )

        if self.epochs is None and self.steps is None:
            raise ValueError("Either epochs or steps must be specified")
        elif self.epochs is not None and self.steps is not None:
            raise ValueError("Only one of epochs or steps must be specified")

        if self.data_mode not in SUPPORTED_TRAINING_DATA_MODES:
            raise ValueError(
                f"Invalid data mode: {self.data_mode}. Please choose from: {SUPPORTED_TRAINING_DATA_MODES}"
            )

        if self.data_mode != "latent" and self.is_latent:
            raise ValueError("Data mode must be 'latent' when any latent loss enabled")
        if self.data_mode == "latent" and not self.is_latent:
            raise ValueError("If data_mode is latent, then at least one latent loss must be enabled")
        if not self.is_latent and (
            self.intermediate_block_latent_loss_range is not None
            or self.latent_thinking_factor is not None
            or self.latent_thinking_bias is not None
            or self.min_latent_thinking_steps is not None
            or self.max_latent_thinking_steps is not None
            or self.mask_latent_reasoning_labels
        ):
            raise ValueError("A latent loss must be specified for any supporting latent setting to be set!")

        if self.intermediate_block_latent_loss_range is not None:
            if not isinstance(self.intermediate_block_latent_loss_range, (tuple, list)):
                raise ValueError(
                    f"intermediate_block_latent_loss_range must be a tuple/list of 2 ints or None, but got {type(self.intermediate_block_latent_loss_range)}"
                )
            if len(self.intermediate_block_latent_loss_range) != 2 or not all(
                isinstance(x, int) for x in self.intermediate_block_latent_loss_range
            ):
                raise ValueError(
                    f"intermediate_block_latent_loss_range must be a tuple/list of 2 ints or None, but got {len(self.intermediate_block_latent_loss_range)}"
                )

        if self.remove_last_reasoning_step and self.data_name not in REMOVE_LAST_STEP_DATA_NAMES:
            raise ValueError(
                f"remove_last_reasoning_step cannot be True if data_name is not one of {REMOVE_LAST_STEP_DATA_NAMES}. data_name is {self.data_name}"
            )

        if not is_power_of_2(self.batch_size):
            raise ValueError(
                f"batch_size must be a power of 2 (for gradient accumulation purposes), but got {self.batch_size}"
            )

        ## Latent loss calc/modes
        if self.is_latent:
            if self.latent_loss_calc_mode not in LATENT_LOSS_CALC_MODES:
                raise ValueError(
                    f"latent_loss_calc_mode must be one of {LATENT_LOSS_CALC_MODES}, but got {self.latent_loss_calc_mode}"
                )
            if self.latent_loss_norm_mode not in LATENT_LOSS_NORM_MODES:
                raise ValueError(
                    f"latent_loss_norm_mode must be one of {LATENT_LOSS_NORM_MODES}, but got {self.latent_loss_norm_mode}"
                )
        else:
            if self.latent_loss_calc_mode is not None or self.latent_loss_norm_mode is not None:
                raise ValueError("latent_loss_calc_mode and latent_loss_norm_mode must be None if is_latent is False")

        if (
            self.max_latent_thinking_steps is not None
            and self.min_latent_thinking_steps is not None
            and self.max_latent_thinking_steps < self.min_latent_thinking_steps
        ):
            raise ValueError(
                f"max_latent_thinking_steps must be greater than or equal to min_latent_thinking_steps! But got max_latent_thinking_steps = {self.max_latent_thinking_steps}, min_latent_thinking_steps = {self.min_latent_thinking_steps}"
            )

        if self.torch_dtype_str not in STR_TO_TORCH_DTYPE.keys():
            raise ValueError(f"torch_dtype_str must be one of {STR_TO_TORCH_DTYPE.keys()}, got {self.torch_dtype_str}")
        if self.torch_dtype_str in ["float16", "fp16"]:
            raise NotImplementedError(
                "Currently float16 is not supported as the training code has not implemented GradScalar. Either try bfloat16 or implement GradScalar in training"
            )

        if not self.scheduler_name and self.scheduler_num_warmup_steps:
            raise ValueError("scheduler_name must be specified if scheduler_num_warmup_steps is specified")

        if self.is_latent and (self.use_gradient_checkpointing or self.use_gradient_checkpointing_if_necessary):
            raise NotImplementedError(
                "Currently gradient checkpointing is not supported with latent training (because it doesn't seem to help much)"
            )

    @property
    def is_latent(self) -> bool:
        return bool(self.codi_loss or self.mean_reasoning_loss)


def get_tokenized_train_dataset(tokenizer, training_config: TrainingConfig):
    if training_config.data_name not in SUPPORTED_TRAINING_DATA_NAMES:
        raise ValueError(
            f"The data_name {training_config.data_name} is not valid! Please pick one from {SUPPORTED_TRAINING_DATA_NAMES}"
        )

    if "gsm8k-aug" in training_config.data_name:
        if training_config.data_name == "gsm8k-aug":
            natural_language = False
        elif training_config.data_name == "gsm8k-aug-nl":
            natural_language = True

        return tokenize_gsm8k_aug_for_training(
            tokenizer,
            natural_language=natural_language,
            mode=training_config.data_mode,
            remove_last_step=training_config.remove_last_reasoning_step,
            latent_thinking_factor=training_config.latent_thinking_factor,
            latent_thinking_bias=training_config.latent_thinking_bias,
            min_latent_thinking_steps=training_config.min_latent_thinking_steps,
            max_latent_thinking_steps=training_config.max_latent_thinking_steps,
            mask_latent_reasoning_labels=training_config.mask_latent_reasoning_labels,
            shift_target_token_one=training_config.shift_target_token_one,
            max_tokens=training_config.max_training_tokens,
            slice_proportion=training_config.slice_proportion,
            num_proc=training_config.num_proc,
            **training_config.additional_trainset_kwargs,
        )


def update_model_creation_config_for_saving(
    initial_model_creation_config: ModelCreationConfig,
    save_dir_or_repo: str | os.PathLike,
    model: PreTrainedModel | PeftModel | DDP,
) -> ModelCreationConfig:
    model = strip_ddp(model)
    save_dir_or_repo = str(save_dir_or_repo)

    model_creation_config = copy.deepcopy(initial_model_creation_config)

    if model_creation_config.is_latent:
        assert isinstance(strip_peft(model), LatentThinkingModel), (
            f"If ModelCreationConfig.is_latent is True we expect model to be a LatentThinkingModel. But got {type(model)}"
        )
    if isinstance(model, PeftModel):
        assert model_creation_config.is_lora, "If model is PeftModel we expect ModelCreationConfig.is_lora to be True"
        # If a PeftModel, then what we saved was the peft adapters.
        model_creation_config.lora_path_or_repo = save_dir_or_repo
        # The peft config was also saved, so no need for us to have lora_config_dict anymore
        model_creation_config.lora_config_dict = None
    else:
        assert not model_creation_config.is_lora, (
            "If model is not PeftModel we expect ModelCreationConfig.is_lora to be False"
        )
        # If the model was not a PeftModel, then we just saved the actual model. So we can just get it from there in the future
        model_creation_config.path_or_repo = save_dir_or_repo
        if model_creation_config.is_latent:
            # Since the actual model was saved, it will save its latent settings too, so we don't need to carry it around anymore
            model_creation_config.latent_settings_dict = None

    return model_creation_config


def save_checkpoint(
    save_dir: os.PathLike,
    initial_model_creation_config: ModelCreationConfig,
    model: PreTrainedModel | PeftModel | DDP,
    tokenizer,
    optimizer,
    scheduler,
    training_config: TrainingConfig,
    val_config: EvalConfig,
    epoch_num: int,
    step_num: int,
    best_val_score: float,
    override_if_exists: bool = False,
):
    save_dir = Path(save_dir)

    if save_dir.exists() and not override_if_exists:
        raise ValueError(f"Save directory {save_dir} already exists and override_if_exists is False")

    save_dir.mkdir(parents=True, exist_ok=True)

    model = strip_ddp(model)

    # Save model
    model.save_pretrained(save_dir)

    # Save tokenizer
    tokenizer.save_pretrained(save_dir)

    ## Update model creation config and save it
    model_creation_config = update_model_creation_config_for_saving(initial_model_creation_config, save_dir, model)
    model_creation_config.save_as_json(save_dir / "model_creation_config.json", override_if_exists=True)

    """ Make directory in which to save additional things with which to resume training """

    training_save_dir = save_dir / "training"
    training_save_dir.mkdir(parents=True, exist_ok=True)

    # Save training config
    training_config.save_as_json(training_save_dir / "training_config.json", override_if_exists=True)

    # Save val config
    val_config.save_as_json(training_save_dir / "val_config.json", override_if_exists=True)

    # Save training state
    training_state_save_path = training_save_dir / "training_state.pt"
    training_state = {
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch_num": epoch_num,
        "step_num": step_num,
        "best_val_score": best_val_score,
    }
    if scheduler is not None:
        training_state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(training_state, training_state_save_path)

    # Create at a glance text file
    at_a_glance_path = training_save_dir / "at_a_glance.txt"
    if at_a_glance_path.exists():
        at_a_glance_path.unlink()

    with open(at_a_glance_path, "w") as f:
        f.write(f"epoch_num: {epoch_num}\n")
        f.write(f"step_num: {step_num}\n")
        f.write(f"best_val_score: {best_val_score}\n")


def strip_ddp(model: DDP | Any) -> Any:
    if isinstance(model, DDP):
        return model.module
    return model


def strip_peft(model: PeftModel | Any) -> Any:
    if isinstance(model, PeftModel):
        return model.get_base_model()
    return model


def strip_ddp_and_peft(model: PeftModel | DDP | Any) -> Any:
    return strip_peft(strip_ddp(strip_peft(model)))


@dataclass
class AutoMicroBatch:
    micro_batch_starting_size: int
    allow_gradient_checkpointing: bool
    allow_micro_batch: bool = True
    micro_batch_key: str = "micro_batch_size"
    print_prefix: str = ""
    printlog: PrintLog | None = None
    min_micro_batch_size: int | None = None

    def __post_init__(self):
        self.micro_batch_size = self.micro_batch_starting_size
        if self.printlog is None:
            self.printlog = PrintLog()

        if self.min_micro_batch_size is not None and not self.allow_micro_batch:
            self.printlog(
                "Min micro batch size will have no effect when adaptive micro batching is not available!", warning=True
            )
            self.min_micro_batch_size = None
        if self.min_micro_batch_size is not None and self.micro_batch_starting_size < self.min_micro_batch_size:
            self.printlog(
                f"Min micro batch size ({self.min_micro_batch_size}) was less than micro batch starting size ({self.micro_batch_starting_size})! Min micro batch size will be ignored.",
                warning=True,
            )
            self.min_micro_batch_size = None

    def __call__(
        self,
        func: Callable,
        step_num: int | None = None,
        epoch_num: int | None = None,
        *func_args,
        **func_kwargs,
    ):
        if step_num is None:
            step_num = "~unknown~"

        if epoch_num is None:
            epoch_num = "~unknown~"

        if self.micro_batch_key in func_kwargs:
            self.printlog(
                self.print_prefix + f"'{self.micro_batch_key}' must not be specified in func_kwargs",
                raise_exception=ValueError,
            )

        # Train model on batch with gradient checkpointing/gradient accumulation retry logic in case of Torch OOM
        while True:
            try:
                func_kwargs[self.micro_batch_key] = self.micro_batch_size

                if (
                    self.allow_micro_batch
                    and self.min_micro_batch_size is not None
                    and self.micro_batch_size < self.min_micro_batch_size
                ):
                    self.printlog(
                        f"Raising micro batch size from {self.micro_batch_size} to minimum of {self.min_micro_batch_size}"
                    )
                    self.micro_batch_size = self.min_micro_batch_size

                return func(*func_args, **func_kwargs)
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()

                for optimizer in find_instances(torch.optim.Optimizer, func_args, func_kwargs):
                    optimizer.zero_grad()

                if not self.allow_gradient_checkpointing and not self.allow_micro_batch:
                    self.printlog(
                        self.print_prefix
                        + f"Torch OOM at epoch {epoch_num} / step_num {step_num}. But neither gradient checkpointing nor micro-batching is allowed, so nothing can be done."
                    )
                    raise

                fix_attempted = False
                # Turn on gradient checkpointing if it was not on
                if self.allow_gradient_checkpointing:
                    models = find_instances((PreTrainedModel, DDP), func_args, func_kwargs)
                    if not models:
                        self.printlog(
                            self.print_prefix
                            + "Tried to turn on gradient checkpointing, but no model found in func_args or func_kwargs! Support for more model types may need to be implemented.",
                            raise_exception=ValueError,
                        )

                    num_turned_gc_on = 0
                    for model in models:
                        model = strip_ddp(model)
                        if not model.is_gradient_checkpointing:
                            model.gradient_checkpointing_enable(
                                gradient_checkpointing_kwargs=_GRADIENT_CHECKPOINTING_KWARGS
                            )
                            num_turned_gc_on += 1

                    if num_turned_gc_on:
                        self.printlog(
                            self.print_prefix
                            + f"Torch OOM at epoch {epoch_num} / step_num {step_num}. Enabled gradient checkpointing on {num_turned_gc_on} models since it was not already enabled"
                        )
                        fix_attempted = True
                # Try decreasing micro batch size
                if not fix_attempted and self.allow_micro_batch:
                    if self.micro_batch_size > 1:
                        self.micro_batch_size //= 2
                        self.printlog(
                            self.print_prefix
                            + f"Torch OOM at epoch {epoch_num} / step_num {step_num}. Halved micro batch size to {self.micro_batch_size}."
                        )
                    else:
                        self.printlog(
                            self.print_prefix
                            + f"Torch OOM at epoch {epoch_num} / step_num {step_num} and could not decrease micro batch size any further. Bummer!"
                        )
                        raise


def ddp_setup():
    """Setup DDP if running under torchrun"""
    if "LOCAL_RANK" in os.environ:
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False


def find_last_step_dir(directory_path: os.PathLike) -> Path | None:
    """
    Find the subdirectory within directory_path matching the name pattern 'step_<number>' with greatest <number> value. If None found, returns None
    """

    path = Path(directory_path)

    if not path.is_dir():
        raise ValueError(f"Directory {path} does not exist")

    # Find all subdirectories matching "step_<number>" pattern
    step_dirs = []
    for item in path.iterdir():
        if item.is_dir() and item.name.startswith("step_"):
            # Extract the number part
            match = re.match(r"step_(\d+)", item.name)
            if match:
                number = int(match.group(1))
                step_dirs.append((number, item))

    if not step_dirs:
        return None

    # Find the directory with the highest number
    highest_number, highest_dir = max(step_dirs, key=lambda x: x[0])
    return highest_dir


class ResumeStepDataLoader:
    """
    Given a DataLoader, starts the first iter(...) call on that dataloader on a specified step to resume from.
    """

    def __init__(self, dataloader: DataLoader, resume_step_num: int = 0):
        self.dataloader = dataloader
        self.resume_step_num = resume_step_num

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        iterator = iter(self.dataloader)
        if self.resume_step_num:
            for _ in range(self.resume_step_num % len(self)):
                next(iterator)
            self.resume_step_num = 0

        return iterator

    @property
    def sampler(self) -> Sampler:
        return self.dataloader.sampler


def list_batchnorm_modules_names(model: nn.Module) -> list[str]:
    """
    Returns list of names for all BatchNorm/SyncBatchNorm modules.
    """
    bn_types = (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)
    bn_mod_names = []
    for name, mod in model.named_modules():
        if isinstance(mod, bn_types):
            bn_mod_names.append(name)
    return bn_mod_names


def torch_save_and_bite(data) -> bytes:
    buffer = BytesIO()
    torch.save(data, buffer)
    return buffer.getvalue()


def bite_str(text: str) -> bytes:
    return BytesIO(text.encode("utf-8")).getvalue()


@dataclass
class CheckpointManager:
    checkpoint_dir_or_repo: str | os.PathLike = None
    hf_username: str | None = None
    should_exist: bool = False
    keep_only_last_checkpoint_local: bool = None
    hf_repo_prefix: str = ""
    initial_model_creation_config: ModelCreationConfig = None
    logging_dir: os.PathLike = None

    def __post_init__(self):
        self.hf_username = str_w_none(self.hf_username)

        self.is_hf = self.hf_username is not None

        if self.is_hf:
            self.hf_api = HfApi()
            self.hf_fs = HfFileSystem()
            self.checkpoint_dir_or_repo = str(self.checkpoint_dir_or_repo).strip(" /").replace("/", "-")
            self.last_step_repo_id = (
                self.hf_username + "/" + self.hf_repo_prefix + self.checkpoint_dir_or_repo + "-last_step"
            )
            self.best_val_repo_id = (
                self.hf_username + "/" + self.hf_repo_prefix + self.checkpoint_dir_or_repo + "-best_val"
            )
        else:
            self.checkpoint_dir_or_repo = Path(self.checkpoint_dir_or_repo)
            self.last_step_save_dir = None

        if self.should_exist:
            if self.is_hf and not self.hf_last_step_exists():
                raise ValueError(f"last_step_repo_id {self.last_step_repo_id} does not exist!")
            if not self.is_hf and not self.exists():
                raise ValueError(f"Checkpoint directory {self.checkpoint_dir_or_repo} does not exist")
        elif not self.should_exist and self.exists():
            if self.is_hf:
                raise ValueError(
                    f"One or both of last_step_repo_id {self.last_step_repo_id} or best_val_repo_id {self.best_val_repo_id} already exist"
                )
            else:
                raise ValueError(f"Checkpoint directory {self.checkpoint_dir_or_repo} already exists")

        if self.initial_model_creation_config is None or self.logging_dir is None:
            raise ValueError("initial_model_creation_config and logging_dir must be specified")

        if isinstance(self.initial_model_creation_config, dict):
            self.initial_model_creation_config = ModelCreationConfig(**self.initial_model_creation_config)
        self.logging_dir = Path(self.logging_dir)

    def hf_best_val_exists(self) -> bool:
        assert self.is_hf, "Should be in hf mode!"
        return self.hf_api.repo_exists(self.best_val_repo_id)

    def hf_last_step_exists(self) -> bool:
        assert self.is_hf, "Should be in hf mode!"
        return self.hf_api.repo_exists(self.last_step_repo_id)

    def exists(self) -> bool:
        if self.is_hf:
            return self.hf_best_val_exists() or self.hf_last_step_exists()
        else:
            return self.checkpoint_dir_or_repo.exists()

    def get_last_step_path_or_repo(self) -> Path | str:
        if self.is_hf:
            if not self.hf_last_step_exists():
                raise ValueError(f"Last step repo {self.last_step_repo_id} does not exist")
            return self.last_step_repo_id

        else:
            last_step_path = find_last_step_dir(self.checkpoint_dir_or_repo)
            if last_step_path is None:
                raise ValueError(f"Last step directory could not be found at {self.checkpoint_dir_or_repo}")
            return last_step_path

    def get_initial_train_items(
        self,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, TrainingConfig, EvalConfig, Path]:
        last_path_repo = self.get_last_step_path_or_repo()
        if not self.is_hf:
            model_creation_config = ModelCreationConfig.load_from_json(last_path_repo / "model_creation_config.json")
            training_config = TrainingConfig.load_from_json(last_path_repo / "training" / "training_config.json")
            val_config = EvalConfig.load_from_json(last_path_repo / "training" / "val_config.json")
            self.last_step_save_dir = last_path_repo
        else:
            model_creation_config = ModelCreationConfig(
                **self.hf_download_and_unjson(last_path_repo + "/model_creation_config.json")
            )
            training_config = TrainingConfig(
                **self.hf_download_and_unjson(last_path_repo + "/training/training_config.json")
            )
            val_config = EvalConfig(**self.hf_download_and_unjson(last_path_repo + "/training/val_config.json"))

        model, tokenizer = model_creation_config.get_model_and_tokenizer()

        return model, tokenizer, training_config, val_config, copy.deepcopy(self.logging_dir)

    def split_hf_path(self, path: str) -> tuple[str, str]:
        resolved_path = self.hf_fs.resolve_path(path)
        repo_id = resolved_path.repo_id
        path_in_repo = resolved_path.path_in_repo
        return repo_id, path_in_repo

    def get_last_step_local_dir(self, step_num: int):
        assert not self.is_hf, "Should not be in hf mode!"
        return self.checkpoint_dir_or_repo / f"step_{step_num}"

    def get_best_val_local_dir(self):
        assert not self.is_hf, "Should not be in hf mode!"
        return self.checkpoint_dir_or_repo / "best_val"

    def hf_download_and_torch_load(self, path: str, map_location) -> Any:
        repo_id, path_in_repo = self.split_hf_path(path)
        file_path = self.hf_api.hf_hub_download(repo_id=repo_id, filename=path_in_repo)
        return torch.load(file_path, map_location=map_location, weights_only=True)

    def hf_download_and_unjson(self, path: str) -> Any:
        repo_id, path_in_repo = self.split_hf_path(path)
        file_path = self.hf_api.hf_hub_download(repo_id=repo_id, filename=path_in_repo)
        with open(file_path, "r") as f:
            return json.load(f)

    def save_checkpoint(
        self,
        model: PreTrainedModel,
        tokenizer,
        optimizer,
        scheduler,
        training_config: TrainingConfig,
        val_config: EvalConfig,
        epoch_num: int,
        step_num: int,
        best_val_score: float,
        mode: str = "last_step",
    ):
        if mode not in ["last_step", "best_val"]:
            raise ValueError(f"Invalid mode: {mode}. Please choose from ['last_step', 'best_val']")

        if not hasattr(self, "_prev_saved_step_num"):
            self._prev_saved_step_num = {}
        if self._prev_saved_step_num.get(mode) == step_num:
            return

        model = strip_ddp(model)

        if not self.is_hf:
            if mode == "last_step":
                save_dir = self.get_last_step_local_dir(step_num)
                if save_dir == self.last_step_save_dir:
                    return
            else:
                save_dir = self.get_best_val_local_dir()

            save_checkpoint(
                save_dir=save_dir,
                initial_model_creation_config=self.initial_model_creation_config,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                training_config=training_config,
                val_config=val_config,
                epoch_num=epoch_num,
                step_num=step_num,
                best_val_score=best_val_score,
                override_if_exists=mode == "best_val",
            )

            if mode == "last_step" and self.keep_only_last_checkpoint_local:
                if self.last_step_save_dir is not None:
                    if self.last_step_save_dir.exists():
                        shutil.rmtree(self.last_step_save_dir)
                    else:
                        raise ValueError(
                            f"self.last_step_save_dir does not exist! This is most confusing. self.last_step_save_dir: {self.last_step_save_dir}"
                        )
                self.last_step_save_dir = save_dir
        else:
            if not hasattr(self, "_prev_hf_run_as_futures"):
                self._prev_hf_run_as_futures = {}
            for k, v in self._prev_hf_run_as_futures.items():
                if v.done():
                    try:
                        v.result()
                    except:
                        print(f"\n\n\nException from the result of {k}\n\n\n")
                        raise

            if mode == "last_step":
                repo_id = self.last_step_repo_id
            else:
                repo_id = self.best_val_repo_id

            model.push_to_hub(repo_id, private=True)

            tokenizer.push_to_hub(repo_id, private=True)

            model_creation_config = update_model_creation_config_for_saving(
                self.initial_model_creation_config, repo_id, model
            )

            self.hf_api.upload_file(
                path_or_fileobj=bite_str(model_creation_config.to_json()),
                path_in_repo="model_creation_config.json",
                repo_id=repo_id,
            )

            self.hf_api.upload_file(
                path_or_fileobj=bite_str(training_config.to_json()),
                path_in_repo="training/training_config.json",
                repo_id=repo_id,
            )

            self.hf_api.upload_file(
                path_or_fileobj=bite_str(val_config.to_json()), path_in_repo="training/val_config.json", repo_id=repo_id
            )

            # Upload is expensive, the optimizer state dict is huge, and resuming from best val isn't the right way anyway. So don't bother
            if mode == "last_step":
                training_state = {
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch_num": epoch_num,
                    "step_num": step_num,
                    "best_val_score": best_val_score,
                }
                if scheduler is not None:
                    training_state["scheduler_state_dict"] = scheduler.state_dict()

                training_state_future = self.hf_api.upload_file(
                    path_or_fileobj=torch_save_and_bite(training_state),
                    path_in_repo="training/training_state.pt",
                    repo_id=repo_id,
                    run_as_future=True,
                )

                self._prev_hf_run_as_futures[f"{mode}/{step_num}/training_state"] = training_state_future

            at_a_glance = f"epoch_num: {epoch_num}\nstep_num: {step_num}\nbest_val_score: {best_val_score}\n"

            self.hf_api.upload_file(
                path_or_fileobj=bite_str(at_a_glance),
                path_in_repo="training/at_a_glance.txt",
                repo_id=repo_id,
            )

        if mode == "last_step":
            self.save_resume_json(self.logging_dir / "resume.json")

        self._prev_saved_step_num[mode] = step_num

    def wait_for_futures(self):
        if self.is_hf:
            if hasattr(self, "_prev_hf_run_as_futures"):
                for k, v in self._prev_hf_run_as_futures.items():
                    try:
                        v.result()
                    except:
                        print(f"\n\n\nException from the result of {k}\n\n\n")
                        raise
            else:
                warnings.warn(
                    "wait_for_futures called and self.is_hf is True but no self._prev_hf_run_as_futures attribute exists"
                )

    def get_last_training_state(self, map_location):
        last_path_repo = self.get_last_step_path_or_repo()

        if not self.is_hf:
            self.last_step_save_dir = last_path_repo
            return torch.load(
                last_path_repo / "training" / "training_state.pt", map_location=map_location, weights_only=True
            )
        else:
            return self.hf_download_and_torch_load(
                last_path_repo + "/training/training_state.pt", map_location=map_location
            )

    def save_resume_json(self, save_path: os.PathLike):
        # logging_dir should be saved as a path relative to the project root, so resuming can successfully be done on other machines
        logging_dir = Path(self.logging_dir).resolve()
        proj_root = get_project_root().resolve()

        assert logging_dir.is_relative_to(proj_root), (
            f"Expected logging dir to be relative to (nested under) the project root. But got logging_dir: {logging_dir} and proj_root: {proj_root}"
        )

        relative_logging_dir = logging_dir.relative_to(proj_root)

        resume_dict = {
            "checkpoint_dir_or_repo": str(self.checkpoint_dir_or_repo),
            "hf_username": self.hf_username,
            "should_exist": True,
            "keep_only_last_checkpoint_local": self.keep_only_last_checkpoint_local,
            "hf_repo_prefix": self.hf_repo_prefix,
            "initial_model_creation_config": asdict(self.initial_model_creation_config),
            "relative_logging_dir": str(relative_logging_dir),
        }

        save_as_json(resume_dict, save_path, override_if_exists=True)

    @classmethod
    def from_resume_json(cls, load_path: os.PathLike) -> "CheckpointManager":
        load_path = Path(load_path)

        assert load_path.exists(), f"Load path {load_path} does not exist"

        if load_path.is_dir():
            load_path = load_path / "resume.json"
            assert load_path.exists(), (
                f"Load path was provided as a directory ({load_path}) but resume.json does not exist in it. If it is named differently, specify the full path."
            )

        resume_dict = load_from_json(load_path)

        resume_dict["initial_model_creation_config"] = ModelCreationConfig(
            **resume_dict["initial_model_creation_config"]
        )

        resume_dict["logging_dir"] = get_project_root() / resume_dict.pop("relative_logging_dir")

        return cls(**resume_dict)
