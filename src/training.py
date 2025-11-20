from contextlib import nullcontext
from math import ceil
import os
from pathlib import Path
import time
import torch
from torch import optim
import torch.distributed as dist
from torch.distributed import destroy_process_group
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.optimization import get_scheduler


from src.data_collation import LatentDataCollator
from src.eval import EvalConfig, eval_from_config, get_tokenized_eval_dataset
from src.losses import calc_latent_loss
from src.modeling import LatentThinkingModel
from src.modeling_utils import ensure_tokenizer_has_latent_tokens
from src.training_utils import (
    AutoMicroBatch,
    CheckpointManager,
    ResumeStepDataLoader,
    TrainingConfig,
    ddp_setup,
    get_tokenized_train_dataset,
    _GRADIENT_CHECKPOINTING_KWARGS,
    strip_ddp,
    strip_peft,
)
from src.utils import (
    STR_TO_TORCH_DTYPE,
    PrintLog,
    get_decay_grouped_params,
    is_power_of_2,
    str_formatted_datetime,
    get_obj_vars_str,
    select_best_device,
    sum_dicts,
    timer,
)

# If modified, it WILL mess up resuming already saved checkpoints
# Why would you want to modify it anyway?
# Don't you have anything better to do?
# You could cook a nice home made meal
DETERMINISTIC_DATASET_SHUFFLE_SEED = 42


def get_loss_for_batch(
    model, batch, training_config: TrainingConfig
) -> tuple[torch.Tensor, dict[str, float], dict[str, float], dict[str, float]]:
    device = strip_ddp(model).get_input_embeddings().weight.device

    loss_components = {}
    time_elapsed = {}
    statistics = {}

    """ Get necessary components from batch """

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    """ Shape checks """

    if not input_ids.shape == attention_mask.shape == labels.shape:
        raise ValueError(
            f"input_ids, attention_mask, and labels must have the same shape. But got {input_ids.shape} != {attention_mask.shape} != {labels.shape}"
        )

    """ Model forward """

    with timer() as t:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            output_hidden_states=training_config.is_latent,
        )

    time_elapsed["teacher_forward"] = t.elapsed

    loss_components["teacher_ce_loss"] = outputs.loss

    """ Latent loss calculation """

    if training_config.is_latent:
        """ Get necessary components from batch """

        latent_input_ids = batch["latent_input_ids"].to(device)
        latent_attention_mask = batch["latent_attention_mask"].to(device)
        latent_labels = batch["latent_labels"].to(device)

        """ Shape checks """

        if not latent_input_ids.shape == latent_attention_mask.shape == latent_labels.shape:
            raise ValueError(
                f"latent_input_ids, latent_attention_mask, and latent_labels must have the same shape. But got {latent_input_ids.shape} != {latent_attention_mask.shape} != {latent_labels.shape}"
            )

        """ Model forward """

        with timer() as t:
            latent_outputs = model(
                input_ids=latent_input_ids,
                attention_mask=latent_attention_mask,
                labels=latent_labels,
                use_cache=False,
                output_hidden_states=True,
            )

        time_elapsed["student_forward"] = t.elapsed

        loss_components["student_ce_loss"] = latent_outputs.loss

        """ Latent loss calculation(s) """

        with timer() as t:
            latent_loss_components, latent_statistics = calc_latent_loss(
                t_hs=outputs.hidden_states,
                s_hs=latent_outputs.hidden_states,
                batch=batch,
                device=device,
                training_config=training_config,
                model_config=strip_ddp(model).config,
            )

        time_elapsed["latent_loss_calc"] = t.elapsed

        loss_components.update(latent_loss_components)

        statistics.update(latent_statistics)

    loss = sum(loss_components.values())

    loss_components = {k: v.item() for k, v in loss_components.items()}
    loss_components["total_loss"] = loss.item()

    return loss, loss_components, time_elapsed, statistics


def step_w_gradient_accumulation(
    model, optimizer, training_config: TrainingConfig, batch: dict, micro_batch_size: int, is_ddp: bool
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    with timer() as train_t:
        first_key = next(iter(batch.keys()))

        # Separate batch into micro batches
        batch_size = len(batch[first_key])

        num_micro_batches = ceil(batch_size / micro_batch_size)
        micro_batches = [
            {k: v[i * micro_batch_size : (i + 1) * micro_batch_size] for k, v in batch.items()}
            for i in range(num_micro_batches)
        ]

        micro_batch_sizes = [len(micro_batch[first_key]) for micro_batch in micro_batches]
        whether_last_micro_batch = [False] * num_micro_batches
        whether_last_micro_batch[-1] = True

        # Lists to hold dicts output by get_loss_for_batch
        all_loss_components = []
        all_time_elapsed = []
        all_statistics = []

        optimizer.zero_grad()

        for micro_batch, micro_batch_size, is_last_micro_batch in zip(
            micro_batches, micro_batch_sizes, whether_last_micro_batch
        ):
            autocast_ctx = (
                torch.autocast(
                    device_type=strip_ddp(model).get_input_embeddings().weight.device.type,
                    dtype=STR_TO_TORCH_DTYPE[training_config.torch_dtype_str],
                )
                if training_config.torch_dtype_str is not None
                else nullcontext()
            )

            with autocast_ctx:
                loss, loss_components, time_elapsed, statistics = get_loss_for_batch(
                    model=model, batch=micro_batch, training_config=training_config
                )

                scale_factor = micro_batch_size / batch_size

                loss = loss * scale_factor

            sync_ctx = model.no_sync() if is_ddp and not is_last_micro_batch else nullcontext()

            with timer() as backward_t, sync_ctx:
                loss.backward()

            time_elapsed["backward"] = backward_t.elapsed

            # Scale the values of the loss components and statistics down so when we sum them later it is basically the same as a weighted mean
            loss_components = {k: scale_factor * v for k, v in loss_components.items()}
            statistics = {k: scale_factor * v for k, v in statistics.items()}

            all_loss_components.append(loss_components)
            all_time_elapsed.append(time_elapsed)
            all_statistics.append(statistics)

        clip_grad_norm_(model.parameters(), max_norm=training_config.max_grad_norm)
        optimizer.step()

    ## Combine the lists of dicts into one dict with the values summed
    loss_components = sum_dicts(all_loss_components)
    time_elapsed = sum_dicts(all_time_elapsed)
    statistics = sum_dicts(all_statistics)

    time_elapsed["total_time"] = train_t.elapsed

    return loss_components, time_elapsed, statistics


def train(
    model,
    tokenizer,
    training_config: TrainingConfig,
    val_config: EvalConfig,
    logging_dir: os.PathLike,
    checkpoint_manager: CheckpointManager,
    resuming: bool = False,
    device_selection_mode="m",
    time_limit: float | None = None,
    do_not_save: bool = False,
    min_micro_batch_size: int | None = None,
):
    training_start_time = time.perf_counter()

    """Args Validation"""

    if training_config.data_mode == "latent" or training_config.is_latent or val_config.latent_thinking:
        assert isinstance(strip_peft(model), LatentThinkingModel), (
            f"model must be a LatentThinkingModel if training_config.data_mode == 'latent' or training_config.is_latent or val_config.latent_thinking, got {type(model)}"
        )
        ensure_tokenizer_has_latent_tokens(tokenizer)

    # Validate batch size is power of 2
    # So that we can easily do gradient accumulation
    # I mean, why wouldn't you want batch size to be a power of 2? What kind of a monster are you??
    # A friendly one, I hope
    if not is_power_of_2(training_config.batch_size):
        raise ValueError(f"batch_size must be a power of 2, but got {training_config.batch_size}")

    """ DDP Setup """

    is_ddp = ddp_setup()
    if is_ddp:
        gpu_id = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    is_main_process = not is_ddp or rank == 0
    world_size = world_size if is_ddp else 1

    if not is_ddp:
        raise ValueError(
            "To enable consistency of DistributedSampler when resuming, please always run training w/ torchrun, even if planning on using a single device."
        )

    """ Logging """

    # Purpose is to guard creation of the main log dir until all processes get here. Otherwise, earlier checks that error if logging dir already exists may occur if some other process lags behind main process
    if is_ddp:
        dist.barrier()

    ### Create dirs

    logging_dir = Path(logging_dir)
    txt_log_path = logging_dir / "log.txt"

    if is_main_process:
        if not resuming and logging_dir.exists():
            raise ValueError(f"Logging directory {logging_dir} already exists!")
        elif resuming and not logging_dir.exists():
            raise ValueError(f"Logging directory {logging_dir} does not exist!")

        logging_dir.mkdir(parents=True, exist_ok=True)

        tensorboard_logdir = logging_dir / "runs"
        writer = SummaryWriter(tensorboard_logdir)

        with open(txt_log_path, "a") as f:
            f.write("\n---------------------------------------\n")
            f.write(f"Starting training on {str_formatted_datetime()}\n")
            f.write("-------------\n")
            f.write("Training Config:\n")
            f.write(get_obj_vars_str(training_config))
            f.write("-------------\n")
            f.write("Validation Config:\n")
            f.write(get_obj_vars_str(val_config))
            f.write("\n---------------------------------------\n")

    # Purpose is to guard use of printlog by processes until it is guarenteed that main process created logging directory
    if is_ddp:
        dist.barrier()

    printlog = PrintLog(txt_log_path, rank=rank if is_ddp else None)

    """ Post-log creation arg processing """

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        printlog("pad_token_id is not set! Setting it to eos_token_id")

    if model.is_gradient_checkpointing:
        printlog(
            f"Gradient checkpointing was detected on. To ensure consistency, gradient checkpointing will be turned off at the start of training. If it gets turned back on, it will get turned back on using the kwargs in _GRADIENT_CHECKPOINTING_KWARGS: {_GRADIENT_CHECKPOINTING_KWARGS}",
            warning=True,
        )
    model.gradient_checkpointing_disable()

    """ Prep data """

    val_dataset = get_tokenized_eval_dataset(tokenizer, val_config)
    train_dataset = get_tokenized_train_dataset(tokenizer, training_config)

    # Get per device batch size
    if is_ddp:
        if world_size > training_config.batch_size:
            printlog(
                f"world_size ({world_size}) was larger than training_config.batch_size ({training_config.batch_size})!",
                raise_exception=ValueError,
            )
        per_device_train_batch_size = round(training_config.batch_size / world_size)
        if training_config.batch_size % world_size:
            printlog(
                f"training_config.batch_size could not be cleanly divided by world size! training_config.batch_size was {training_config.batch_size} but world_size was {world_size}. The per_device_train_batch_size will be {per_device_train_batch_size}, which equates to a total batch size of {per_device_train_batch_size * world_size}. If you are resuming, this may cause slight mismatch as you might be resuming with a slightly different batch size",
                warning=True,
            )

    else:
        per_device_train_batch_size = training_config.batch_size

    train_collator = LatentDataCollator(pad_token_id=tokenizer.pad_token_id, padding_side="left")

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
        collate_fn=train_collator,
        sampler=DistributedSampler(train_dataset, seed=training_config.seed) if is_ddp else None,
        num_workers=training_config.num_workers,
        shuffle=False,
        persistent_workers=False,
    )

    """ Set up auto gradient accumulation/gradient checkpointing on OOM """

    if (
        (training_config.use_gradient_checkpointing_if_necessary and not training_config.use_gradient_checkpointing)
        and world_size > 1
        and is_main_process
    ):
        printlog(
            "use_gradient_checkpointing_if_necessary was True but world_size > 1 so GC will not be used since there may be problems if some but not all processes are using GC. If GC is desired, please set training_config.use_gradient_checkpointing=True instead",
            warning=True,
        )

    initial_micro_batch_size = training_config.initial_micro_batch_size
    if initial_micro_batch_size is None:
        initial_micro_batch_size = per_device_train_batch_size
    elif initial_micro_batch_size > per_device_train_batch_size:
        if is_main_process:
            printlog(
                f"initial_micro_batch_size {initial_micro_batch_size} was larger than per_device_train_batch_size! Defaulting to per_device_train_batch_size",
                warning=True,
            )
        initial_micro_batch_size = per_device_train_batch_size

    train_micro_batcher = AutoMicroBatch(
        micro_batch_starting_size=initial_micro_batch_size,
        allow_gradient_checkpointing=(world_size == 1 and training_config.use_gradient_checkpointing_if_necessary)
        or training_config.use_gradient_checkpointing,  # DDP may (mayyyyy) have issues if only some processes are using GC. It should be okay to use DDP if all processes start with GC. An option for that could be added, I just don't have the need for it right now
        allow_micro_batch=world_size
        == 1,  ## Adaptive micro-batching does not seem to work when there are multiple processes.
        print_prefix="[Training] ",
        printlog=printlog,
        min_micro_batch_size=min_micro_batch_size,
    )
    val_batcher = AutoMicroBatch(
        micro_batch_starting_size=val_config.batch_size or 2 * initial_micro_batch_size,
        allow_gradient_checkpointing=False,
        micro_batch_key="batch_size",
        print_prefix="[Validation] ",
        printlog=printlog,
        min_micro_batch_size=min_micro_batch_size,
    )

    """ Prep model, optimizer, scheduler """
    ## Prep device
    device = torch.device(f"cuda:{gpu_id}") if is_ddp else select_best_device(device_selection_mode)
    printlog(f"Device: {device}")

    ## Prep model
    model.to(device)
    model.train()

    if training_config.use_gradient_checkpointing:
        printlog("Activating gradient checkpointing!")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=_GRADIENT_CHECKPOINTING_KWARGS)

    if is_ddp:
        model = DDP(model, device_ids=[gpu_id])

    ## Prep optimizer
    if training_config.no_decay_some_params:
        optim_params = get_decay_grouped_params(model, decay_val=training_config.weight_decay)
    else:
        optim_params = model.parameters()

    optimizer = optim.AdamW(optim_params, lr=training_config.lr, weight_decay=training_config.weight_decay)

    ## Prep scheduler
    if training_config.scheduler_name is not None:
        if training_config.steps is not None:
            num_training_steps = training_config.steps
        else:
            num_training_steps = training_config.epochs * len(train_loader)
        scheduler = get_scheduler(
            training_config.scheduler_name,
            optimizer,
            num_warmup_steps=training_config.scheduler_num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        scheduler = None

    """ Load previous checkpoint if resuming """

    # For esoteric reasons, epoch_num is always the epoch the next training step belongs to
    epoch_num = 1
    # step_num is always the step number of the step you just finished
    step_num = 0
    if is_main_process:
        best_val_score = float("-inf")  # HIGHER is better

    if resuming:
        print(f"Resuming from: {checkpoint_manager.get_last_step_path_or_repo()}")

        checkpoint = checkpoint_manager.get_last_training_state(map_location=device)

        # Scheduler state must be restored BEFORE optimizer state: https://github.com/pytorch/pytorch/issues/119168
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_num = checkpoint["epoch_num"]
        step_num = checkpoint["step_num"]
        if is_main_process:
            best_val_score = checkpoint["best_val_score"]

    train_loader = ResumeStepDataLoader(train_loader, resume_step_num=step_num)

    """ Training loop """

    if is_ddp:
        dist.barrier()

    end_training = False
    while True:
        if is_ddp:
            train_loader.sampler.set_epoch(epoch_num)

        for batch in train_loader:
            """ Validation/Saving """

            is_best_val = False
            if is_main_process and step_num != 0 and not resuming:
                ## Validate
                if step_num % training_config.val_period == 0:
                    was_gradient_checkpointing = strip_ddp(model).is_gradient_checkpointing
                    if was_gradient_checkpointing:
                        strip_ddp(model).gradient_checkpointing_disable()

                    # Evaluate with support for retry with smaller batch size if OOM
                    autocast_ctx = (
                        torch.autocast(
                            device_type=device.type, dtype=STR_TO_TORCH_DTYPE[training_config.torch_dtype_str]
                        )
                        if training_config.torch_dtype_str is not None
                        else nullcontext()
                    )
                    with timer() as eval_t, torch.no_grad(), autocast_ctx:
                        eval_dict = val_batcher(
                            eval_from_config,
                            step_num=step_num,
                            epoch_num=epoch_num,
                            model=strip_ddp(model),
                            tokenizer=tokenizer,
                            eval_config=val_config,
                            dataset=val_dataset,
                            verbose=False,
                        )

                    if was_gradient_checkpointing:
                        strip_ddp(model).gradient_checkpointing_enable(_GRADIENT_CHECKPOINTING_KWARGS)

                    # Logging to tensorboard
                    for k, v in eval_dict.items():
                        if v is not None:
                            writer.add_scalar(f"val/{k}", v, step_num)
                    writer.add_scalar("time/eval", eval_t.elapsed, step_num)

                    # Update best_val_score
                    if eval_dict["score"] > best_val_score:
                        best_val_score = eval_dict["score"]
                        is_best_val = True

                if not do_not_save:
                    ## Saving
                    with timer() as save_t:
                        # Save checkpoint if best val
                        if is_best_val:
                            checkpoint_manager.save_checkpoint(
                                model=model,
                                tokenizer=tokenizer,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                training_config=training_config,
                                val_config=val_config,
                                epoch_num=epoch_num,
                                step_num=step_num,
                                best_val_score=best_val_score,
                                mode="best_val",
                            )

                        # Save checkpoint based on save_checkpoint_period
                        if (
                            training_config.save_checkpoint_period is not None
                            and step_num % training_config.save_checkpoint_period == 0
                        ):
                            checkpoint_manager.save_checkpoint(
                                model=model,
                                tokenizer=tokenizer,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                training_config=training_config,
                                val_config=val_config,
                                epoch_num=epoch_num,
                                step_num=step_num,
                                best_val_score=best_val_score,
                                mode="last_step",
                            )

                    writer.add_scalar("time/saving", save_t.elapsed, step_num)

            """ Logic to end training if criteria reached """

            if training_config.epochs is not None and epoch_num > training_config.epochs:
                printlog(f"Ending training as epoch count of {training_config.epochs} reached!")
                end_training = True
                break

            if training_config.steps is not None and step_num >= training_config.steps:
                printlog(f"Ending training as step count of {training_config.steps} reached!")
                end_training = True
                break

            training_elapsed_time = time.perf_counter() - training_start_time
            if time_limit is not None and training_elapsed_time / 3600 > time_limit:
                printlog(f"Ending training as time limit of {time_limit} hours reached!")
                end_training = True
                break

            """ Do the step """

            if is_ddp:
                dist.barrier()

            resuming = False  # Once we start the next step, all resume logic officially ends

            step_num += 1

            # Do batch step w/ support for gradient checkpointing/reduced micro batch size + retry if OOM
            loss_components, time_elapsed, statistics = train_micro_batcher(
                step_w_gradient_accumulation,
                step_num=step_num,
                epoch_num=epoch_num,
                model=model,
                optimizer=optimizer,
                training_config=training_config,
                batch=batch,
                is_ddp=is_ddp,
            )

            ## Logging to tensorboard
            if is_main_process:
                for k, v in loss_components.items():
                    writer.add_scalar(f"train_loss/{k}", v, step_num)
                for k, v in time_elapsed.items():
                    writer.add_scalar(f"time/train_absolute/{k}", v, step_num)
                proportional_time_elapsed = {k: v / time_elapsed["total_time"] for k, v in time_elapsed.items()}
                for k, v in proportional_time_elapsed.items():
                    writer.add_scalar(f"time/train_proportional/{k}", v, step_num)
                for k, v in statistics.items():
                    writer.add_scalar(f"train_statistics/{k}", v, step_num)
                writer.add_scalar(
                    "lr", scheduler.get_last_lr()[-1] if scheduler is not None else training_config.lr, step_num
                )

            if scheduler is not None:
                scheduler.step()

        if end_training:
            break

        epoch_num += 1

    if is_ddp:
        destroy_process_group()

    # Save final checkpoint
    if is_main_process:
        if not do_not_save:
            checkpoint_manager.save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                training_config=training_config,
                val_config=val_config,
                epoch_num=epoch_num,
                step_num=step_num,
                best_val_score=best_val_score,
                mode="last_step",
            )

            checkpoint_manager.wait_for_futures()

        writer.close()
