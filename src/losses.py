import torch
from torch.nn.functional import smooth_l1_loss
from transformers.configuration_utils import PretrainedConfig

from src.training_utils import LATENT_LOSS_CALC_MODES, LATENT_LOSS_NORM_MODES, TrainingConfig
from src.utils import FloatingTensor


def layerwise_smooth_l1_loss(
    input: FloatingTensor, target: FloatingTensor, layer_dim: int, smooth_l1_kwargs: dict | None = None
) -> FloatingTensor:
    """
    Args:
        input (FloatingTensor): Hidden states of shape (..., num_layers, ..., hidden_size)
        target (FloatingTensor): Target tensor broadcastable to `input`'s shape
        layer_dim (int): The dimension of the layers. 0-indexed. MUST BE NON-NEGATIVE
        smooth_l1_kwargs (dict | None): Extra kwargs forwarded to torch.nn.functional.smooth_l1_loss.
            The 'reduction' argument is overridden to 'none'.

    Returns:
        FloatingTensor: Tensor of shape (num_layers,) representing the mean SmoothL1 loss for each layer
    """

    """ Checks """

    if layer_dim < 0:
        raise ValueError(f"layer_dim must be non-negative. But got {layer_dim}")
    if layer_dim >= input.dim():
        raise ValueError(
            f"layer_dim must be less than the number of dimensions of input. But got {layer_dim} >= {input.dim()}"
        )
    if layer_dim == input.dim() - 1:
        raise ValueError(
            "layer_dim must not be the last dimension of input, since the last dimension is the hidden size."
        )

    """ DO it """

    if smooth_l1_kwargs is None:
        smooth_l1_kwargs = {}

    if "reduction" in smooth_l1_kwargs:
        raise NotImplementedError(
            f"Got 'reduction' kwarg '{smooth_l1_kwargs['reduction']}' in smooth_l1_kwargs. If you want to sum the layerwise loss so bad you'll have to implement it yourself! If you want mean reduction, this already does that..."
        )

    loss = smooth_l1_loss(input, target, reduction="none", **smooth_l1_kwargs)
    dims_to_mean_across = tuple(i for i in range(loss.dim()) if i != layer_dim)
    layerwise_loss = loss.mean(dim=dims_to_mean_across)

    return layerwise_loss


def get_layerwise_average_l2(hidden_states: FloatingTensor, layer_dim: int) -> FloatingTensor:
    """
    Args:
        hidden_states (FloatingTensor): Hidden state of shape (..., num_layers, ..., hidden_size)
        layer_dim (int): The dimension of the layers. 0-indexed. MUST BE NON-NEGATIVE

    Returns:
        FloatingTensor: Tensor of shape (num_layers,) representing the average l2 of hidden states for each layer
    """

    """ Checks """
    if layer_dim < 0:
        raise ValueError(f"layer_dim must be non-negative. But got {layer_dim}")
    if layer_dim >= hidden_states.dim():
        raise ValueError(
            f"layer_dim must be less than the number of dimensions of hidden_states. But got {layer_dim} >= {hidden_states.dim()}"
        )
    if layer_dim == hidden_states.dim() - 1:
        raise ValueError(
            "layer_dim must not be the last dimension of hidden_states, since the last dimension is the hidden size."
        )

    """ DO it """

    lengths = hidden_states.norm(dim=-1)
    dims_to_mean_across = tuple(i for i in range(lengths.dim()) if i != layer_dim)
    lengths = lengths.mean(dim=dims_to_mean_across)

    return lengths


def get_average_l2(hidden_states: FloatingTensor) -> FloatingTensor:
    """
    Args:
        hidden_states (FloatingTensor): Hidden state of shape (..., hidden_size)

    Returns:
        FloatingTensor: Scalar tensor representing the average l2 of hidden states
    """

    lengths = hidden_states.norm(dim=-1)
    average_length = lengths.mean()

    return average_length


def get_max_l2(hidden_states: FloatingTensor) -> FloatingTensor:
    """
    Args:
        hidden_states (FloatingTensor): Hidden state of shape (..., hidden_size)

    Returns:
        FloatingTensor: Scalar tensor representing the maximum l2 of hidden states
    """

    lengths = hidden_states.norm(dim=-1)
    max_length = lengths.max()

    return max_length


def get_min_l2(hidden_states: FloatingTensor) -> FloatingTensor:
    """
    Args:
        hidden_states (FloatingTensor): Hidden state of shape (..., hidden_size)

    Returns:
        FloatingTensor: Scalar tensor representing the minimum l2 of hidden states
    """

    lengths = hidden_states.norm(dim=-1)
    min_length = lengths.min()

    return min_length


def _check_latent_loss_inputs(
    teacher_hidden_states: FloatingTensor,
    teacher_map: torch.BoolTensor,
    student_hidden_states: FloatingTensor,
    student_map: torch.BoolTensor,
) -> tuple[int, int, int]:
    """

    Args:
        teacher_hidden_states (FloatingTensor): The hidden states of the model. Must have shape (batch_size, teacher_sequence_length, num_blocks, hidden_size).
            * Make sure the teacher hidden states are detached from the computational graph!
        teacher_map (torch.BoolTensor): A boolean tensor representing the elements of teacher_hidden_states to be used for loss calculation. Shape (batch_size, teacher_sequence_length)
        student_hidden_states (FloatingTensor): The hidden states of the student model output. Shape (batch_size, student_sequence_length, num_blocks, hidden_size)
        student_map (torch.BoolTensor): Boolean tensor map for the student. Shape (batch_size, student_sequence_length)

    Returns:
        tuple[int, int, int]: The sizes (batch_size, num_blocks, hidden_size)

    Raises:
        ValueError: If shapes or dtypes don't match expected values
    """

    ## Check shapes

    batch_size, t_seq_len, num_blocks, hidden_size = teacher_hidden_states.shape
    s_batch_size, s_seq_len, s_num_blocks, s_hidden_size = student_hidden_states.shape

    if (s_batch_size, s_num_blocks, s_hidden_size) != (batch_size, num_blocks, hidden_size):
        raise ValueError(
            f"Student hidden states must match teacher in batch_size, num_blocks, and hidden_size. Got student: {student_hidden_states.shape}, expected: ({batch_size}, *, {num_blocks}, {hidden_size})"
        )

    if teacher_map.shape != (batch_size, t_seq_len):
        raise ValueError(
            f"Teacher map must have shape (batch_size, t_sequence_length). Got: {teacher_map.shape}, expected: ({batch_size}, {t_seq_len})"
        )
    if student_map.shape != (batch_size, s_seq_len):
        raise ValueError(
            f"Student map must have shape (batch_size, s_sequence_length). Got: {student_map.shape}, expected: ({batch_size}, {s_seq_len})"
        )

    ## Check dtypes

    if teacher_map.dtype != torch.bool:
        raise ValueError(f"teacher_map must be a boolean tensor. Got {teacher_map.dtype}")
    if student_map.dtype != torch.bool:
        raise ValueError(f"student_map must be a boolean tensor. Got {student_map.dtype}")

    # make sure hs are float tensors
    if not teacher_hidden_states.is_floating_point():
        raise ValueError(f"Teacher hidden states must be a floating point tensor. Got {teacher_hidden_states.dtype}")
    if not student_hidden_states.is_floating_point():
        raise ValueError(f"Student hidden states must be a floating point tensor. Got {student_hidden_states.dtype}")

    ## Make sure teacher hidden states are detached
    if teacher_hidden_states.requires_grad:
        raise ValueError(
            f"Teacher hidden states must be detached from the computational graph. Got: teacher_hidden_states.requires_grad = {teacher_hidden_states.requires_grad}"
        )

    return (batch_size, num_blocks, hidden_size)


def extract_target_token_hidden_states(
    teacher_hidden_states: FloatingTensor,
    teacher_target_token_map: torch.BoolTensor,
    student_hidden_states: FloatingTensor,
    student_target_token_map: torch.BoolTensor,
) -> tuple[FloatingTensor, FloatingTensor]:
    """

    Args:
        teacher_hidden_states (FloatingTensor): The hidden states of the model. Must have shape (batch_size, t_sequence_length, num_blocks, hidden_size).
            * Make sure the teacher hidden states are detached from the computational graph!
        teacher_target_token_map (torch.BoolTensor): A boolean tensor representing the position of the target token. Shape (batch_size, t_sequence_length)
        student_hidden_states (FloatingTensor): The hidden states of the student model output.
        student_target_token_map (torch.BoolTensor): Target token map for the student

    Returns:
        FloatingTensor: Returns the extracted target hidden states for student and teacher, both of shape. (batch_size, num_blocks, hidden_size)
    """

    """ Checking inputs """

    batch_size, num_blocks, hidden_size = _check_latent_loss_inputs(
        teacher_hidden_states=teacher_hidden_states,
        teacher_map=teacher_target_token_map,
        student_hidden_states=student_hidden_states,
        student_map=student_target_token_map,
    )

    # Check that each target token map contains exactly one true per row
    if (teacher_target_token_map.sum(dim=1) != 1).any():
        raise ValueError(
            f"Teacher target token map must contain exactly one true per row. Got: {teacher_target_token_map}"
        )
    if (student_target_token_map.sum(dim=1) != 1).any():
        raise ValueError(
            f"Student target token map must contain exactly one true per row. Got: {student_target_token_map}"
        )

    ## Extract Target token hidden states
    # Both have shape (batch_size, num_blocks, hidden_size)
    teacher_token_hidden_states = teacher_hidden_states[teacher_target_token_map]
    student_token_hidden_states = student_hidden_states[student_target_token_map]
    # Check shape
    for hs in (teacher_token_hidden_states, student_token_hidden_states):
        if hs.shape != (batch_size, num_blocks, hidden_size):
            raise ValueError(
                f"Target hidden states must have shape (batch_size, num_blocks, hidden_size). Got: {hs.shape}, expected: ({batch_size}, {num_blocks}, {hidden_size})"
            )

    return teacher_token_hidden_states, student_token_hidden_states


def _mean_reasoning_tokens(hs: FloatingTensor, reasoning_map: torch.BoolTensor):
    """
    Args:
        hs (FloatingTensor): Float tensor of shape (batch_size, seq_len, num_blocks, hidden_size) representing model output hidden states
        reasoning_map (torch.BoolTensor): Boolean tensor of shape (batch_size, seq_len)

    Returns:
        FloatingTensor: Float tensor of shape (batch_size, num_blocks, hidden_size) representing the mean hidden state across the sequence dimension.
    """

    ## Multiply hidden states by reasoning map to mask out hidden states that don't correspond to reasoning tokens
    # The [..., None, None] is unsqueezing to transform the shapes of the maps into (batch_size, seq_len, 1, 1) for broadcasting with the hidden states
    hs = hs * reasoning_map[..., None, None]

    # Sum across the sequence dimension
    hs = hs.sum(dim=1)  # Shape (batch_size, num_blocks, hidden_size)

    # Get number of reasoning tokens for each batch
    n_reasoning_per_batch = reasoning_map.sum(dim=1)  # Shape (batch_size)

    # Replace all occurrences of 0 with 1 to prevent division by 0
    n_reasoning_per_batch = torch.where(n_reasoning_per_batch == 0, 1, n_reasoning_per_batch)

    # Divide each batch by the number of reasoning tokens it had
    hs = hs / n_reasoning_per_batch[..., None, None]

    return hs


def mean_reasoning_tokens(
    teacher_hidden_states: FloatingTensor,
    teacher_reasoning_map: torch.BoolTensor,
    student_hidden_states: FloatingTensor,
    student_reasoning_map: torch.BoolTensor,
    training_config: TrainingConfig | None = None,
) -> tuple[FloatingTensor, FloatingTensor]:
    """

    Args:
        teacher_hidden_states (FloatingTensor): The hidden states of the model. Must have shape (batch_size, t_sequence_length, num_blocks, hidden_size).
            * Make sure the teacher hidden states are detached from the computational graph!
        teacher_reasoning_map (torch.BoolTensor): A boolean tensor representing the positions of the reasoning tokens. Shape (batch_size, t_sequence_length)
        student_hidden_states (FloatingTensor): The hidden states of the student model output.
        student_reasoning_map (torch.BoolTensor): Reasoning map for the student
        training_config (TrainingConfig): Training config, used for validation purposes. If None, this validation will not occur

    Returns:
        tuple[FloatingTensor, FloatingTensor]: A tuple containing (teacher_meaned_hs, student_meaned_hs), each with shape (batch_size, num_blocks, hidden_size)
    """

    """ Checking inputs """

    batch_size, num_blocks, hidden_size = _check_latent_loss_inputs(
        teacher_hidden_states=teacher_hidden_states,
        teacher_map=teacher_reasoning_map,
        student_hidden_states=student_hidden_states,
        student_map=student_reasoning_map,
    )

    # Make sure there aren't too many or too little student reasoning tokens per sequence
    if training_config is not None:
        min_latent_steps = training_config.min_latent_thinking_steps
        max_latent_steps = training_config.max_latent_thinking_steps

        if min_latent_steps is not None or max_latent_steps is not None:
            num_student_reasoning_tokens = student_reasoning_map.sum(dim=1)  # shape (batch_size)
            if min_latent_steps is not None and (num_student_reasoning_tokens < min_latent_steps).any():
                raise ValueError(
                    f"Student reasoning map must have at least {min_latent_steps} reasoning tokens per batch element. But got batch-wise reasoning token counts: {num_student_reasoning_tokens}"
                )
            if max_latent_steps is not None and (num_student_reasoning_tokens > max_latent_steps).any():
                raise ValueError(
                    f"Student reasoning map must have at most {max_latent_steps} reasoning tokens per batch element. But got batch-wise reasoning token counts: {num_student_reasoning_tokens}"
                )

    """ Get the mean reasoning tokens """

    with torch.no_grad():
        t_meaned_hs = _mean_reasoning_tokens(hs=teacher_hidden_states, reasoning_map=teacher_reasoning_map)
    s_meaned_hs = _mean_reasoning_tokens(hs=student_hidden_states, reasoning_map=student_reasoning_map)

    ## In all locations where the teacher did not have any reasoning tokens, mask the student out too so we aren't enforcing them to go to 0
    teacher_reasoning_mask = teacher_reasoning_map.any(dim=1)[..., None, None]  # Shape (batch_size, 1, 1)
    if not teacher_reasoning_mask.all():
        s_meaned_hs = s_meaned_hs * teacher_reasoning_mask

    # Check shape
    for hs in (t_meaned_hs, s_meaned_hs):
        if hs.shape != (batch_size, num_blocks, hidden_size):
            raise ValueError(
                f"Meaned states must have shape (batch_size, num_blocks, hidden_size). Got: {hs.shape}, expected: ({batch_size}, {num_blocks}, {hidden_size})"
            )

    return t_meaned_hs, s_meaned_hs


def slice_and_stack_hidden_states(
    hidden_states: tuple[FloatingTensor], block_range: tuple[int, int] | list[int, int] | None, detach: bool = False
) -> FloatingTensor:
    # Exclude initial embeddings
    hidden_states = hidden_states[1:]

    # Slice according to block range
    if block_range is not None:
        block_start, block_end = block_range

        # Validate block range
        num_hidden_layers = len(hidden_states)
        if not (0 <= block_start <= block_end < num_hidden_layers):
            raise ValueError(
                f"Invalid block range: block_start={block_start}, block_end={block_end} for {num_hidden_layers} layers. Must satisfy 0 <= block_start <= block_end < num_hidden_layers. Reminder: block_range is 0-indexed and inclusive on both ends"
            )

        hidden_states = hidden_states[block_start : block_end + 1]

    # Detach if specified
    if detach:
        hidden_states = [hidden_state.detach() for hidden_state in hidden_states]

    # Stack
    # Form into a tensor by stacking on dim 2. New shape: (batch_size, seq_len, num_blocks_specified, hidden_size)
    hidden_states = torch.stack(hidden_states, dim=2)

    return hidden_states


def calc_latent_loss(
    t_hs: tuple[FloatingTensor],
    s_hs: tuple[FloatingTensor],
    batch: dict,
    device,
    training_config: TrainingConfig,
    model_config: PretrainedConfig,
    eps: float = 1e-8,
) -> tuple[dict[str, FloatingTensor], dict[str, float]]:
    """Initial Setup"""

    ## Config validation
    if not training_config.is_latent:
        raise ValueError("calc_latent_loss can only be used in latent mode. How did we even get here?")

    if training_config.latent_loss_calc_mode not in LATENT_LOSS_CALC_MODES:
        raise ValueError(
            f"latent_loss_calc_mode must be one of {LATENT_LOSS_CALC_MODES}. But got {training_config.latent_loss_calc_mode}"
        )
    if training_config.latent_loss_norm_mode not in LATENT_LOSS_NORM_MODES:
        raise ValueError(
            f"latent_loss_norm_mode must be one of {LATENT_LOSS_NORM_MODES}. But got {training_config.latent_loss_norm_mode}"
        )

    loss_components = {}
    statistics = {}

    """ Get maps from batch """

    reasoning_map = batch["reasoning_map"]
    target_token_map = batch["target_token_map"]
    latent_continue_map = batch["latent_continue_map"]
    latent_target_token_map = batch["latent_target_token_map"]

    """ Check shapes and dtypes """

    # Extract various shapes
    num_hidden_layers = model_config.num_hidden_layers
    hidden_size = model_config.hidden_size
    batch_size, t_seq_len = batch["input_ids"].shape
    _, s_seq_len = batch["latent_input_ids"].shape

    # Ensure all maps are BoolTensors
    if reasoning_map.dtype != torch.bool:
        raise ValueError(f"reasoning_map must be a boolean tensor. But got {reasoning_map.dtype}")
    if target_token_map.dtype != torch.bool:
        raise ValueError(f"target_token_map must be a boolean tensor. But got {target_token_map.dtype}")
    if latent_continue_map.dtype != torch.bool:
        raise ValueError(f"latent_continue_map must be a boolean tensor. But got {latent_continue_map.dtype}")
    if latent_target_token_map.dtype != torch.bool:
        raise ValueError(f"latent_target_token_map must be a boolean tensor. But got {latent_target_token_map.dtype}")

    if not (reasoning_map.shape == target_token_map.shape == (batch_size, t_seq_len)):
        raise ValueError(
            f"reasoning_map and target_token_map must have the same shape and be (batch_size, t_seq_len). But got {reasoning_map.shape} and {target_token_map.shape}. Expected: ({batch_size}, {t_seq_len})"
        )
    if not (latent_continue_map.shape == latent_target_token_map.shape == (batch_size, s_seq_len)):
        raise ValueError(
            f"latent_continue_map and latent_target_token_map must have the same shape and be (batch_size, s_seq_len). But got {latent_continue_map.shape} and {latent_target_token_map.shape}. Expected: ({batch_size}, {s_seq_len})"
        )

    # Make sure shapes are consistent

    if len(t_hs) != num_hidden_layers + 1:
        # + 1 due to initial embeddings
        raise ValueError(f"t_hs must have length num_hidden_layers + 1. But got {len(t_hs)}")
    if len(s_hs) != num_hidden_layers + 1:
        raise ValueError(f"s_hs must have length num_hidden_layers + 1. But got {len(s_hs)}")

    """ Slice and stack hidden states """

    block_range = training_config.intermediate_block_latent_loss_range

    t_hs = slice_and_stack_hidden_states(t_hs, block_range=block_range, detach=True)

    s_hs = slice_and_stack_hidden_states(s_hs, block_range=block_range)

    """ More shape checks """

    if block_range is not None:
        block_start, block_end = block_range
        num_blocks = block_end - block_start + 1
    else:
        num_blocks = num_hidden_layers

    if s_hs.shape != (batch_size, s_seq_len, num_blocks, hidden_size):
        raise ValueError(f"s_hs must have shape (batch_size, s_seq_len, num_blocks, hidden_size). But got {s_hs.shape}")

    if t_hs.shape != (batch_size, t_seq_len, num_blocks, hidden_size):
        raise ValueError(f"t_hs must have shape (batch_size, t_seq_len, num_blocks, hidden_size). But got {t_hs.shape}")

    """ Extract target layer-wise hidden states """

    if training_config.mean_reasoning_loss:
        reasoning_map = reasoning_map.to(device)
        latent_continue_map = latent_continue_map.to(device)
        loss_components["mean_reasoning_loss"] = mean_reasoning_tokens(
            teacher_hidden_states=t_hs,
            teacher_reasoning_map=reasoning_map,
            student_hidden_states=s_hs,
            student_reasoning_map=latent_continue_map,
            training_config=training_config,
        )

    if training_config.codi_loss:
        target_token_map = target_token_map.to(device)
        latent_target_token_map = latent_target_token_map.to(device)
        loss_components["codi_loss"] = extract_target_token_hidden_states(
            teacher_hidden_states=t_hs,
            teacher_target_token_map=target_token_map,
            student_hidden_states=s_hs,
            student_target_token_map=latent_target_token_map,
        )

    ## Validation of shapes because I have allllll the time in world!!

    for k, target_hss in loss_components.items():
        for target_hs in target_hss:
            if target_hs.shape != (batch_size, num_blocks, hidden_size):
                raise ValueError(
                    f"Target hidden states must have shape (batch_size, num_blocks, hidden_size). But got {target_hs.shape} for key {k}"
                )

    # Statistics
    with torch.no_grad():
        for k, (t_target_hs, s_target_hs) in loss_components.items():
            statistics["pre-norm avg-l2/teacher/" + k] = get_average_l2(t_target_hs).item()
            statistics["pre-norm avg-l2/student/" + k] = get_average_l2(s_target_hs).item()
            statistics["pre-norm max-l2/teacher/" + k] = get_max_l2(t_target_hs).item()
            statistics["pre-norm max-l2/student/" + k] = get_max_l2(s_target_hs).item()
            statistics["pre-norm min-l2/teacher/" + k] = get_min_l2(t_target_hs).item()
            statistics["pre-norm min-l2/student/" + k] = get_min_l2(s_target_hs).item()

    """ Calculate loss + normalize if applicable """

    for k, (t_target_hs, s_target_hs) in loss_components.items():
        ## Calculate the loss
        if training_config.latent_loss_calc_mode == "mean_l2":
            layerwise_loss = get_layerwise_average_l2(t_target_hs - s_target_hs, layer_dim=1)
        elif training_config.latent_loss_calc_mode == "smooth_l1":
            layerwise_loss = layerwise_smooth_l1_loss(s_target_hs, t_target_hs, layer_dim=1)
        else:
            raise ValueError("Something has gone very wrong. Call a priest probably")

        ## Normalize if applicable
        if training_config.latent_loss_norm_mode in ["layerwise_avg_l2", "layerwise_std", "avg_l2", "std"]:
            ## Get scaling factor
            with torch.no_grad():
                if training_config.latent_loss_norm_mode == "layerwise_avg_l2":
                    scaling_factor = get_layerwise_average_l2(t_target_hs, layer_dim=1)  # Shape (num_blocks,)
                elif training_config.latent_loss_norm_mode == "layerwise_std":
                    scaling_factor = t_target_hs.std(dim=(0, 2))  # Shape (num_blocks,)
                elif training_config.latent_loss_norm_mode == "avg_l2":
                    scaling_factor = get_average_l2(t_target_hs)
                elif training_config.latent_loss_norm_mode == "std":
                    scaling_factor = t_target_hs.std()
                else:
                    raise ValueError("Something has gone very wrong. Call a priest maybe")

                scaling_factor = scaling_factor + eps

                statistics[training_config.latent_loss_norm_mode + "-norm/mean_scaling_factor/" + k] = (
                    scaling_factor.mean().item()
                )

            layerwise_loss = layerwise_loss / scaling_factor

        # Mean loss across layers
        loss = layerwise_loss.mean()

        loss_components[k] = loss

    """ Apply component weights + return """

    # Apply loss weights to loss components
    if training_config.mean_reasoning_loss:
        loss_components["mean_reasoning_loss"] = (
            loss_components["mean_reasoning_loss"] * training_config.mean_reasoning_loss
        )
    if training_config.codi_loss:
        loss_components["codi_loss"] = loss_components["codi_loss"] * training_config.codi_loss

    statistics = {"latent/" + k: v for k, v in statistics.items()}

    return loss_components, statistics
