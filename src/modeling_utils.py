import torch
from torch import nn

START_TOKEN_STR = "<START>"
END_TOKEN_STR = "<END>"
CONTINUE_TOKEN_STR = "<|CONTINUE|>"
STOP_TOKEN_STR = "<|STOP|>"

SPECIAL_TOKEN_STRS = [START_TOKEN_STR, END_TOKEN_STR, CONTINUE_TOKEN_STR, STOP_TOKEN_STR]


def get_special_tokens_mapping_from_tokenizer(tokenizer) -> dict[str, int]:
    ensure_tokenizer_has_latent_tokens(tokenizer)
    special_token_to_id = {k: tokenizer.convert_tokens_to_ids(k) for k in SPECIAL_TOKEN_STRS}
    for k, v in special_token_to_id.items():
        if v is None:
            raise ValueError(f"Token {k} not found in tokenizer")
        elif not isinstance(v, int):
            raise ValueError(f"Unexpected type for token {k}. Got {type(v)}. Expected int")
    return special_token_to_id


def add_special_latent_tokens_to_tokenizer(
    tokenizer, replace_additional_special_tokens: bool = False
) -> dict[str, int]:
    tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKEN_STRS},
        replace_additional_special_tokens=replace_additional_special_tokens,
    )

    return get_special_tokens_mapping_from_tokenizer(tokenizer)


def ensure_tokenizer_has_latent_tokens(tokenizer, raise_error: bool = True) -> bool:
    has_latent_tokens = set(SPECIAL_TOKEN_STRS).issubset(set(tokenizer.additional_special_tokens))
    if raise_error and not has_latent_tokens:
        raise ValueError(
            f"Tokenizer must have the following additional special tokens: {SPECIAL_TOKEN_STRS}. But got {tokenizer.additional_special_tokens}"
        )
    return has_latent_tokens


class RecurrentFilterLinear(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size

        self.linear_layer = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.norm = nn.RMSNorm(self.hidden_size)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_layer(x)
        x = self.norm(x)
        x = x + self.bias
        return x


class RecurrentFilterLinearLN(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size

        self.linear_layer = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_layer(x)
        x = self.norm(x)
        return x


class RecurrentFilterMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size

        self.input_layer = nn.Linear(hidden_size, hidden_size)
        self.act_func = nn.functional.gelu
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.act_func(x)
        x = self.output_layer(x)
        x = self.norm(x)
        return x


def sanitize_logits_for_generation(logits: torch.Tensor, original_mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Sanitize logits to avoid NaN in softmax during generation by handling degenerate cases.

    This function fixes two problems that cause NaN in stable softmax:
      1) All logits are -inf → softmax does -inf - (-inf) = NaN
      2) Any +inf exists → softmax does +inf - (+inf) = NaN for that position

    Rules applied per-row along the last dimension (mutually exclusive by construction):
      1) If all logits in a row are -inf: set all to 1
      2) Else if any +inf exists in a row: set all to -inf, then set +inf positions to 1

    NaNs are first mapped to -inf globally (they can arise from inf + (-inf) during masking).

    Args:
        logits: Tensor of shape [..., vocab_size]. May be returned as-is if no sanitization needed.
        original_mask: Optional mask tensor. Positions where this == -inf will be forced back to -inf
                       after sanitization (preserves LoRA start token replacement and other mask semantics).

    Returns:
        Sanitized logits of the same shape. May be the input tensor if no changes were needed.
    """

    # Start from input; will only allocate new tensors when conditions trigger
    sanitized = logits

    # Replace any NaNs with -inf (NaNs can arise from inf + (-inf) during masking operations)
    nan_mask = torch.isnan(sanitized)
    if nan_mask.any():
        sanitized = torch.where(nan_mask, -float("inf"), sanitized)

    # Compute helper masks BEFORE any modifications (ensures cases are truly mutually exclusive)
    is_posinf = torch.isposinf(sanitized)
    is_neginf = torch.isneginf(sanitized)

    # Case 1: all logits in a row are -inf
    rows_all_neginf = is_neginf.all(dim=-1, keepdim=True)
    if rows_all_neginf.any():
        sanitized = torch.where(rows_all_neginf.expand_as(sanitized), 1, sanitized)

    # Case 2: any +inf exists in a row
    # (Mutually exclusive with Case 1: if all values were -inf, is_posinf would be all False)
    any_posinf_rows = is_posinf.any(dim=-1, keepdim=True)
    if any_posinf_rows.any():
        # Set all entries in those rows to -inf, then restore +inf entries to 1
        sanitized = torch.where(any_posinf_rows.expand_as(sanitized), -float("inf"), sanitized)
        sanitized = torch.where(any_posinf_rows.expand_as(sanitized) & is_posinf, 1, sanitized)

    # Re-apply original mask: restore -inf at positions that were masked
    if original_mask is not None:
        # Extract only the -inf positions from the mask and add them back
        # (This preserves any non-inf values like LoRA start token replacement)
        neg_inf_mask = torch.where(torch.isneginf(original_mask), -float("inf"), 0.0)
        sanitized = sanitized + neg_inf_mask

    return sanitized
