from functools import partial
import warnings
from datasets import load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.data_processing import filter_gsm8k_aug_answers, shuffle_and_slice
from src.modeling_utils import START_TOKEN_STR, ensure_tokenizer_has_latent_tokens


def get_gsm8k_aug_for_rl(
    append_start_token: bool,
    split: str = "train",
    max_prompt_tokens: int | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    slice_proportion: float | None = None,
    num_proc: int = 12,
    batch_size: int = 1024,
):
    """Checking / Processing Args"""

    if max_prompt_tokens is not None and tokenizer is None:
        raise ValueError("max_prompt_tokens requires a tokenizer to be provided")

    if append_start_token and tokenizer is not None:
        ensure_tokenizer_has_latent_tokens(tokenizer)

    if split != "train" and max_prompt_tokens is not None:
        warnings.warn("max_prompt_tokens is not supported for non-train splits")

    """ Get dataset """
    dataset = load_dataset("whyNLP/gsm8k-aug", split=split)

    if slice_proportion is not None:
        if split != "train":
            raise ValueError(f"slice_proportion is not supported for non-train splits. Got split='{split}' and slice_proportion={slice_proportion}")
        dataset = shuffle_and_slice(dataset, start_proportion=slice_proportion, end_proportion=1)

    """ Filter by answers (only for train split) """

    if split == "train":
        remove_negative_answers = False
        partial_filter_gsm8k_aug_answers = partial(
            filter_gsm8k_aug_answers, remove_negative_answers=remove_negative_answers
        )
        dataset = dataset.filter(
            partial_filter_gsm8k_aug_answers,
            num_proc=num_proc,
            desc=f"Filtering rows for valid answers{' and removing negative answers' if remove_negative_answers else ''}",
        )

    """ Remove steps """

    dataset = dataset.remove_columns("steps")

    """ Add prompt_id to each row """

    dataset = dataset.map(
        lambda example, idx: {"prompt_id": idx}, with_indices=True, num_proc=num_proc, desc="Adding prompt_id"
    )

    """ Rename columns """

    column_mapping = {"question": "prompt", "answer": "label"}
    dataset = dataset.rename_columns(column_mapping)

    """ Append <START> token """

    if append_start_token:

        def _append_start_token(example):
            example["prompt"] += START_TOKEN_STR
            return example

        dataset = dataset.map(_append_start_token, num_proc=num_proc, desc=f"Appending {START_TOKEN_STR} token")

    """ Filter by prompt token length (only for train split) """

    if split == "train" and max_prompt_tokens:

        def batched_max_token_filter(examples) -> list[bool]:
            tokenized = tokenizer(examples["prompt"], truncation=False, padding=False, add_special_tokens=True)
            within_max = [len(input_ids) <= max_prompt_tokens for input_ids in tokenized["input_ids"]]
            return within_max

        dataset = dataset.filter(
            batched_max_token_filter,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc=f"Filtering for max prompt token length ({max_prompt_tokens})",
        )

    return dataset


