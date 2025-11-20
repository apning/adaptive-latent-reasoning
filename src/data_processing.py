import copy
from functools import partial
from tqdm.auto import tqdm

import numpy as np
from datasets import Dataset, load_dataset

from src.modeling_utils import START_TOKEN_STR, CONTINUE_TOKEN_STR, STOP_TOKEN_STR, ensure_tokenizer_has_latent_tokens

SUPPORTED_TRAINING_DATA_NAMES = {
    "gsm8k-aug",
    "gsm8k-aug-nl",
}

SUPPORTED_EVAL_DATA_NAMES = {
    "gsm8k-aug",
    "gsm8k-aug-nl",
}


DEMARCATE_STR = "<DEMARCATE>"


def token_in_vocab(tokenizer, token):
    """Check if a token exists in the tokenizer's vocabulary"""
    if not isinstance(token, str):
        raise ValueError(f"Token must be a string. Got {type(token)}")
    token_id = tokenizer.convert_tokens_to_ids(token)
    return token_id != tokenizer.unk_token_id and token_id is not None


def _remove_demarc_add_labels_and_maps(
    example: dict, demarcate_id: int, mask_reasoning_labels: bool, shift_target_token_one: bool
) -> dict:
    """
    Remove demarcation ids from the input_ids and attention_mask. Uses the former positions of the demarcation ids to construct:
        * labels, a list of ints with same shape as input_ids that has the question tokens masked out with -100
        * reasoning_map, a list of bools with same shape as input_ids where all reasoning tokens are True and all others False
        * target_token_map, a list of bools with same shape as input_ids where the target token is True and all others False

    Made for dataset.map(...). Needs to be used with functools partial which will provide demarcate_id before being given to .map(...)

    Args:
        example (dict): A dictionary containing the keys "input_ids" and "attention_mask", both a single list containing one data point (NOT BATCHED). "input_ids" must contain 4 instances of demarcate_id
        demarcate_id (int): Id of special demarcation token
        mask_reasoning_labels (bool): Whether the labels for the reasoning tokens should be masked (ignored with index -100)
        shift_target_token_one (bool): If True, target token will be shifted one down (+1 position)

    Returns:
        dict: Containing modified "input_ids" and "attention_mask". Also with new keys "labels", "reasoning_map", and "target_token_map"

    """
    input_ids = example["input_ids"]
    attention_mask = example["attention_mask"]

    input_ids = np.array(input_ids)
    attention_mask = np.array(attention_mask)

    # Get index of all demarcation tokens
    demarcation_idxs = np.where(input_ids == demarcate_id)[0]
    if len(demarcation_idxs) != 4:
        raise ValueError(f"Expected 4 demarcation tokens in the input_ids. But got {len(demarcation_idxs)}")

    # Remove demarcation tokens from input_ids and attention_mask
    input_ids = np.delete(input_ids, demarcation_idxs)
    attention_mask = np.delete(attention_mask, demarcation_idxs)

    # Unpack original demarcation indices (before deletion)
    i1, i2, i3, i4 = demarcation_idxs

    # Create labels as a copy of input_ids and mask questions (tokens before the first demarcation)
    labels = input_ids.copy()
    labels[:i1] = -100

    # Create reasoning_map
    # everything that isn't part of the steps tokens are False. Steps tokens are True
    # Reasoning tokens are those originally in-between i1 and i2
    reasoning_map = np.full_like(input_ids, False, dtype=bool)

    # After deletion, indices shift:
    # Tokens originally in (i1, i2) map to [i1, i2-1) in the new array
    steps_start = i1
    steps_end = i2 - 1
    if steps_start < steps_end:
        reasoning_map[steps_start:steps_end] = True

    # Create target_token_map
    # everything that isn't the target token is False. The target token is True
    # The target token was originally in-between i3 and i4
    target_token_map = np.full_like(input_ids, False, dtype=bool)
    if i4 - i3 != 2:
        raise ValueError(
            f"Expected only one target token in the input_ids. But there were {i4 - i3 - 1} tokens between i3 and i4!"
        )
    target_token_map[i3 - 2 + shift_target_token_one] = True

    # If mask_reasoning_labels is True then mask out the reasoning labels
    if mask_reasoning_labels:
        labels[reasoning_map] = -100

    # Convert back to lists and return
    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": labels.tolist(),
        "reasoning_map": reasoning_map.tolist(),
        "target_token_map": target_token_map.tolist(),
    }


def remove_demarc_add_labels_and_maps(
    example: dict, demarcate_id: int, mask_latent_reasoning_labels: bool, shift_target_token_one: bool
) -> dict:
    latent_example = {}
    for k in list(example.keys()):
        if "latent_" in k:
            latent_example[k] = example.pop(k)

    example = _remove_demarc_add_labels_and_maps(
        example, demarcate_id, mask_reasoning_labels=False, shift_target_token_one=shift_target_token_one
    )

    if latent_example:
        latent_example = {k.split("latent_")[-1]: v for k, v in latent_example.items()}
        latent_example = _remove_demarc_add_labels_and_maps(
            latent_example,
            demarcate_id,
            mask_reasoning_labels=mask_latent_reasoning_labels,
            shift_target_token_one=shift_target_token_one,
        )
        latent_example = {"latent_" + k: v for k, v in latent_example.items()}
        example.update(latent_example)

    return example


def _check_and_demarcate_tokenizer(tokenizer, latent: bool = False):
    if latent:
        ensure_tokenizer_has_latent_tokens(tokenizer)

    # Deepcopy tokenizer so as to not affect the original. Then add special thought demarcation token
    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [DEMARCATE_STR]}, replace_additional_special_tokens=False
    )

    demarcate_id = tokenizer.convert_tokens_to_ids(DEMARCATE_STR)
    eos_str = tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)

    return tokenizer, demarcate_id, eos_str


def filter_gsm8k_aug_answers(example, remove_negative_answers: bool):
    """
    GSM8k-Aug and GSM8k-Aug-NL from whynlp contains a small number of answers which are not plain numbers in the train split (194/385620 samples).
        * Eg. they have other symbols or actual text
        * 193 cannot be converted to float
        * A single one, which is just the text 'infinity', can be converted to float by Python's float(...). So we separately filter that out

    All original GSM8k answers are plain numbers, so this does not seem intentional.
    This function is used with dataset.filter(...) to filter these out.
    """
    answer = example["answer"]
    if "inf" in answer:
        return False
    try:
        ans = float(answer.replace(",", ""))
        if remove_negative_answers and ans < 0:
            return False
        return True
    except ValueError:
        return False


def shuffle_and_slice(dataset: Dataset, start_proportion: float, end_proportion: float, seed=42):

    ''' Arg Checks '''
    
    if not (0 <= start_proportion <= 1):
        raise ValueError(f"start_proportion must be between 0 and 1, got {start_proportion}")
    if not (0 <= end_proportion <= 1):
        raise ValueError(f"end_proportion must be between 0 and 1, got {end_proportion}")
    
    if start_proportion >= end_proportion:
        raise ValueError(f"start_proportion must be less than end_proportion, got start_proportion={start_proportion}, end_proportion={end_proportion}")

    ''' Shuffle + Slice '''

    dataset = dataset.shuffle(seed=seed)

    dataset_len = len(dataset)

    start_idx = round(start_proportion * dataset_len)
    end_idx = round(end_proportion * dataset_len)

    dataset = dataset.select(range(start_idx, end_idx))

    return dataset




def tokenize_gsm8k_aug_for_training(
    tokenizer,
    natural_language: bool = True,
    mode: str = "cot",
    remove_last_step: bool = False,
    latent_thinking_factor: int | None = None,
    latent_thinking_bias: int | None = None,
    min_latent_thinking_steps: int | None = None,
    max_latent_thinking_steps: int | None = None,
    mask_latent_reasoning_labels: bool = False,
    remove_negative_answers: bool = False,
    shift_target_token_one: bool = False,
    max_tokens: int | None = None,
    split: str = "train",
    slice_proportion: float | None = None,
    num_proc: int = 12,
    batch_size: int = 1024,
    keep_columns: bool = False,
):
    """Check args"""

    if mode != "latent" and (
        latent_thinking_factor is not None
        or latent_thinking_bias is not None
        or min_latent_thinking_steps is not None
        or max_latent_thinking_steps is not None
        or mask_latent_reasoning_labels
    ):
        raise ValueError(
            f"latent_thinking_factor, latent_thinking_bias, min_latent_thinking_steps, max_latent_thinking_steps, and mask_latent_reasoning_labels are only defined for mode = 'latent'. But got mode '{mode}' and latent_thinking_factor = {latent_thinking_factor}, latent_thinking_bias = {latent_thinking_bias}, min_latent_thinking_steps = {min_latent_thinking_steps}, max_latent_thinking_steps = {max_latent_thinking_steps}, mask_latent_reasoning_labels = {mask_latent_reasoning_labels}. Please make sure they are None or False"
        )
    if mode == "latent":
        if latent_thinking_factor is None:
            raise ValueError("Latent thinking factor is required when mode is 'latent'")
        latent_thinking_bias = latent_thinking_bias or 0

    """Check and modify tokenizer"""
    tokenizer, demarcate_id, eos_str = _check_and_demarcate_tokenizer(tokenizer, latent=mode == "latent")

    """ Get Dataset """

    if natural_language:
        dataset = load_dataset("whyNLP/gsm8k-aug-nl", split=split)
    else:
        dataset = load_dataset("whyNLP/gsm8k-aug", split=split)

    if slice_proportion is not None:
        dataset = shuffle_and_slice(dataset, start_proportion=0, end_proportion=slice_proportion)

    """ Filtering """

    # GSM8k-Aug/Aug-NL contains a very small number of non-plain number answers. We filter these out
    # And filter out negative answers if specified
    partial_filter_gsm8k_aug_answers = partial(
        filter_gsm8k_aug_answers, remove_negative_answers=remove_negative_answers
    )
    dataset = dataset.filter(
        partial_filter_gsm8k_aug_answers,
        num_proc=num_proc,
        desc=f"Filtering rows for valid answers{' and removing negative answers' if remove_negative_answers else ''}",
    )

    """ Process dataset steps """

    if remove_last_step:

        def _remove_last_step(example):
            example["steps"] = example["steps"][:-1]
            return example

        dataset = dataset.map(_remove_last_step, num_proc=num_proc, desc="Removing last step")

    if mode == "latent":

        def create_latent_text(example):
            # If remove_last_step was True, we add an extra one in so the number of latent thinking steps is not reduced
            # This DOES assume that every question had at least one step to begin with; true for GSM8k-Aug
            num_continue_tokens = (
                len(example["steps"]) + remove_last_step
            ) * latent_thinking_factor + latent_thinking_bias
            if min_latent_thinking_steps is not None:
                num_continue_tokens = max(num_continue_tokens, min_latent_thinking_steps)
            if max_latent_thinking_steps is not None:
                num_continue_tokens = min(num_continue_tokens, max_latent_thinking_steps)
            return {"latent_text": START_TOKEN_STR + CONTINUE_TOKEN_STR * num_continue_tokens + STOP_TOKEN_STR}

        dataset = dataset.map(create_latent_text, num_proc=num_proc, desc="Creating latent text")
    elif mode == "no cot":

        def remove_steps(example):
            example["steps"] = []
            return example

        dataset = dataset.map(remove_steps, num_proc=num_proc, desc="Removing steps")
    elif mode == "cot":
        pass  # lucky day
    else:
        raise ValueError(f"Invalid mode: {mode}")

    """ Create full text """

    def _full_text_template(question: str, reasoning: str, answer: str):
        return (
            "Question: "
            + question
            + DEMARCATE_STR
            + reasoning
            + DEMARCATE_STR
            + " The answer is"
            + DEMARCATE_STR
            + ":"
            + DEMARCATE_STR
            + " "
            + answer
            + eos_str
        )

    def create_full_text(example):
        if natural_language:
            steps_text = " ".join(example["steps"])
            if steps_text:
                steps_text = " " + steps_text
        else:
            steps_text = "".join(example["steps"])

        example["full text"] = _full_text_template(
            question=example["question"], reasoning=steps_text, answer=example["answer"]
        )

        if mode == "latent":
            example["full latent text"] = _full_text_template(
                question=example["question"], reasoning=example["latent_text"], answer=example["answer"]
            )

        return example

    dataset = dataset.map(create_full_text, num_proc=num_proc, desc="Creating full text")
    if not keep_columns:
        dataset = dataset.remove_columns(["question", "steps", "answer"])
        if mode == "latent":
            dataset = dataset.remove_columns("latent_text")

    """ Tokenize """

    def tokenize_func(examples):
        tokenized = tokenizer(examples["full text"], truncation=False, padding=False, add_special_tokens=True)

        if mode == "latent":
            latent_tokenized = tokenizer(
                examples["full latent text"], truncation=False, padding=False, add_special_tokens=True
            )
            latent_tokenized = {"latent_" + k: v for k, v in latent_tokenized.items()}
            tokenized.update(latent_tokenized)

        return tokenized

    dataset = dataset.map(tokenize_func, batched=True, batch_size=batch_size, num_proc=num_proc, desc="Tokenizing")
    if not keep_columns:
        dataset = dataset.remove_columns("full text")
        if mode == "latent":
            dataset = dataset.remove_columns("full latent text")

    """ Filter for max tokens """
    if max_tokens is not None:

        def max_tokens_filter(example):
            for k, v in example.items():
                if "input_ids" in k:
                    if len(v) > max_tokens:
                        return False
            return True

        dataset = dataset.filter(
            max_tokens_filter,
            num_proc=num_proc,
            desc=f"Filtering rows for max tokenization length ({max_tokens} tokens)",
        )

    """ Add labels and maps """

    partial_remove_demarc_add_labels_and_maps = partial(
        remove_demarc_add_labels_and_maps,
        demarcate_id=demarcate_id,
        mask_latent_reasoning_labels=mask_latent_reasoning_labels,
        shift_target_token_one=shift_target_token_one,
    )

    dataset = dataset.map(
        partial_remove_demarc_add_labels_and_maps,
        num_proc=num_proc,
        desc="Removing demarcate tokens and creating labels and maps",
    )

    if mode == "latent":
        continue_id = tokenizer.convert_tokens_to_ids(CONTINUE_TOKEN_STR)

        def add_latent_continue_map(example):
            latent_input_ids = example["latent_input_ids"]
            latent_input_ids = np.array(latent_input_ids)
            latent_continue_map = latent_input_ids == continue_id
            return {"latent_continue_map": latent_continue_map.tolist()}

        dataset = dataset.map(
            add_latent_continue_map, num_proc=num_proc, desc=f"Adding latent {CONTINUE_TOKEN_STR} map"
        )

    return dataset


def tokenize_gsm8k_aug_for_eval(
    tokenizer,
    natural_language: bool = True,
    append_start_token: bool = False,
    latent_thought_count_override: int | None = None,
    answer_prefix: str | None = " The answer is:",
    split: str = "test",
    num_proc: int = 12,
    batch_size: int = 1000,
    keep_columns: bool = False,
    validate_no_eos: bool = True,
):
    """
    Tokenize GSM8k-Aug dataset for evaluation.

    Args:
        tokenizer: The tokenizer to use for encoding questions
        natural_language (bool): Whether to use natural language dataset variant
        append_start_token (bool): Whether to append START_TOKEN_STR to questions
        latent_thought_count_override (int | None): If True, questions will be tokenized with complete latent thoughts. If True, answer_prefix must also be specified.
        answer_prefix (str | None): Must be specified if latent_thought_count_override is specified. Will not be used otherwise
        split (str): Dataset split to use (default: "test")
        num_proc (int): Number of processes for parallel processing
        batch_size (int): Batch size for tokenization
        keep_columns (bool): Whether to keep original columns (does not gaurentee original columns will not be modified, only that they are not deleted)
        validate_no_eos (bool): Make sure there are no EOS tokens at the end of tokenized sequences, since that is not appropriate for generation

    Returns:
        Tokenized dataset ready for evaluation
    """

    """Check args"""

    if append_start_token and not token_in_vocab(tokenizer, START_TOKEN_STR):
        raise ValueError(f"Tokenizer must have the {START_TOKEN_STR} token when append_start_token is True.")
    if latent_thought_count_override:
        if answer_prefix is None:
            raise ValueError("answer_prefix must be specified if latent_thought_count_override is specified.")
        append_start_token = False
        ensure_tokenizer_has_latent_tokens(tokenizer)

    """ Get and process dataset """

    if natural_language:
        dataset = load_dataset("whyNLP/gsm8k-aug-nl", split=split)
    else:
        dataset = load_dataset("whyNLP/gsm8k-aug", split=split)

    ## On second thought, a validation/testing dataset should not get filtered
    # # GSM8k-Aug/Aug-NL contains a very small number of non-plain number answers. We filter these out
    # dataset = dataset.filter(filter_gsm8k_aug_answers, num_proc=num_proc, desc="Filtering rows for valid answers")

    if not keep_columns:
        dataset = dataset.remove_columns("steps")

    if append_start_token:

        def _append_start_token(example):
            example["question"] += START_TOKEN_STR
            return example

        dataset = dataset.map(_append_start_token, num_proc=num_proc, desc=f"Appending {START_TOKEN_STR} token")

    elif latent_thought_count_override:

        def _append_latent_thought(example):
            example["question"] = (
                example["question"]
                + START_TOKEN_STR
                + latent_thought_count_override * CONTINUE_TOKEN_STR
                + STOP_TOKEN_STR
                + answer_prefix
            )
            return example

        dataset = dataset.map(_append_latent_thought, num_proc=num_proc, desc="Appending complete latent thoughts")

    def tokenize_func(examples):
        return tokenizer(examples["question"], truncation=False, padding=False, add_special_tokens=True)

    dataset = dataset.map(tokenize_func, batched=True, batch_size=batch_size, num_proc=num_proc, desc="Tokenizing")

    if not keep_columns:
        dataset = dataset.remove_columns("question")

    if validate_no_eos:
        # Check to make sure there are no EOS tokens at the end of the input_ids, since that would not be appropriate for generation
        for input_ids in tqdm(dataset["input_ids"], desc="Validating no EOS tokens"):
            if input_ids and input_ids[-1] == tokenizer.eos_token_id:
                raise ValueError(
                    "Detected EOS token at end of a tokenized sequence! This is not appropriate for generation. Fix your tokenizer, or add additional processing to this function to remove the EOS token"
                )

    return dataset
