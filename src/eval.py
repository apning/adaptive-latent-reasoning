from dataclasses import dataclass
import time
import warnings
from tqdm.auto import tqdm

import torch
from transformers import GenerationConfig, GenerationMixin
from torch.utils.data import DataLoader

from src.data_collation import LatentDataCollator
from src.data_processing import SUPPORTED_EVAL_DATA_NAMES, tokenize_gsm8k_aug_for_eval
from src.modeling_utils import START_TOKEN_STR, ensure_tokenizer_has_latent_tokens
from src.utils import JsonMixin, RunningAvg, find_number


def compare_gsm8k_aug_answer(label: str, output: str, tolerance: float = 1e-6) -> bool:
    """
    Attempts to extract a float out of output, then compares to label. If within tolerance, returns True. Else, False.

    Args:
        output (str): The output string to compare to the label
        label (str): The label string to compare to the output
        tolerance (float): The tolerance for the comparison

    Returns:
        bool: True if the output is within the tolerance of the label, False otherwise
    """

    # Normalize label and convert to float
    # We don't expect any issues for conversion to float due to data preprocessing
    label = float(find_number(label))

    extracted_num = find_number(output)

    # Unable to find any number in output
    if extracted_num is None:
        return False

    extracted_num = float(extracted_num)

    abs_diff = abs(label - extracted_num)

    if abs_diff < tolerance:
        return True
    return False


@torch.inference_mode()
def eval_gsm8k_aug(
    model: GenerationMixin,
    tokenizer,
    dataset,
    latent_thinking: bool = False,
    batch_size: int = 64,
    max_new_tokens: int = 256,
    answer_prefix: str = " The answer is:",
    thought_count_override: int | None = None,
    return_decoded: bool = False,
    verbose: bool = True,
    num_workers: int = 0,
) -> dict:
    """
    Evaluate a model on GSM8K-Aug style data and report accuracy and reasoning-length statistics. Uses greedy decoding.

    This function generates outputs for each input, extracts a numeric answer (after the case-insensitive occurrence of 'answer_prefix'), compares it to the provided label with a small tolerance, and aggregates metrics. It also estimates the number of "reasoning" tokens that precede the answer.

    Args:
        * model (transformers.GenerationMixin): Autoregressive model exposing '.generate'. Must NOT have
        gradient checkpointing enabled. The function temporarily switches the model to 'eval()' and
        restores the previous training state on exit.
        * tokenizer: Tokenizer with 'eos_token_id'. If 'pad_token_id' is None, 'eos_token_id' is used as
        padding.
        * dataset: Iterable of dicts with keys 'input_ids', 'attention_mask', and 'answer'
        * latent_thinking (bool): If True, verifies the tokenizer has latent tokens and requires that each
        input ends with '<START>' when 'thought_count_override' is None. Reasoning length will include
        the '<START>' token (added as +1 to the measured reasoning length).
        * batch_size (int): Per-device batch size used by the DataLoader.
        * max_new_tokens (int): Maximum tokens to generate for each sample.
        * answer_prefix (str): Case-insensitive marker used to split the generation into reasoning and
        answer segments. If not found, the output is counted as bad-format.
        * thought_count_override (int | None): If provided, assumes the dataset already contains the full
        set of thoughts and the model directly generates the answer; 'answer_prefix' is ignored and the
        entire generation is treated as the answer. The reported reasoning length is set to this value
        as-is.
        * return_decoded (bool): If True, also returns decoded inputs/outputs, per-example reasoning lengths,
        and correctness flags.
        * verbose (bool): Controls tqdm progress and the padding-token warning.
        * num_workers (int): Number of workers for the DataLoader.

    Returns
        dict with fields:
        * score: Overall accuracy (n_correct / n_total).
        * correct_p: Alias of score.
        * n_correct, n_total
        * bad_format_p, n_bad_format
        * n_length_cutoffs: Count of generations that likely hit 'max_new_tokens' without emitting PAD/EOS
        in the last position.
        * n_avg_reasoning_tokens: Mean length of the reasoning segment across examples that were correctly
        formatted; may be None if none were parsable.
        * n_reasoning_tokens_std, min_reasoning_tokens, max_reasoning_tokens
        * time_taken: Wall-clock seconds for the evaluation.
        If 'return_decoded=True', also:
            * decoded: List[(decoded_input, decoded_output_with_special_tokens)]
            * reasoning_lengths: List[int]
            * correct_list: List[bool]

    """

    start_time = time.perf_counter()

    """Check args"""

    if latent_thinking:
        ensure_tokenizer_has_latent_tokens(tokenizer)

        if not thought_count_override and dataset[0]["input_ids"][-1] != tokenizer.convert_tokens_to_ids(
            START_TOKEN_STR
        ):
            raise ValueError(
                "If latent_thinking is True, the dataset must have a START token at the end of each of the input_ids."
            )

    if tokenizer.pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
        if verbose:
            warnings.warn(
                f"Tokenizer did not have a pad token id. The eos token will be used as the padding token ({tokenizer.eos_token_id})."
            )
    else:
        pad_token_id = tokenizer.pad_token_id

    max_input_length = max(len(_input) for _input in dataset["input_ids"])
    if max_input_length + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError(
            f"Max input length {max_input_length} + max new tokens {max_new_tokens} > model.config.max_position_embeddings {model.config.max_position_embeddings}"
        )

    if model.is_gradient_checkpointing:
        # We don't disable here because we do not know the gradient checkpointing kwargs with which to re-enable
        raise ValueError("model cannot be gradient checkpointing! Please ensure it is disabled before evaluation")

    """ Setup for evaluation """

    # Save the model's training state and switch to eval mode
    model_was_training = model.training
    model.eval()

    data_collator = LatentDataCollator(pad_token_id=pad_token_id, padding_side="left")
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator, num_workers=num_workers)

    gen_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    n_total = len(dataset)
    n_bad_format = 0
    n_correct = 0
    n_length_cutoffs = 0

    n_avg_reasoning_tokens = RunningAvg()

    device = model.get_input_embeddings().weight.device

    if return_decoded:
        decoded = []
        reasoning_lengths = []
        correct_list = []

    for batch in tqdm(dataloader, desc="Evaluating model on GSM8k-Aug", disable=not verbose):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        answers = batch["answer"]

        outputs = model.generate(input_ids, attention_mask=attention_mask, generation_config=gen_config)

        # slice away the original inputs
        outputs = outputs[:, input_ids.shape[-1] :]

        # Indicates some elements of output very likely got cut off
        if outputs.shape[-1] == max_new_tokens:
            # An element is considered cut off if its last token was neither padding nor EOS
            outputs_n_length_cutoffs = sum(
                (outputs[:, -1] != pad_token_id) & (outputs[:, -1] != tokenizer.eos_token_id)
            )
            n_length_cutoffs += outputs_n_length_cutoffs.item()

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_outputs_w_special_toks = tokenizer.batch_decode(outputs)

        if return_decoded:
            decoded_inputs = tokenizer.batch_decode(input_ids)
            for _decoded_input, _decoded_output in zip(decoded_inputs, decoded_outputs_w_special_toks):
                decoded.append((_decoded_input, _decoded_output))

        for output, output_w_special_toks, answer in zip(decoded_outputs, decoded_outputs_w_special_toks, answers):
            if not isinstance(output, str):
                raise ValueError(f"output must be a string, got {type(output)}")
            if not isinstance(output_w_special_toks, str):
                raise ValueError(f"output_w_special_toks must be a string, got {type(output)}")
            if not isinstance(answer, str):
                raise ValueError(f"answer must be a string, got {type(answer)}")

            """ Evaluate correctness """

            if thought_count_override is None:
                split_output = output.lower().split(answer_prefix.lower())

                # This indicates answer_prefix was not found in output
                if len(split_output) == 1:
                    n_bad_format += 1
                    continue

                output_answer = split_output[-1]
            else:
                output_answer = output.lower()

            is_correct = compare_gsm8k_aug_answer(answer, output_answer)
            n_correct += is_correct

            """ Measuring reasoning length """

            if thought_count_override is None:
                # Obtain starting index of answer_prefix (also the ending index of reasoning) using .lower() on both output and answer_prefix
                # We do this because some tokenizers could have normalized the input to the model by lowercasing

                split_output_w_special_toks = output_w_special_toks.lower().split(answer_prefix.lower())

                if len(split_output_w_special_toks) == 1:
                    warnings.warn(
                        f"split_output_w_special_toks had length 1, which indicates answer_prefix was not found in output_w_special_toks. This is extraordinarily odd, as for the code to get here we already know that answer_prefix was found in output. Suggest investigating this issue.\n\toutput: '{output.lower()}'\n\tsplit_output_w_special_toks: '{split_output_w_special_toks}'\n\tanswer_prefix: '{answer_prefix.lower()}'"
                    )
                    continue

                # Sum the lengths of all the substrings before the last substring (the one that contains the answer). This is the total charater count of the reasoning substring
                reasoning_end_idx = sum(len(split) for split in split_output_w_special_toks[:-1])
                reasoning_str = output_w_special_toks[:reasoning_end_idx]

                reasoning_str_input_ids = tokenizer(reasoning_str, add_special_tokens=False)["input_ids"]

                reasoning_length = len(reasoning_str_input_ids)
                if latent_thinking:
                    if not reasoning_length:
                        raise ValueError(
                            "latent_thinking was True but somehow there was no reasoning detected (we have not accounted for the <START> token yet, which we will manually account for layer as it is part of the inputs and thus not in the outputs). There should be at least one due to the <|STOP|> token at the end of latent reasoning"
                        )
                    # To account for the <START> token that was part of the input
                    reasoning_length += 1
            else:
                reasoning_length = thought_count_override

            n_avg_reasoning_tokens.add(reasoning_length)
            if return_decoded:
                reasoning_lengths.append(reasoning_length)
                correct_list.append(is_correct)

    # Restore model's original training state
    model.train(model_was_training)

    return_dict = {
        "score": n_correct / n_total,  # A standardized name that must be the same across datasets. HIGHER is better
        "correct_p": n_correct / n_total,
        "n_correct": n_correct,
        "n_total": n_total,
        "bad_format_p": n_bad_format / n_total,
        "n_bad_format": n_bad_format,
        "n_length_cutoffs": n_length_cutoffs,
        "n_avg_reasoning_tokens": n_avg_reasoning_tokens.avg,
        "n_reasoning_tokens_std": n_avg_reasoning_tokens.std,
        "min_reasoning_tokens": n_avg_reasoning_tokens.min,
        "max_reasoning_tokens": n_avg_reasoning_tokens.max,
        "time_taken": time.perf_counter() - start_time,
    }

    if return_decoded:
        return_dict["decoded"] = decoded
        return_dict["reasoning_lengths"] = reasoning_lengths
        return_dict["correct_list"] = correct_list

    return return_dict


@dataclass
class EvalConfig(JsonMixin):
    # Data details
    data_name: str
    split: str
    latent_thinking: bool = False
    latent_thought_count_override: int | None = None

    # Generation details
    max_new_tokens: int = 512
    batch_size: int | None = None

    ## Computational
    # Number of processes used to process data
    num_proc: int = 12
    # Number of workers used to load workers
    num_workers: int = 0

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate args"""
        if self.data_name not in SUPPORTED_EVAL_DATA_NAMES:
            raise ValueError(f"Invalid data name: {self.data_name}. Please choose from: {SUPPORTED_EVAL_DATA_NAMES}")

        if self.latent_thought_count_override is not None and not self.latent_thinking:
            raise ValueError(
                f"latent_thought_count_override must be None if latent_thinking is False. But got latent_thought_count_override = {self.latent_thought_count_override}"
            )


def get_tokenized_eval_dataset(tokenizer, eval_config: EvalConfig):
    if eval_config.data_name not in SUPPORTED_EVAL_DATA_NAMES:
        raise ValueError(
            f"The data_name {eval_config.data_name} is not valid! Please pick one from {SUPPORTED_EVAL_DATA_NAMES}"
        )

    if "gsm8k-aug" in eval_config.data_name:
        if eval_config.data_name == "gsm8k-aug":
            natural_language = False
        elif eval_config.data_name == "gsm8k-aug-nl":
            natural_language = True

        return tokenize_gsm8k_aug_for_eval(
            tokenizer,
            natural_language=natural_language,
            append_start_token=eval_config.latent_thinking,
            latent_thought_count_override=eval_config.latent_thought_count_override,
            split=eval_config.split,
            num_proc=eval_config.num_proc,
        )


def eval_from_config(
    model,
    tokenizer,
    eval_config: EvalConfig,
    dataset=None,
    batch_size: int | None = None,
    verbose: bool = True,
    return_decoded: bool = False,
):
    if dataset is None:
        dataset = get_tokenized_eval_dataset(tokenizer, eval_config)
    if batch_size is None:
        batch_size = eval_config.batch_size

    if "gsm8k-aug" in eval_config.data_name:
        if batch_size is None:
            batch_size = 64

        thought_count_override = None
        if eval_config.latent_thought_count_override is not None:
            # To account for <START> and <|STOP|>
            thought_count_override = eval_config.latent_thought_count_override + 2

        return eval_gsm8k_aug(
            model,
            tokenizer,
            dataset,
            latent_thinking=eval_config.latent_thinking,
            batch_size=batch_size,
            max_new_tokens=eval_config.max_new_tokens,
            thought_count_override=thought_count_override,
            return_decoded=return_decoded,
            verbose=verbose,
            num_workers=eval_config.num_workers,
        )
