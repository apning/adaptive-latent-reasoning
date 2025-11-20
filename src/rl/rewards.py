from dataclasses import dataclass
from typing import Any, Callable
import warnings

from accelerate import PartialState
from accelerate.utils import gather
from src.utils import closeto
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.eval import compare_gsm8k_aug_answer
from src.modeling_utils import START_TOKEN_STR, CONTINUE_TOKEN_STR, STOP_TOKEN_STR


LATENT_PREFIX = " The answer is:"


def latent_prefix_format_verifier(completion_w_special_toks: str, **kwargs) -> bool:
    """
    Verifies that the completion follows the expected latent prefix format.
    That is
    <|CONTINUE|><|CONTINUE|><|CONTINUE|>...<|CONTINUE|><|STOP|> The answer is: <answer here>
    Optionally there could be a <START> before the <|CONTINUE|> tokens

    Only the completion_w_special_toks input argument is used. Extra keyword arguments are accepted
    for API compatibility and ignored.

    Args:
        completion_w_special_toks (str): The generated completion string with special tokens
        **kwargs: Ignored extra keyword-only parameters

    Returns:
        bool: True if the completion is properly formatted, False otherwise
    """
    # Type checking
    if not isinstance(completion_w_special_toks, str):
        raise TypeError(f"Expected completion_w_special_toks to be str, got {type(completion_w_special_toks)}")

    prefix_split_completion = completion_w_special_toks.split(LATENT_PREFIX)

    # If the length is 1 then no prefix exists
    # If the length > 2 then multiple prefixes exist
    if len(prefix_split_completion) != 2:
        return False

    reasoning_text = prefix_split_completion[0].strip()

    # Remove all reasoning tokens from reasoning text. So that we can see if there is anything left
    for str_to_remove in (START_TOKEN_STR, CONTINUE_TOKEN_STR, STOP_TOKEN_STR):
        reasoning_text = reasoning_text.replace(str_to_remove, "")

    # If anything is left over, then there were additional non-latent tokens in the thought
    if reasoning_text.strip():
        return False

    return True


def gsm8k_aug_answer_verifier(completion: str, label: str, bad_format_false: bool = False, **kwargs) -> bool:
    """

    Determines whether a completion has the correct answer. For GSM8k-Aug dataset completions.
    Assumes answers are preceeded by LATENT_PREFIX.
    Uses only the completion and label input arguments. Extra keyword arguments are accepted for
    API compatibility and ignored.


    Args:
        completion (str): The generated completion string
        label (str): The label string
        bad_format_false (bool): If True, returns False when answer prefix is not detected instead of an error.
        **kwargs: Ignored extra keyword-only parameters

    Returns:
        bool: True if the completion has the correct answer, False otherwise
    """

    # Type checking
    if not isinstance(completion, str):
        raise TypeError(f"Expected completion to be str, got {type(completion)}")
    if not isinstance(label, str):
        raise TypeError(f"Expected label to be str, got {type(label)}")

    prefix_split_completion = completion.split(LATENT_PREFIX)

    if len(prefix_split_completion) == 1:
        if bad_format_false:
            return False
        raise ValueError(
            f"Completion {completion} does not contain the expected latent prefix {LATENT_PREFIX}. It should not possible for the code here, because all outputs sent to the answer verifier should have been approved by the format verifier! Something is wrong"
        )

    output = prefix_split_completion[-1]

    return compare_gsm8k_aug_answer(label=label, output=output)


def latent_length_judge(prompt: str, completion_w_special_toks: str, **kwargs) -> int:
    """
    Simply counts the number of occurrences of <START>, <|CONTINUE|>, and <|STOP|> tokens in prompt and completion combined.
    Assumes inputs have already had formatting verified earlier.
    Uses only the prompt and completion_w_special_toks input arguments. Extra keyword arguments are accepted
    for API compatibility and ignored.

    Args:
        prompt (str): The input prompt string
        completion_w_special_toks (str): The generated completion string with special tokens
        **kwargs: Ignored extra keyword-only parameters

    Returns:
        int: Latent token count
    """
    # Type checking
    if not isinstance(prompt, str):
        raise TypeError(f"Expected prompt to be str, got {type(prompt)}")
    if not isinstance(completion_w_special_toks, str):
        raise TypeError(f"Expected completion_w_special_toks to be str, got {type(completion_w_special_toks)}")

    text = prompt + completion_w_special_toks

    count = 0

    for token_str in (START_TOKEN_STR, CONTINUE_TOKEN_STR, STOP_TOKEN_STR):
        count += text.count(token_str)

    return count


@dataclass
class RelativeLengthRewarder:
    tokenizer: PreTrainedTokenizerBase
    num_generations: int

    format_verifier: Callable[..., bool]
    answer_verifier: Callable[..., bool]
    length_judge: Callable[..., int | float]

    format_penalty: float = -1.0
    answer_reward: float = 1.0
    relative_length_penalty: float = 1.0
    relative_length_accuracy_requirement: float | int | None = (
        None  # 1 or 1.0 will be interpreted as 100% instead of 1 generation, since more than 1 correct generation is required to assign RELATIVE penalties
    )

    relative_length_reward: None | float | str = None

    all_gather: bool = False

    def __post_init__(self):
        if self.format_penalty > 0:
            raise ValueError(f"format_penalty must be <= 0. Got: {self.format_penalty}")

        if (
            self.relative_length_accuracy_requirement is not None
            and self.relative_length_accuracy_requirement > self.num_generations
        ):
            raise ValueError(
                f"relative_length_accuracy_requirement must be <= num_generations. Got: {self.relative_length_accuracy_requirement} > {self.num_generations}"
            )

        if self.relative_length_reward and not self.relative_length_accuracy_requirement:
            raise ValueError(
                f"relative_length_reward requires ({self.relative_length_reward}) relative_length_accuracy_requirement to be set (currently set to {self.relative_length_accuracy_requirement})"
            )

        if (
            isinstance(self.relative_length_accuracy_requirement, float)
            and self.relative_length_accuracy_requirement != 1
            and closeto(self.relative_length_accuracy_requirement, 1)
        ):
            self.relative_length_accuracy_requirement = 1
            warnings.warn(
                f"relative_length_accuracy_requirement set to 1.0 since it was close to 1. Got: {self.relative_length_accuracy_requirement}. This will be interpreted as 100% instead of 1 generation, since more than 1 correct generation is required to assign RELATIVE penalties"
            )

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        completion_ids: list[list[int]],
        prompt_id: list[int],
        label: list[Any],
        trainer_state: Any = None,
    ) -> list[float]:
        device = PartialState().device

        """ Checking/Proccesing Args """

        if not (len(prompts) == len(completions) == len(completion_ids) == len(prompt_id) == len(label)):
            raise ValueError("prompts, completions, completion_ids, prompt_id, and label must all be the same length")

        prompt_ids = prompt_id
        labels = label

        n_local_items = len(prompts)

        completions_w_special_toks = self.tokenizer.batch_decode(completion_ids)

        """ Get 1d maps for formatting, correctness, and length """

        # List of bools representing whether each item is properly formatted
        format_map = [
            self.format_verifier(
                prompt=prompt,
                completion=completion,
                completion_w_special_toks=completion_w_special_toks,
                completion_ids=completion_id,
            )
            for prompt, completion, completion_w_special_toks, completion_id in zip(
                prompts, completions, completions_w_special_toks, completion_ids
            )
        ]

        # list of bools representing whether each item is correct. Those that were not formatted correctly are automatically wrong
        correct_map = [
            self.answer_verifier(
                prompt=prompt,
                completion=completion,
                completion_w_special_toks=completion_w_special_toks,
                completion_ids=completion_id,
                label=label,
            )
            if is_formatted
            else False
            for prompt, completion, completion_w_special_toks, completion_id, label, is_formatted in zip(
                prompts, completions, completions_w_special_toks, completion_ids, labels, format_map
            )
        ]

        # list of numbers representing the length of each item. Only correct items are considered.
        length_map = [
            self.length_judge(
                prompt=prompt,
                completion=completion,
                completion_w_special_toks=completion_w_special_toks,
                completion_ids=completion_id,
            )
            if is_correct
            else torch.nan
            for prompt, completion, completion_w_special_toks, completion_id, is_correct in zip(
                prompts, completions, completions_w_special_toks, completion_ids, correct_map
            )
        ]

        """ Calculate relative length for each prompt """

        # Turn length_map and prompt_ids into 1d tensors so can do slicing and indexing later
        # Local means it is only for the items on this device
        local_length_map = torch.tensor(length_map, dtype=torch.float32, device=device)
        local_prompt_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)

        # Gather lengths and prompts from all devices
        # Global since they are across all devices
        # Probably not necessary, since are you really going to have num_generations so large they are > per_device_train_batch_size * gradient_accumulation_steps?
        if self.all_gather:
            global_length_map, global_prompt_ids = gather((local_length_map.clone(), local_prompt_ids.clone()))
        else:
            global_length_map, global_prompt_ids = local_length_map.clone(), local_prompt_ids.clone()

        local_rel_length_map = local_length_map.clone()

        unique_prompt_ids = set(prompt_ids)  # local

        for prompt_id in unique_prompt_ids:
            global_prompt_mask = global_prompt_ids == prompt_id
            global_prompt_occurrences = global_prompt_mask.sum().item()

            # Why not == instead of <? Some dataloader samplers might repeat samples
            if global_prompt_occurrences < self.num_generations:
                raise ValueError(
                    f"Prompt {prompt_id} has only {global_prompt_occurrences} generations, but {self.num_generations} are required"
                )
            elif global_prompt_occurrences % self.num_generations:
                raise ValueError(
                    f"Prompt {prompt_id} has {global_prompt_occurrences} generations, which is not a multiple of num_generations={self.num_generations}"
                )

            local_prompt_mask = local_prompt_ids == prompt_id

            # Mask to get all local rows corresponding to the prompt and which are not NaN
            local_valid_prompt_mask = local_prompt_mask & ~local_length_map.isnan()

            if not local_valid_prompt_mask.any():
                continue

            global_valid_prompt_lens = global_length_map[global_prompt_mask & ~global_length_map.isnan()]

            rel_len_coeff = (
                -self.relative_length_penalty
            )  # Negative so that longer lengths get a penalty and shorter get a reward
            n_global_valid = len(global_valid_prompt_lens)

            ## If number of global correct completions does not meet required, use relative length reward if specified, or just early exit otherwise
            if self.relative_length_accuracy_requirement:
                if self.relative_length_accuracy_requirement <= 1:
                    n_global_valid_requirement = self.relative_length_accuracy_requirement * global_prompt_occurrences
                else:
                    n_global_valid_requirement = self.relative_length_accuracy_requirement * (
                        global_prompt_occurrences / self.num_generations
                    )

                if n_global_valid < n_global_valid_requirement:
                    if not self.relative_length_reward:
                        local_rel_length_map[local_valid_prompt_mask] = 0
                        continue
                    elif self.relative_length_reward == "same":
                        rel_len_coeff = self.relative_length_penalty
                    else:
                        rel_len_coeff = self.relative_length_reward

            assert global_valid_prompt_lens.numel(), (
                f"No valid lengths found for prompt {prompt_id}. Something is wrong. The code should not have gotten here!"
            )

            ## Calculate max and min length for this prompt globally
            max_len = global_valid_prompt_lens.max().item()
            min_len = global_valid_prompt_lens.min().item()
            len_range = max_len - min_len
            mean_len = global_valid_prompt_lens.mean().item()

            # Normalize the lens so that they have mean 0 and range of 1
            # then mulitply by rel_len_coeff
            # If all lengths are the same, they all get relative length 0
            if len_range:
                local_rel_length_map[local_valid_prompt_mask] = (
                    rel_len_coeff * (local_rel_length_map[local_valid_prompt_mask] - mean_len) / len_range
                )
            else:
                local_rel_length_map[local_valid_prompt_mask] = 0

        # Give the NaN values a relative length of 0
        local_rel_length_map[local_rel_length_map.isnan()] = 0

        """ Calculate rewards """

        format_map = torch.tensor(format_map, dtype=bool, device=device)
        correct_map = torch.tensor(correct_map, dtype=bool, device=device)

        rewards = torch.zeros(n_local_items, dtype=torch.float32, device=device)

        rewards[~format_map] += self.format_penalty
        rewards[correct_map] += self.answer_reward
        rewards += local_rel_length_map

        rewards = rewards.tolist()

        return rewards

    @property
    def __name__(self):
        return self.__class__.__name__


@dataclass
class SimpleFormatRewarder:
    """
    A simple format rewarder. Most useful as a reward func for logging purposes (can do so by setting its reward weight to 0).
    """

    tokenizer: PreTrainedTokenizerBase
    format_verifier: Callable[..., bool]

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        completion_ids: list[list[int]],
        prompt_id: list[int],
        label: list[Any],
        trainer_state: Any = None,
    ) -> list[bool]:
        """Checking/Proccesing Args"""

        if not (len(prompts) == len(completions) == len(completion_ids) == len(prompt_id) == len(label)):
            raise ValueError("prompts, completions, completion_ids, prompt_id, and label must all be the same length")

        completions_w_special_toks = self.tokenizer.batch_decode(completion_ids)

        """ Get format map """

        format_map = [
            self.format_verifier(
                prompt=prompt,
                completion=completion,
                completion_w_special_toks=completion_w_special_toks,
                completion_ids=completion_id,
            )
            for prompt, completion, completion_w_special_toks, completion_id in zip(
                prompts, completions, completions_w_special_toks, completion_ids
            )
        ]

        return format_map

    @property
    def __name__(self):
        return self.__class__.__name__


@dataclass
class SimpleLatentLengthRewarder:
    """
    Returns a reward which is simply the number of latent thinking tokens. Not per correctly formatted completion or correct answer. But for all completions.

    Most useful as a reward func for logging purposes (can do so by setting its reward weight to 0). If this is used as an actual reward function with nonzero weight, will INCENTIVIZE longer responses.

    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        completion_ids: list[list[int]],
        prompt_id: list[int],
        label: list[Any],
        trainer_state: Any = None,
    ) -> list[int]:
        """Checking/Proccesing Args"""

        if not (len(prompts) == len(completions) == len(completion_ids) == len(prompt_id) == len(label)):
            raise ValueError("prompts, completions, completion_ids, prompt_id, and label must all be the same length")

        completions_w_special_toks = self.tokenizer.batch_decode(completion_ids)

        """ Get lengths """

        length_map = [
            latent_length_judge(prompt=prompt, completion_w_special_toks=completion_w_special_toks)
            for prompt, completion_w_special_toks in zip(prompts, completions_w_special_toks)
        ]

        return length_map

    @property
    def __name__(self):
        return self.__class__.__name__


@dataclass
class SimpleAccuracyRewarder:
    """
    A simple accuracy rewarder. Most useful as a reward func for logging purposes (can do so by setting its reward weight to 0).
    Makes some assumptions. Chiefly:
        * That the answer verifier does NOT need completion_w_special_toks to work
        * That the answer verifier has a bad_format_false argument that can be used in lieu of running a format verifier beforehand (that would weed out bad formats before ever getting to the answer verifier)
    """

    answer_verifier: Callable[..., bool]

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        completion_ids: list[list[int]],
        prompt_id: list[int],
        label: list[Any],
        trainer_state: Any = None,
    ) -> list[bool]:
        """Checking/Proccesing Args"""

        if not (len(prompts) == len(completions) == len(completion_ids) == len(prompt_id) == len(label)):
            raise ValueError("prompts, completions, completion_ids, prompt_id, and label must all be the same length")

        labels = label

        """ Get correct map """

        correct_map = [
            self.answer_verifier(
                prompt=prompt, completion=completion, completion_ids=completion_id, label=label, bad_format_false=True
            )
            for prompt, completion, completion_id, label in zip(prompts, completions, completion_ids, labels)
        ]

        return correct_map

    @property
    def __name__(self):
        return self.__class__.__name__


@dataclass
class _CorrectProportionRewarder(SimpleAccuracyRewarder):
    num_generations: int

    def __call__(self, **kwargs):
        ## Currently does not implement gather across devices for prompts, so relies on the number of generations per device being evenly disible by the number of generations per prompt

        device = PartialState().device

        ## Get correct map via call to super
        correct_map = super().__call__(**kwargs)

        ## Prep variables and convert to tensor
        prompt_ids = kwargs["prompt_id"]
        unique_prompt_ids = set(prompt_ids)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        correct_map = torch.tensor(correct_map, dtype=bool, device=device)

        correct_proportion_map = {}

        # For each prompt, get proportion of correct generations
        for prompt_id in unique_prompt_ids:
            prompt_mask = prompt_ids == prompt_id
            prompt_occurrences = prompt_mask.sum().item()

            # Why not == instead of <? Some dataloader samplers might repeat samples
            if prompt_occurrences < self.num_generations:
                raise ValueError(
                    f"Prompt {prompt_id} has only {prompt_occurrences} generations, but {self.num_generations} are required"
                )
            elif prompt_occurrences % self.num_generations:
                raise ValueError(
                    f"Prompt {prompt_id} has {prompt_occurrences} generations, which is not a multiple of num_generations={self.num_generations}"
                )

            n_correct = correct_map[prompt_mask].sum().item()

            correct_proportion_map[prompt_id] = n_correct / prompt_occurrences

        return correct_proportion_map


class AllGensCorrectProportion(_CorrectProportionRewarder):
    def __call__(self, **kwargs):
        n_items = len(kwargs["prompts"])

        correct_proportion_map = super().__call__(**kwargs)

        all_correct_proportion = sum(1 for prop in correct_proportion_map.values() if closeto(prop, 1)) / len(
            correct_proportion_map
        )

        return [all_correct_proportion] * n_items


class AllGensWrongProportion(_CorrectProportionRewarder):
    def __call__(self, **kwargs):
        n_items = len(kwargs["prompts"])

        correct_proportion_map = super().__call__(**kwargs)

        all_wrong_proportion = sum(1 for prop in correct_proportion_map.values() if closeto(prop, 0)) / len(
            correct_proportion_map
        )

        return [all_wrong_proportion] * n_items


class SomeGensCorrectProportion(_CorrectProportionRewarder):
    def __call__(self, **kwargs):
        n_items = len(kwargs["prompts"])

        correct_proportion_map = super().__call__(**kwargs)

        some_correct_proportion = sum(
            1 for prop in correct_proportion_map.values() if not closeto(prop, 1) and not closeto(prop, 0)
        ) / len(correct_proportion_map)

        return [some_correct_proportion] * n_items
