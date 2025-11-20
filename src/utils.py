import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
import pickle
import re
from pathlib import Path
import random
import math
from collections.abc import Iterator
import time
import warnings
from typing import Any, TypeVar

from huggingface_hub import HfApi, HfFileSystem

import torch
from transformers.trainer_pt_utils import get_parameter_names

try:
    import pynvml

    HAS_PNYVML = True
except ImportError:
    HAS_PNYVML = False

STR_TO_TORCH_DTYPE = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

FloatingTensor = TypeVar("FloatingTensor", torch.FloatTensor, torch.HalfTensor, torch.BFloat16Tensor)


def get_project_root() -> Path:
    """
    Determines the project root using a fixed relative path from this file.
    It assumes this file is located within the 'project_root/src' directory.
    Validates the assumed root by checking for an 'src' directory within it.

    Returns:
        Path: The absolute path to the project root directory.

    Raises:
        FileNotFoundError: If the 'src' directory is not found at the
                           assumed project root, indicating a potential
                           misconfiguration or change in directory structure.
        RuntimeError: If this utility file's location has changed such that
                      the fixed relative path logic is no longer valid.
    """
    try:
        # Get the absolute path of this file
        this_file_path = Path(__file__).resolve()

        # Assumed structure: project_root/src/<this file>.py
        assumed_project_root = this_file_path.parent.parent
    except IndexError:
        # This would happen if Path(__file__).parent goes above filesystem root
        raise RuntimeError(
            f"The utility file '{__file__}' seems to be located too high in the "
            f"directory tree for the fixed relative path logic to apply. "
            f"Expected 'project_root/src/<this file>.py'. "
            f"Instead got {this_file_path}."
        )

    # Validate: Check for the presence of an 'src' directory in the assumed root.
    # This 'src' directory is the one directly under the project_root.
    expected_src_dir = assumed_project_root / "src"

    if not (expected_src_dir.exists() and expected_src_dir.is_dir()):
        raise FileNotFoundError(
            f"Validation failed: An 'src' directory was not found at the assumed "
            f"project root '{assumed_project_root}'.\n"
            f"This function expects the project structure to be 'project_root/src/...', "
            f"and this utility file ('{__file__}') to be at a certain fixed location "
            f"within 'src/'. If the structure or file location has changed, "
            f"this function may need an update."
        )

    return assumed_project_root


def select_best_device(mode):
    """
    Select the best available GPU device based on specified criteria.

    Args:
        mode (str): Selection mode - 'm' for most free memory, 'u' for least utilization. 'u' requires pynvml package to be installed.

    Returns:
        torch.device: Selected device (GPU or CPU if no GPU available).

    Raises:
        Exception: If mode is not 'm' or 'u'.
    """

    if mode not in ["m", "u"]:
        raise ValueError(
            f'select_best_device: Acceptable inputs for mode are "m" (most free memory) and "u" (least utilization). You specified: {mode}'
        )

    if not torch.cuda.is_available():
        return torch.device("cpu")

    indices = list(range(torch.cuda.device_count()))
    random.shuffle(
        indices
    )  # shuffle the indices we iterate through so that, if, say, a bunch of processes scramble for GPUs at once, the first one won't get them all

    if mode == "m":
        max_free_memory = 0
        device_index = 0
        for i in indices:
            free_memory = torch.cuda.mem_get_info(i)[0]
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                device_index = i
        return torch.device(f"cuda:{device_index}")

    elif mode == "u":
        if HAS_PNYVML:
            pynvml.nvmlInit()
            min_util = 100
            device_index = 0
            for i in indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # Get the handle for the target GPU
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = util.gpu  # GPU utilization percentage (integer)
                if gpu_utilization < min_util:
                    min_util = gpu_utilization
                    device_index = i
            pynvml.nvmlShutdown()

            # If all the GPUs are basically at max util, then make choice via memory availiability
            if min_util > 95:
                return select_best_device(mode="m")

            return torch.device(f"cuda:{device_index}")
        else:
            warnings.warn(
                "Utilization 'u' based selection is only available if pnyvml is available, but it is not. Please install pnyvml to use mode 'u'. Switching to mode 'm' (memory-based device selection)"
            )
            return select_best_device("m")


class RunningAvg:
    def __init__(self):
        self.n = 0
        self.avg = None
        self._m2 = 0.0  # Sum of squared deviations from the running mean (for variance)
        self.min = None
        self.max = None

    def add(self, val: int | list[int] | tuple[int] | float | list[float] | tuple[float] | Iterator):
        if not isinstance(val, (int, float, list, tuple, Iterator)):
            raise ValueError(f"val must be a int, float, list, tuple, or Iterator. Got {type(val)}")

        # Normalize input to an iterable of numeric values
        if isinstance(val, Iterator):
            seq = list(val)
        elif isinstance(val, (list, tuple)):
            seq = val
        else:  # int or float
            seq = [val]

        # No-op for empty iterables
        if isinstance(seq, (list, tuple)) and len(seq) == 0:
            return

        if self.avg is None:
            self.avg = 0.0
            self._m2 = 0.0

        # Welford's online algorithm for mean and variance
        for x in seq:
            if not isinstance(x, (int, float)):
                raise ValueError(f"All values must be numeric (int or float). Got {type(x)}")
            self.n += 1
            delta = x - self.avg
            self.avg += delta / self.n
            delta2 = x - self.avg
            self._m2 += delta * delta2

            # Update min and max
            if self.min is None or x < self.min:
                self.min = x
            if self.max is None or x > self.max:
                self.max = x

    @property
    def var(self) -> float | None:
        """Population variance (dividing by n). Returns None if no values have been added."""
        if self.n == 0:
            return None
        return self._m2 / self.n

    @property
    def std(self) -> float | None:
        """Population standard deviation. Returns None if no values have been added."""
        v = self.var
        return math.sqrt(v) if v is not None else None

    @property
    def sample_var(self) -> float | None:
        """Sample variance (unbiased; dividing by n-1). Returns None if fewer than 2 values."""
        if self.n < 2:
            return None
        return self._m2 / (self.n - 1)

    @property
    def sample_std(self) -> float | None:
        """Sample standard deviation (unbiased). Returns None if fewer than 2 values."""
        v = self.sample_var
        return math.sqrt(v) if v is not None else None


def find_number(text, strip_commas: bool = True) -> str:
    """
    Finds the first number in text. Number can contain decimal point or negative symbol.

    Args:
        text (str): The string to process
        strip_commas (bool): If True, strips commas from text first. If False, numbers separated by commas will be considerd separate numbers, and only the first will be returned. Ie. "109,234" -> "109"

    Returns:
        str | None: The extracted number. None if none could be found

    """

    if strip_commas:
        text = text.replace(",", "")

    match = re.search(r"-?(?:\d+(?:\.\d*)?|\.\d+)", text)
    return match.group() if match else None


def str_formatted_datetime():
    """
    Get current datetime as a formatted string.

    Returns:
        str: Datetime string in format 'YYYYMMDD-HHMMSS'.
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def get_checkpoint_dir(
    name: str, datetime: str | None = None, base_dir: os.PathLike | None = None, checkpoints_name: str = "checkpoints"
) -> Path:
    """
    Get the checkpoint directory path for a given name and datetime.

    Args:
        name (str): The name/path of the checkpoint, can contain forward slashes
                   to indicate subdirectories (e.g., "model/experiment1")
        datetime (str | None): The datetime string to append to the path. If None then one will be obtained using current date and time
        base_dir (os.PathLike | None): The base directory to save the checkpoint to. If None, then the project root will be used.
        checkpoints_name (str): The name of the checkpoints directory.

    Returns:
        Path: The full path to the checkpoint directory, constructed as
              project_root/<name_parts>/<datetime>
    """

    if datetime is None:
        datetime = str_formatted_datetime()

    path_parts = name.split("/")

    # Remove empty strings or strings of just whitespace
    path_parts = [part.strip() for part in path_parts if part.strip()]

    if base_dir is None:
        base_dir = get_project_root()

    return Path(base_dir, checkpoints_name, *path_parts, datetime)


def pickle_data(data, save_path):
    """
    Pickle data to a file.

    Args:
        data: Any Python object to be pickled.
        save_path (str): Path where to save the pickled data. Should end with .pkl.
    """
    with open(save_path, "wb") as file:
        pickle.dump(data, file)


def unpickle_data(save_path):
    """
    Load pickled data from a file.

    Args:
        save_path (str): Path to the pickled file.

    Returns:
        The unpickled Python object.
    """
    with open(save_path, "rb") as file:
        data = pickle.load(file)
    return data


def get_obj_vars_str(obj) -> str:
    text = ""
    for k, v in vars(obj).items():
        text += f"{k}\t=\t{v}\n"
    return text


class timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


def is_power_of_2(val: int):
    """
    Check if a value is a power of 2.

    Args:
        val (int): The value to check.

    Returns:
        bool: True if val is a power of 2, False otherwise.
    """
    return val > 0 and (val & (val - 1)) == 0


@dataclass
class PrintLog:
    """Prints and logs to text file"""

    txt_log_path: str | None = None
    rank: int | None = None

    def __call__(self, message: str, warning: bool = False, raise_exception: None | Exception = None):
        rank_str = f"[Device {self.rank}] " if self.rank is not None else ""
        message = rank_str + message

        if not warning:
            print(message)
        else:
            warnings.warn(message)

        # if condition only True when self.rank is None or 0
        if not self.rank and self.txt_log_path is not None:
            with open(self.txt_log_path, "a") as f:
                f.write(message)
                f.write("\n")

        if raise_exception is not None:
            raise raise_exception(message)


def sum_dicts(dicts: list[dict]) -> dict:
    keys = dicts[0].keys()

    # Check that all dicts have the same keys
    for d in dicts[1:]:
        if d.keys() != keys:
            raise ValueError(f"All dictionaries must have the same keys. Expected {keys}, but got {d.keys()}")

    return {k: sum(d[k] for d in dicts) for k in keys}


def mean_dicts(dicts: list[dict]) -> dict:
    keys = dicts[0].keys()

    # Check that all dicts have the same keys
    for d in dicts[1:]:
        if d.keys() != keys:
            raise ValueError(f"All dictionaries must have the same keys. Expected {keys}, but got {d.keys()}")

    return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}


def save_as_json(data, save_path: os.PathLike, override_if_exists: bool = False) -> None:
    """Save arbitrary basic Python structures (primitives, lists, dicts, tuples) as JSON.

    Note: Tuples will be converted to lists when saved, as JSON doesn't have a tuple type.
    Sets will also be converted to lists. Other non-JSON-serializable types will raise an error.

    Args:
        data: The data to save (primitives, lists, dicts, tuples of primitives)
        save_path (os.PathLike): Path where to save the JSON file
        override_if_exists (bool): If False, raises error if file already exists. If True, overwrites.
    """
    save_path = Path(save_path)

    if save_path.exists() and not override_if_exists:
        raise ValueError(f"File {save_path} already exists and override_if_exists is False")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)


def load_from_json(load_path: os.PathLike) -> Any:
    """Load data from JSON file.

    Returns the loaded data structure. Note that tuples saved as JSON will be loaded as lists, since JSON doesn't distinguish between lists and tuples.
    """
    with open(load_path, "r") as f:
        return json.load(f)


class JsonMixin:
    """Mixin class to add JSON save/load functionality to any dataclass"""

    def save_as_json(self, filepath: os.PathLike, override_if_exists: bool = False):
        """Save to JSON file

        Args:
            filepath (os.PathLike): Path where to save the JSON file
            override_if_exists (bool): If False, raises error if file already exists. If True, overwrites.
        """
        save_as_json(asdict(self), filepath, override_if_exists)

    @classmethod
    def load_from_json(cls, filepath: os.PathLike) -> "JsonMixin":
        """Load from JSON file"""
        data = load_from_json(filepath)
        return cls(**data)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str):
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls(**data)


def str2bool(v: str) -> bool:
    """
    Accept common bool spellings and return Python bools.
    Returns None unchanged so the default can stay None.
    """
    if v is None or isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "t", "yes", "y", "1"):
        return True
    if v in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def str2float_w_none(v: str) -> float | None:
    if v is None or v is False or v.lower() in ("", "none", "false"):
        return None
    return float(v)


def str2int_w_none(v: str) -> int | None:
    if v is None or v == "":
        return None
    return int(v)


def str_w_none(v: str) -> str | None:
    if v is None or v.strip() == "":
        return None
    return str(v)


def find_instances(class_types, args: tuple, kwargs: dict[str, Any]) -> list:
    """
    Class can be a single class or a tuple of classes

    Args:
        args (tuple)
        kwargs (dict[str, Any])
    """
    instances = []

    for arg in args:
        if isinstance(arg, class_types):
            instances.append(arg)
    for v in kwargs.values():
        if isinstance(v, class_types):
            instances.append(v)
    return instances


def get_decay_parameter_names(model) -> list[str]:
    """
    A modified version of transformers.Trainer.get_decay_parameter_names that additionally excludes some parameters.
    """
    forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
    additional_forbidden_name_patterns = [r"start_embed", r"end_embed", r"continue_embed", r"stop_embed"]
    forbidden_name_patterns += additional_forbidden_name_patterns
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm, torch.nn.RMSNorm], forbidden_name_patterns)
    return decay_parameters


def get_decay_grouped_params(model, decay_val: float) -> list[dict[str, list]]:
    decay_parameters = get_decay_parameter_names(model)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            "weight_decay": decay_val,
        },
        {
            "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def split_hf_path(path: str) -> tuple[str, str]:
    hf_fs = HfFileSystem()
    resolved_path = hf_fs.resolve_path(path)
    repo_id = resolved_path.repo_id
    path_in_repo = resolved_path.path_in_repo
    return repo_id, path_in_repo


def hf_download_and_unjson(path: str) -> Any:
    hf_api = HfApi()
    repo_id, path_in_repo = split_hf_path(path)
    file_path = hf_api.hf_hub_download(repo_id=repo_id, filename=path_in_repo)
    with open(file_path, "r") as f:
        return json.load(f)


def closeto(val1: float, val2: float, eps: float = 1e-8):
    """Check if two floating point values are close within epsilon tolerance."""
    return val1 - eps <= val2 <= val1 + eps
