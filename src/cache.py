from typing import Any, Dict, Optional, Tuple
import torch
from transformers.cache_utils import Cache


class LatentThinkingCheckpointingCache(Cache):
    def __init__(
        self,
        key_cache: dict | None = None,
        value_cache: dict | None = None,
        disable_copy_add: bool = False,
        disable_update: bool = False,
    ):
        super().__init__()

        # key_cache and value_cache are both dicts where the key (of dict variety) is the layer_idx and the value (of dict variety) is a list of all the KV (of transformer variety) tensors that have been added to the cache
        self.key_cache = key_cache if key_cache is not None else {}
        self.value_cache = value_cache if value_cache is not None else {}
        self.disable_copy_add = disable_copy_add
        self.disable_update = disable_update

    @property
    def _seen_tokens(self) -> int:
        return self.get_seq_length()

    def get_new_kv(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the values of key_states and value_States after they have been concatenated with the rest of the cache.

        Mimics the input/output of .update(...), but doesn't actually update the cache.
        """

        if layer_idx in self.key_cache:
            key_states = torch.cat(self.key_cache[layer_idx] + [key_states], dim=-2)
            value_states = torch.cat(self.value_cache[layer_idx] + [value_states], dim=-2)

        return key_states, value_states

    def copy_add(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "LatentThinkingCheckpointingCache":
        """
        Adds the key_states and value_states into a copy of the current cache.

        Returns the copy.
        """

        if self.disable_copy_add:
            raise ValueError("copy_add has been disabled!")

        # Shallow copy the dicts and the lists in them
        new_key_cache = {k: v.copy() for k, v in self.key_cache.items()}
        new_value_cache = {k: v.copy() for k, v in self.value_cache.items()}

        if layer_idx not in new_key_cache:
            new_key_cache[layer_idx] = []
            new_value_cache[layer_idx] = []

        new_key_cache[layer_idx].append(key_states)
        new_value_cache[layer_idx].append(value_states)

        new_cache = LatentThinkingCheckpointingCache(
            key_cache=new_key_cache,
            value_cache=new_value_cache,
            disable_copy_add=self.disable_copy_add,
            disable_update=self.disable_update,
        )

        return new_cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A regular update method. Not really the point of this cache... you may as well use DynamicCache if you find yourself using this method.

        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """

        if self.disable_update:
            raise ValueError("update has been disabled!")

        key_states, value_states = self.get_new_kv(key_states, value_states, layer_idx, cache_kwargs)

        self.key_cache[layer_idx] = [key_states]
        self.value_cache[layer_idx] = [value_states]

        return key_states, value_states

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed.

        When updating HF model source code to work with this class, one must be careful to ensure that once the call to a past_key_values.update(...) has been replaced with a .get_new_kv(...), that get_seq_length(...) is never called on that same cache object in the remaining code. This is because replacing the .update(...) with a .get_new_kv(...) will mean that the sequence length of that cache object will not increase as is otherwise expected.
        """
        if layer_idx not in self.key_cache:
            return 0
        return sum(k.shape[-2] for k in self.key_cache[layer_idx])

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. LatentThinkingStaticCache does not have a maximum length."""
        return None

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns (i.e. sliding_window, chunk_size),
        for each layer.
        """
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, 0
