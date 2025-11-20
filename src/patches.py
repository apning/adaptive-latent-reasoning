## The purpose of these patches was to enable gradient checkpointing on latent thinking models when combined with LatentThinkingCheckpointingCache. Although it worked, it did not seem to provide the expected benefit to memory usage. So it has been removed.

# from functools import partial
# import types
# from typing import Callable
# from torch import nn
# from transformers.modeling_utils import PreTrainedModel
# from transformers.modeling_layers import GradientCheckpointingLayer, logger

# from src.cache import LatentThinkingCheckpointingCache


# def patched_gradient_checkpointing_layer__call__(self, *args, **kwargs):
#     # Modify to give an exception to the no-cache-if-gradient-checkpointing rule

#     past_kv = None
#     if "past_key_value" in kwargs:
#         past_kv = kwargs["past_key_value"]
#     elif "past_key_values" in kwargs:
#         past_kv = kwargs["past_key_values"]

#     if not isinstance(past_kv, LatentThinkingCheckpointingCache) and self.gradient_checkpointing and self.training:
#         do_warn = False
#         layer_name = self.__class__.__name__
#         message = f"Caching is incompatible with gradient checkpointing in {layer_name} when past_key_value(s) is not an instance of LatentThinkingCheckpointingCache (it was {type(past_kv)}). Setting"

#         if "use_cache" in kwargs and kwargs["use_cache"]:
#             kwargs["use_cache"] = False
#             message += " `use_cache=False`,"
#             do_warn = True

#         # different names for the same thing in different layers
#         if "past_key_value" in kwargs and kwargs["past_key_value"] is not None:
#             kwargs["past_key_value"] = None
#             message += " `past_key_value=None`,"
#             do_warn = True

#         if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
#             kwargs["past_key_values"] = None
#             message += " `past_key_values=None`,"
#             do_warn = True

#         if "layer_past" in kwargs and kwargs["layer_past"] is not None:
#             kwargs["layer_past"] = None
#             message += " `layer_past=None`,"
#             do_warn = True

#         # warn if anything was changed
#         if do_warn:
#             message = message.rstrip(",") + "."
#             logger.warning(message)

#         return self._gradient_checkpointing_func(
#             partial(super(GradientCheckpointingLayer, self).__call__, **kwargs), *args
#         )
#     return super(GradientCheckpointingLayer, self).__call__(*args, **kwargs)


# ## We must globally patch GradientCheckpointingLayer.__call__ or else it just won't just us use cache during gradient checkpointing
# GradientCheckpointingLayer.__call__ = patched_gradient_checkpointing_layer__call__


# def get_patch_functions(
#     model_class: type[PreTrainedModel],
#     block_class: type[nn.Module],
#     attn_class: type[nn.Module],
#     model_forward_patch: Callable,
#     block_forward_patch: Callable,
#     attn_forward_patch: Callable,
#     additional_patches: dict | list[dict] | None = None,
# ):
#     if isinstance(additional_patches, dict):
#         additional_patches = [additional_patches]

#     def patch_model(model: model_class, strict: bool = True):
#         if not isinstance(model, model_class):
#             raise ValueError(f"model must be an instance of {model_class}. Instead got {type(model)}")

#         for m in model.modules():
#             if isinstance(m, model_class):
#                 if strict and "forward" in vars(m):
#                     raise ValueError(f"Received {model_class} that already has a .forward instance attribute!")
#                 m.forward = types.MethodType(model_forward_patch, m)
#             elif isinstance(m, block_class):
#                 if strict and "forward" in vars(m):
#                     raise ValueError(f"Received {block_class} that already has a .forward instance attribute!")
#                 m.forward = types.MethodType(block_forward_patch, m)
#             elif isinstance(m, attn_class):
#                 if strict and "forward" in vars(m):
#                     raise ValueError(f"Received {attn_class} that already has a .forward instance attribute!")
#                 m.forward = types.MethodType(attn_forward_patch, m)
#             if additional_patches:
#                 for additional_patch in additional_patches:
#                     _class = additional_patch["class"]
#                     if isinstance(m, _class):
#                         to_patch = additional_patch["to_patch"]
#                         patch = additional_patch["patch"]
#                         if strict and to_patch in vars(m):
#                             raise ValueError(f"Received {_class} that already has a .{to_patch} instance attribute!")
#                         setattr(m, to_patch, types.MethodType(patch, m))

#     def unpatch_model(model: model_class, strict: bool = True):
#         if not isinstance(model, model_class):
#             raise ValueError(f"model must be an instance of {model_class}. Instead got {type(model)}")

#         for m in model.modules():
#             if isinstance(m, model_class):
#                 if "forward" in vars(m):
#                     del m.forward
#                 elif strict:
#                     raise ValueError(f"Received {model_class} that did not have a .forward instance attribute!")
#             elif isinstance(m, block_class):
#                 if "forward" in vars(m):
#                     del m.forward
#                 elif strict:
#                     raise ValueError(f"Received {block_class} that did not have a .forward instance attribute!")
#             elif isinstance(m, attn_class):
#                 if "forward" in vars(m):
#                     del m.forward
#                 elif strict:
#                     raise ValueError(f"Received {attn_class} that did not have a .forward instance attribute!")
#             if additional_patches:
#                 for additional_patch in additional_patches:
#                     _class = additional_patch["class"]
#                     if isinstance(m, _class):
#                         to_patch = additional_patch["to_patch"]
#                         if to_patch in vars(m):
#                             delattr(m, to_patch)
#                         elif strict:
#                             raise ValueError(f"Received {_class} that did not have a .{to_patch} instance attribute!")

#     return patch_model, unpatch_model
