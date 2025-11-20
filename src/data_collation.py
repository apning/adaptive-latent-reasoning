import torch


def pad_lists_to_pt(inputs: list[list], pad_value, padding_side: str, pt_dtype=torch.long) -> torch.Tensor:
    """
    Pad a list of lists to equal length and convert to PyTorch tensor.

    Args:
        inputs (list[list]): List of lists to pad
        pad_value: Value to use for padding
        padding_side (str): "left" or "right" padding
        pt_dtype (torch.long): PyTorch tensor dtype

    Returns:
        torch.Tensor: Padded tensor
    """

    """Verify args"""
    if padding_side not in ["left", "right"]:
        raise ValueError(f"padding_side must be either 'left' or 'right', got {padding_side}")

    if isinstance(inputs, list) and not inputs:
        return torch.empty(0, dtype=pt_dtype)

    if not (isinstance(inputs, list) and isinstance(inputs[0], list)):
        raise ValueError(f"inputs must be a list of lists, got {inputs}")

    """ Pad inputs """

    max_len = max(len(input) for input in inputs)

    padded_inputs = []
    for _input in inputs:
        n_pad = max_len - len(_input)
        if padding_side == "left":
            padded_input = [pad_value] * n_pad + _input
        else:
            padded_input = _input + [pad_value] * n_pad
        padded_inputs.append(padded_input)

    padded_inputs = torch.tensor(padded_inputs, dtype=pt_dtype)
    return padded_inputs


class LatentDataCollator:
    # Any key with any of these name parts in it should have the same shape as all the others
    EXPECT_SAME_SHAPE_NAME_PARTS = {"input_ids", "attention_mask", "labels", "_map"}

    def __init__(self, pad_token_id: int, padding_side: str):
        if padding_side not in ["left", "right"]:
            raise ValueError(f"padding_side must be either 'left' or 'right', got {padding_side}")
        if pad_token_id is None:
            raise ValueError("pad_token_id must not be None")

        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, features: list[dict]) -> dict:
        keys = features[0].keys()

        # Turn list of dicts into dict of lists
        features = {k: [feature[k] for feature in features] for k in keys}

        # Update the components if they exist
        for k in keys:
            if "input_ids" in k:
                features[k] = pad_lists_to_pt(features[k], self.pad_token_id, padding_side=self.padding_side)
            elif "attention_mask" in k:
                features[k] = pad_lists_to_pt(features[k], 0, padding_side=self.padding_side)
            elif "labels" in k:
                features[k] = pad_lists_to_pt(features[k], -100, padding_side=self.padding_side)
            elif "_map" in k:
                features[k] = pad_lists_to_pt(features[k], False, padding_side=self.padding_side, pt_dtype=bool)

        # Check that all components that are expected to be of same shape indeed are
        expect_same_shape_components_not_latent = {
            k: v
            for k, v in features.items()
            if "latent" not in k and any(part in k for part in self.EXPECT_SAME_SHAPE_NAME_PARTS)
        }
        expect_same_shape_components_latent = {
            k: v
            for k, v in features.items()
            if "latent" in k and any(part in k for part in self.EXPECT_SAME_SHAPE_NAME_PARTS)
        }
        for expect_same_shape_components in (
            expect_same_shape_components_not_latent,
            expect_same_shape_components_latent,
        ):
            if expect_same_shape_components:
                first_item_k, first_item = next(iter(expect_same_shape_components.items()))
                first_item_shape = first_item.shape

                for k, v in expect_same_shape_components.items():
                    if v.shape != first_item_shape:
                        raise ValueError(
                            f"Component '{k}' with shape {v.shape} did not match shape of component '{first_item_k}' which had shape {first_item_shape}"
                        )

        return features
