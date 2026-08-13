import torch


def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Sets logits of invalid placement actions to a safe large negative scalar.
    Supports FP16, BF16, and FP32 without overflow or NaN.
    logits: (B, M_max, C_max) or (M_max, C_max)
    mask: (B, M_max, C_max) or (M_max, C_max) bool/binary
    """
    mask_bool = mask.bool()

    # Guard against all-False rows which cause softmax([-inf, ...]) = NaN
    all_false = (mask_bool.sum(dim=-1, keepdim=True) == 0)
    if all_false.any():
        mask_bool = mask_bool.clone()
        mask_bool[all_false.expand_as(mask_bool)] = False
        mask_bool[..., 0] = mask_bool[..., 0] | all_false.squeeze(-1)

    fill_value = -1e4
    masked_logits = logits.masked_fill(~mask_bool, fill_value)
    return torch.nan_to_num(masked_logits, nan=0.0, posinf=1e4, neginf=-1e4)


def combine_masks(*masks: torch.Tensor) -> torch.Tensor:
    """
    Logical AND across all provided constraint mask tensors.
    """
    result = masks[0].bool()
    for m in masks[1:]:
        result = result & m.bool()
    return result
