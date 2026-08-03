import torch


def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Sets logits of invalid placement actions to -1e9.
    logits: (B, M_max, C_max) or (M_max, C_max)
    mask: (B, M_max, C_max) or (M_max, C_max) bool/binary
    """
    mask_bool = mask.bool()
    masked_logits = logits.masked_fill(~mask_bool, -1e9)
    return masked_logits


def combine_masks(*masks: torch.Tensor) -> torch.Tensor:
    """
    Logical AND across all provided constraint mask tensors.
    """
    result = masks[0].bool()
    for m in masks[1:]:
        result = result & m.bool()
    return result
