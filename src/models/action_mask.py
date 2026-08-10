import torch


def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Sets logits of invalid placement actions to a large negative scalar.
    Supports FP16 (Half precision) without overflow.
    logits: (B, M_max, C_max) or (M_max, C_max)
    mask: (B, M_max, C_max) or (M_max, C_max) bool/binary
    """
    mask_bool = mask.bool()
    fill_value = -1e4 if logits.dtype == torch.float16 else -1e9
    masked_logits = logits.masked_fill(~mask_bool, fill_value)
    return masked_logits


def combine_masks(*masks: torch.Tensor) -> torch.Tensor:
    """
    Logical AND across all provided constraint mask tensors.
    """
    result = masks[0].bool()
    for m in masks[1:]:
        result = result & m.bool()
    return result
