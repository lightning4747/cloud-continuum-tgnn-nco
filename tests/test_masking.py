import torch
from src.models.action_mask import apply_action_mask, combine_masks


def test_apply_action_mask():
    logits = torch.zeros((2, 5, 10))
    mask = torch.ones((2, 5, 10), dtype=torch.bool)
    mask[0, 0, 0] = False  # Invalid action

    masked = apply_action_mask(logits, mask)
    assert masked[0, 0, 0].item() == -1e4
    assert masked[0, 0, 1].item() == 0.0


def test_apply_action_mask_all_false_row():
    logits = torch.zeros((1, 2, 5))
    mask = torch.ones((1, 2, 5), dtype=torch.bool)
    mask[0, 1, :] = False  # Row 1 is completely masked out (all False)

    masked = apply_action_mask(logits, mask)
    # Row 1 should have unmasked node 0 as fallback to prevent softmax([-inf]) = NaN
    assert masked[0, 1, 0].item() == 0.0
    assert masked[0, 1, 1].item() == -1e4


def test_combine_masks():
    m1 = torch.tensor([True, False, True])
    m2 = torch.tensor([True, True, False])
    comb = combine_masks(m1, m2)
    assert torch.equal(comb, torch.tensor([True, False, False]))
