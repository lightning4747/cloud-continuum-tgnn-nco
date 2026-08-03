import torch
from src.models.action_mask import apply_action_mask, combine_masks


def test_apply_action_mask():
    logits = torch.zeros((2, 5, 10))
    mask = torch.ones((2, 5, 10), dtype=torch.bool)
    mask[0, 0, 0] = False  # Invalid action

    masked = apply_action_mask(logits, mask)
    assert masked[0, 0, 0].item() == -1e9
    assert masked[0, 0, 1].item() == 0.0


def test_combine_masks():
    m1 = torch.tensor([True, False, True])
    m2 = torch.tensor([True, True, False])
    comb = combine_masks(m1, m2)
    assert torch.equal(comb, torch.tensor([True, False, False]))
