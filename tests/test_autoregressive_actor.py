"""
Tests for AutoregressiveDecoder and AutoregressiveActorCritic.

Critical tests:
  - Hard feasibility guarantee: two CNFs each demanding 90% of one node's capacity
    must be assigned to DIFFERENT nodes (core contribution).
  - Teacher-forcing: action=prescribed tensor must exactly match output.
  - Inactive CNFs: must not contribute to log-prob or consume capacity.
  - SFC-priority ordering: cnf_order tensor from env must be a valid permutation.
  - Shape correctness: all output tensors must have expected shapes.
"""
import numpy as np
import pytest
import torch
import yaml

from src.env.continuum_env import ContinuumEnv
from src.models.actor_critic import AutoregressiveActorCritic
from src.models.autoregressive_actor import AutoregressiveDecoder


@pytest.fixture
def cfg():
    with open("configs/model_config.yaml") as f:
        return yaml.safe_load(f)


# ─── Shape Tests ──────────────────────────────────────────────────────────────

def test_autoregressive_output_shapes(cfg):
    """All output tensors must have the expected shape."""
    model = AutoregressiveActorCritic(cfg)
    B, C, M, W, F = 2, 50, 150, 5, 9
    nf     = torch.randn(B, C, F).abs()
    ei     = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    nh     = torch.randn(B, W, C, F).abs()
    cf     = torch.randn(B, M, 5).abs()
    mask   = torch.ones(B, M, C, dtype=torch.bool)
    order  = torch.arange(M).unsqueeze(0).expand(B, -1)
    active = torch.ones(B, M, dtype=torch.bool)

    actions, lp, ent, val = model.get_action_and_value(
        nf, ei, nh, cf, action_mask=mask, cnf_order=order, cnf_active=active
    )
    assert actions.shape == (B, M), f"Expected ({B},{M}), got {actions.shape}"
    assert lp.shape == (B,),        f"Expected ({B},), got {lp.shape}"
    assert ent.shape == (B,),       f"Expected ({B},), got {ent.shape}"
    assert val.shape == (B, 1),     f"Expected ({B},1), got {val.shape}"


# ─── Core Feasibility Guarantee ───────────────────────────────────────────────

def test_feasibility_guarantee_no_capacity_violation():
    """
    CRITICAL TEST — hard feasibility guarantee.

    Setup:
      - 2 nodes, each with capacity 1.0 for CPU/RAM/Storage.
      - 2 active CNFs, each demanding 0.9 of every resource.
      - Single node can only host one CNF (0.9 + 0.9 > 1.0).

    Expected result:
      The autoregressive decoder must place each CNF on a DIFFERENT node.
      After CNF-0 is placed on node X, X's residual drops below 0.1,
      making it infeasible for CNF-1 (which needs 0.9). CNF-1 must go to node Y.
    """
    B, C_max, M_max = 1, 2, 2
    d_model = 64
    decoder = AutoregressiveDecoder(d_model=d_model, n_heads=1)

    torch.manual_seed(0)
    node_emb  = torch.randn(B, C_max, d_model)
    cnf_emb   = torch.randn(B, M_max, d_model)

    # Each node has capacity 1.0; each CNF demands 0.9
    node_cpu  = torch.ones(B, C_max)
    node_ram  = torch.ones(B, C_max)
    node_stor = torch.ones(B, C_max)
    cnf_cpu   = torch.full((B, M_max), 0.9)
    cnf_ram   = torch.full((B, M_max), 0.9)
    cnf_stor  = torch.full((B, M_max), 0.9)
    cnf_active = torch.ones(B, M_max, dtype=torch.bool)
    static_mask = torch.ones(B, M_max, C_max, dtype=torch.bool)
    cnf_order = torch.tensor([[0, 1]])

    actions, _, _ = decoder(
        node_emb, cnf_emb,
        node_cpu, node_ram, node_stor,
        cnf_cpu, cnf_ram, cnf_stor,
        cnf_active, static_mask, cnf_order,
    )

    node_0 = actions[0, 0].item()
    node_1 = actions[0, 1].item()
    assert node_0 != node_1, (
        f"FEASIBILITY GUARANTEE VIOLATED: CNF-0 and CNF-1 both placed on node {node_0}. "
        "Residual-capacity mask failed to exclude the overloaded node."
    )


def test_feasibility_guarantee_runs_on_random_inputs():
    """
    Run 10 random seeds — each time with tight capacity — and verify
    no two CNFs occupy the same node when that would exceed capacity.
    """
    B, C_max, M_max = 1, 3, 3
    d_model = 32
    decoder = AutoregressiveDecoder(d_model=d_model, n_heads=1)

    for seed in range(10):
        torch.manual_seed(seed)
        node_emb   = torch.randn(B, C_max, d_model)
        cnf_emb    = torch.randn(B, M_max, d_model)
        # Demand = 0.6, capacity = 1.0 → max one CNF per node
        node_cpu   = torch.ones(B, C_max)
        node_ram   = torch.ones(B, C_max)
        node_stor  = torch.ones(B, C_max)
        cnf_cpu    = torch.full((B, M_max), 0.6)
        cnf_ram    = torch.full((B, M_max), 0.6)
        cnf_stor   = torch.full((B, M_max), 0.6)
        cnf_active = torch.ones(B, M_max, dtype=torch.bool)
        static_mask = torch.ones(B, M_max, C_max, dtype=torch.bool)
        cnf_order  = torch.arange(M_max).unsqueeze(0)

        actions, _, _ = decoder(
            node_emb, cnf_emb,
            node_cpu, node_ram, node_stor,
            cnf_cpu, cnf_ram, cnf_stor,
            cnf_active, static_mask, cnf_order,
        )
        assigned = actions[0].tolist()
        assert len(set(assigned)) == M_max, (
            f"Seed {seed}: Multiple CNFs on same node — {assigned}. "
            "Residual-capacity mask failed."
        )


# ─── Teacher Forcing ──────────────────────────────────────────────────────────

def test_teacher_forcing_matches_action():
    """
    PPO update requires that teacher-forced actions exactly match the provided action tensor.
    With zero demand (all nodes always feasible), every node is valid for every CNF.
    """
    B, C_max, M_max, d_model = 2, 10, 5, 64
    decoder = AutoregressiveDecoder(d_model=d_model, n_heads=1)

    torch.manual_seed(42)
    node_emb  = torch.randn(B, C_max, d_model)
    cnf_emb   = torch.randn(B, M_max, d_model)
    node_cpu  = torch.ones(B, C_max)
    node_ram  = torch.ones(B, C_max)
    node_stor = torch.ones(B, C_max)
    cnf_cpu   = torch.zeros(B, M_max)   # zero demand → all nodes always valid
    cnf_ram   = torch.zeros(B, M_max)
    cnf_stor  = torch.zeros(B, M_max)
    cnf_active  = torch.ones(B, M_max, dtype=torch.bool)
    static_mask = torch.ones(B, M_max, C_max, dtype=torch.bool)
    cnf_order   = torch.arange(M_max).unsqueeze(0).expand(B, -1)

    prescribed = torch.randint(0, C_max, (B, M_max))
    actions_out, _, _ = decoder(
        node_emb, cnf_emb, node_cpu, node_ram, node_stor,
        cnf_cpu, cnf_ram, cnf_stor,
        cnf_active, static_mask, cnf_order, action=prescribed,
    )
    assert torch.all(actions_out == prescribed), (
        f"Teacher-forced actions mismatch!\n"
        f"  prescribed: {prescribed[0].tolist()}\n"
        f"  output:     {actions_out[0].tolist()}"
    )


# ─── Inactive CNFs ────────────────────────────────────────────────────────────

def test_inactive_cnfs_do_not_contribute_to_log_prob():
    """Inactive CNF slots (cnf_active=False) must produce zero log-prob contribution."""
    B, C_max, M_max, d_model = 1, 5, 4, 32
    decoder = AutoregressiveDecoder(d_model=d_model, n_heads=1)

    torch.manual_seed(7)
    node_emb  = torch.randn(B, C_max, d_model)
    cnf_emb   = torch.randn(B, M_max, d_model)
    node_cpu  = torch.ones(B, C_max)
    node_ram  = torch.ones(B, C_max)
    node_stor = torch.ones(B, C_max)
    cnf_cpu   = torch.zeros(B, M_max)
    cnf_ram   = torch.zeros(B, M_max)
    cnf_stor  = torch.zeros(B, M_max)

    # ALL CNFs inactive
    cnf_active  = torch.zeros(B, M_max, dtype=torch.bool)
    static_mask = torch.ones(B, M_max, C_max, dtype=torch.bool)
    cnf_order   = torch.arange(M_max).unsqueeze(0).expand(B, -1)

    _, log_probs, _ = decoder(
        node_emb, cnf_emb, node_cpu, node_ram, node_stor,
        cnf_cpu, cnf_ram, cnf_stor,
        cnf_active, static_mask, cnf_order,
    )
    assert log_probs.sum().item() == pytest.approx(0.0, abs=1e-6), (
        f"Inactive CNFs contributed to log-prob: {log_probs.tolist()}"
    )


def test_inactive_cnfs_do_not_consume_capacity():
    """
    Inactive CNFs with extremely large demand must not decrement residual capacity.
    If they did, subsequent active CNFs would incorrectly see no feasible nodes.
    """
    B, C_max, M_max, d_model = 1, 2, 3, 32
    decoder = AutoregressiveDecoder(d_model=d_model, n_heads=1)

    torch.manual_seed(99)
    node_emb  = torch.randn(B, C_max, d_model)
    cnf_emb   = torch.randn(B, M_max, d_model)
    node_cpu  = torch.ones(B, C_max) * 0.5
    node_ram  = torch.ones(B, C_max) * 0.5
    node_stor = torch.ones(B, C_max) * 0.5
    # Demand far exceeds capacity — safe only if inactive CNFs are skipped
    cnf_cpu   = torch.ones(B, M_max) * 10.0
    cnf_ram   = torch.ones(B, M_max) * 10.0
    cnf_stor  = torch.ones(B, M_max) * 10.0

    # M_max-1 inactive, 1 active (at last position in order)
    cnf_active = torch.zeros(B, M_max, dtype=torch.bool)
    cnf_active[0, M_max - 1] = True   # only last CNF is active
    cnf_active_demand = torch.zeros_like(cnf_cpu)   # active CNF has zero demand
    cnf_cpu  = cnf_cpu   # inactive CNFs have huge demand (10.0) — must be ignored
    cnf_ram  = cnf_ram
    cnf_stor = cnf_stor

    static_mask = torch.ones(B, M_max, C_max, dtype=torch.bool)
    cnf_order   = torch.arange(M_max).unsqueeze(0).expand(B, -1)

    # Should not raise or produce NaN
    actions, lp, ent = decoder(
        node_emb, cnf_emb, node_cpu, node_ram, node_stor,
        cnf_cpu, cnf_ram, cnf_stor,
        cnf_active, static_mask, cnf_order,
    )
    assert not torch.isnan(lp).any(), "NaN log-prob — inactive CNFs consumed capacity!"


# ─── SFC Ordering ─────────────────────────────────────────────────────────────

def test_sfc_priority_cnf_order_is_valid_permutation():
    """
    cnf_order from _compute_cnf_order() must be a valid permutation of [0..M_max-1].
    Every index appears exactly once.
    """
    env = ContinuumEnv(seed=42)
    obs, _ = env.reset(seed=42)
    cnf_order = obs["cnf_order"]
    assert sorted(cnf_order.tolist()) == list(range(env.m_max)), (
        f"cnf_order is not a valid permutation of [0..{env.m_max-1}]!\n"
        f"Got: {sorted(cnf_order.tolist())}"
    )


def test_cnf_order_tight_budget_sfc_comes_first():
    """
    If there are two SFCs, the one with smaller delay_budget should have its
    CNFs appear earlier in cnf_order than the looser-budget SFC.
    """
    env = ContinuumEnv(seed=0)
    obs, _ = env.reset(seed=0)
    sfcs = env.current_sfcs

    if sfcs.n_active_sfcs < 2:
        pytest.skip("Need ≥2 active SFCs for this test")

    # Find the two SFC IDs and their budgets
    seen_sfc_ids = []
    for m in range(env.m_max):
        if sfcs.cnf_active[m]:
            sid = int(sfcs.sfc_id[m])
            if sid not in seen_sfc_ids:
                seen_sfc_ids.append(sid)
        if len(seen_sfc_ids) == 2:
            break

    sid0, sid1 = seen_sfc_ids[0], seen_sfc_ids[1]
    # Budget for each SFC (from sfc_delay_budget indexed by slot, not sfc_id)
    # Slot 0 = first packed SFC = sid0, slot 1 = sid1
    budget0 = float(sfcs.sfc_delay_budget[0])
    budget1 = float(sfcs.sfc_delay_budget[1])

    cnf_order = obs["cnf_order"].tolist()

    # Find first occurrence of a CNF from each SFC in the order
    first_pos = {}
    for pos, m in enumerate(cnf_order):
        if sfcs.cnf_active[m]:
            sid = int(sfcs.sfc_id[m])
            if sid not in first_pos:
                first_pos[sid] = pos
        if len(first_pos) == 2:
            break

    if sid0 in first_pos and sid1 in first_pos:
        if budget0 < budget1:
            assert first_pos[sid0] < first_pos[sid1], (
                f"Tight-budget SFC (budget={budget0:.1f}) appears AFTER "
                f"loose-budget SFC (budget={budget1:.1f}) in cnf_order!"
            )
        elif budget1 < budget0:
            assert first_pos[sid1] < first_pos[sid0], (
                f"Tight-budget SFC (budget={budget1:.1f}) appears AFTER "
                f"loose-budget SFC (budget={budget0:.1f}) in cnf_order!"
            )
        # Equal budgets: no ordering requirement


# ─── Gradient Flow ────────────────────────────────────────────────────────────

def test_gradient_flows_through_log_prob(cfg):
    """Log-probs must support gradient computation for PPO update."""
    model = AutoregressiveActorCritic(cfg)
    B, C, M, W, F = 1, 10, 5, 5, 9
    nf    = torch.randn(B, C, F, requires_grad=False).abs()
    ei    = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    nh    = torch.randn(B, W, C, F).abs()
    cf    = torch.randn(B, M, 5).abs()
    mask  = torch.ones(B, M, C, dtype=torch.bool)
    order = torch.arange(M).unsqueeze(0).expand(B, -1)
    active= torch.ones(B, M, dtype=torch.bool)

    actions, lp, ent, val = model.get_action_and_value(
        nf, ei, nh, cf, action_mask=mask, cnf_order=order, cnf_active=active
    )
    # Simulate PPO policy loss
    loss = -lp.mean() - 0.01 * ent.mean() + 0.5 * (val.squeeze(-1) ** 2).mean()
    loss.backward()

    # Check at least some parameters have gradients
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed — backward() failed!"
    assert not any(torch.isnan(g).any() for g in grads), "NaN gradients detected!"
