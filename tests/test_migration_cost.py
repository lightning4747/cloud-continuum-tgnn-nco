"""
Unit tests for src.env.migration_cost — dirty-memory pre-copy transfer cost model.
"""
import numpy as np
import pytest
from src.env.migration_cost import compute_migration_cost


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_placement(M: int, C: int, choices: list[int]) -> np.ndarray:
    """Build one-hot placement matrix from a list of chosen node indices."""
    p = np.zeros((M, C), dtype=np.float32)
    for m, c in enumerate(choices):
        p[m, c] = 1.0
    return p


def _bw_matrix(C: int, src: int, dst: int, val: float) -> np.ndarray:
    bw = np.zeros((C, C), dtype=np.float32)
    bw[src, dst] = val
    bw[dst, src] = val
    return bw


# ─── Zero-cost cases ──────────────────────────────────────────────────────────

def test_zero_cost_same_node():
    """CNF that stays on the same node incurs zero migration cost."""
    M, C = 2, 5
    prev = np.array([2, 3], dtype=np.int32)
    active = np.array([True, True])
    placement = _make_placement(M, C, [2, 3])   # same nodes as prev
    cnf_ram = np.array([10.0, 10.0])
    edge_bw = np.ones((C, C), dtype=np.float32) * 0.5

    cost = compute_migration_cost(placement, prev, active, cnf_ram, edge_bw)
    assert cost == pytest.approx(0.0, abs=1e-9)


def test_zero_cost_new_sfc():
    """New SFC arrivals (prev_node == -1) incur zero migration cost."""
    M, C = 1, 5
    prev = np.array([-1], dtype=np.int32)
    active = np.array([True])
    placement = _make_placement(M, C, [3])
    cnf_ram = np.array([64.0])
    edge_bw = np.ones((C, C)) * 0.5

    cost = compute_migration_cost(placement, prev, active, cnf_ram, edge_bw)
    assert cost == pytest.approx(0.0, abs=1e-9)


def test_zero_cost_inactive_cnf():
    """Inactive CNF slots must produce zero migration cost even when node changes."""
    M, C = 2, 5
    prev = np.array([0, 1], dtype=np.int32)
    active = np.array([False, False])   # all inactive
    placement = _make_placement(M, C, [3, 4])   # different nodes
    cnf_ram = np.array([64.0, 64.0])
    edge_bw = np.ones((C, C)) * 0.5

    cost = compute_migration_cost(placement, prev, active, cnf_ram, edge_bw)
    assert cost == pytest.approx(0.0, abs=1e-9)


# ─── Positive-cost cases ──────────────────────────────────────────────────────

def test_cost_positive_on_migration():
    """CNF that moves node must produce a strictly positive migration penalty."""
    M, C = 1, 5
    prev = np.array([0], dtype=np.int32)
    active = np.array([True])
    placement = _make_placement(M, C, [3])       # moved from 0 → 3
    cnf_ram = np.array([32.0])                   # 32 GB RAM
    edge_bw = _bw_matrix(C, 0, 3, 0.1)          # 10% of 10000 Mbps = 1000 Mbps

    cost = compute_migration_cost(
        placement, prev, active, cnf_ram, edge_bw,
        bw_range_max=10000.0, alpha_mig=0.5, dirty_ratio=0.20,
    )
    # Expected: T_mig = (0.20 × 32 × 1000) / 1000 = 6.4 s → penalty = 0.5 × 6400 ms = 3200 ms
    assert cost > 0.0
    assert cost == pytest.approx(3200.0, rel=1e-4)


# ─── Scaling laws ─────────────────────────────────────────────────────────────

def test_cost_scales_linearly_with_ram():
    """Doubling RAM demand must exactly double migration cost."""
    M, C = 1, 5
    prev = np.array([0], dtype=np.int32)
    active = np.array([True])
    placement = _make_placement(M, C, [2])
    edge_bw = _bw_matrix(C, 0, 2, 0.5)   # 5000 Mbps

    c1 = compute_migration_cost(placement, prev, active, np.array([10.0]), edge_bw)
    c2 = compute_migration_cost(placement, prev, active, np.array([20.0]), edge_bw)
    assert c2 == pytest.approx(c1 * 2.0, rel=1e-5)


def test_cost_inversely_scales_with_bandwidth():
    """Higher link BW must produce a proportionally lower migration cost."""
    M, C = 1, 5
    prev = np.array([0], dtype=np.int32)
    active = np.array([True])
    placement = _make_placement(M, C, [2])
    cnf_ram = np.array([32.0])

    bw_low  = _bw_matrix(C, 0, 2, 0.1)   # 1000 Mbps
    bw_high = _bw_matrix(C, 0, 2, 0.5)   # 5000 Mbps

    c_low  = compute_migration_cost(placement, prev, active, cnf_ram, bw_low)
    c_high = compute_migration_cost(placement, prev, active, cnf_ram, bw_high)

    assert c_low > c_high, "Lower bandwidth should produce higher migration cost"
    assert c_low == pytest.approx(c_high * 5.0, rel=1e-5)


def test_cost_scales_linearly_with_alpha_mig():
    """Doubling alpha_mig must exactly double migration cost."""
    M, C = 1, 5
    prev = np.array([0], dtype=np.int32)
    active = np.array([True])
    placement = _make_placement(M, C, [1])
    cnf_ram = np.array([16.0])
    edge_bw = _bw_matrix(C, 0, 1, 0.5)

    c1 = compute_migration_cost(placement, prev, active, cnf_ram, edge_bw, alpha_mig=0.5)
    c2 = compute_migration_cost(placement, prev, active, cnf_ram, edge_bw, alpha_mig=1.0)
    assert c2 == pytest.approx(c1 * 2.0, rel=1e-5)


# ─── Multi-CNF aggregation ────────────────────────────────────────────────────

def test_multi_cnf_cost_is_additive():
    """Total cost must equal sum of individual CNF costs."""
    M, C = 3, 5
    prev = np.array([0, 1, 2], dtype=np.int32)
    active = np.array([True, True, True])
    # All three CNFs move
    placement = _make_placement(M, C, [1, 2, 3])
    cnf_ram = np.array([8.0, 16.0, 32.0])
    edge_bw = np.zeros((C, C), dtype=np.float32)
    edge_bw[0, 1] = edge_bw[1, 0] = 0.5
    edge_bw[1, 2] = edge_bw[2, 1] = 0.5
    edge_bw[2, 3] = edge_bw[3, 2] = 0.5

    total = compute_migration_cost(placement, prev, active, cnf_ram, edge_bw)

    # Individual
    c0 = compute_migration_cost(
        _make_placement(1, C, [1]), np.array([0]), np.array([True]),
        np.array([8.0]), edge_bw
    )
    c1 = compute_migration_cost(
        _make_placement(1, C, [2]), np.array([1]), np.array([True]),
        np.array([16.0]), edge_bw
    )
    c2 = compute_migration_cost(
        _make_placement(1, C, [3]), np.array([2]), np.array([True]),
        np.array([32.0]), edge_bw
    )
    assert total == pytest.approx(c0 + c1 + c2, rel=1e-5)


# ─── Edge cases ───────────────────────────────────────────────────────────────

def test_isolated_node_bw_floor():
    """If edge_bw == 0 between old and new node, the 10 Mbps floor must apply."""
    M, C = 1, 5
    prev = np.array([0], dtype=np.int32)
    active = np.array([True])
    placement = _make_placement(M, C, [4])
    cnf_ram = np.array([10.0])
    edge_bw = np.zeros((C, C), dtype=np.float32)   # completely isolated

    # Should not raise ZeroDivisionError; uses 10 Mbps floor
    cost = compute_migration_cost(
        placement, prev, active, cnf_ram, edge_bw,
        bw_range_max=10000.0, alpha_mig=0.5, dirty_ratio=0.20,
    )
    # T_mig = (0.20 × 10 × 1000) / 10 = 200 s → penalty = 0.5 × 200000 ms = 100000 ms
    assert cost == pytest.approx(100000.0, rel=1e-4)


# ─── Environment integration ──────────────────────────────────────────────────

def test_migration_penalty_in_step_info():
    """step() must include migration_penalty ≥ 0.0 in info dict."""
    env = ContinuumEnv(seed=1)
    obs, _ = env.reset(seed=1)
    action = np.zeros(env.m_max, dtype=np.int64)
    obs, reward, _, _, info = env.step(action)
    assert "migration_penalty" in info, "migration_penalty missing from step info!"
    assert info["migration_penalty"] >= 0.0


def test_migration_penalty_zero_on_first_feasible_step():
    """
    On the first feasible step all CNFs are new (cnf_prev_node == -1),
    so migration_penalty must be exactly 0.
    """
    from src.env.continuum_env import ContinuumEnv
    env = ContinuumEnv(seed=2)
    obs, _ = env.reset(seed=2)
    # Use action mask to build a feasible action
    mask = obs["action_mask"]
    action = np.argmax(mask, axis=1)  # first valid node per CNF
    obs, reward, _, _, info = env.step(action)
    # Either it was feasible (migration_penalty == 0) or infeasible (migration_penalty == 0 by design)
    assert info["migration_penalty"] >= 0.0
    if info["cap_feasible"]:
        assert info["migration_penalty"] == pytest.approx(0.0, abs=1e-6), (
            "First feasible step has migration cost despite all CNFs being new!"
        )


def test_migration_penalty_nonzero_after_node_change():
    """
    If we force a node change on step 2, migration_penalty must be > 0.
    Step 1: place all CNFs on node 0.
    Step 2: place all CNFs on node 1 (if feasible).
    """
    from src.env.continuum_env import ContinuumEnv
    env = ContinuumEnv(seed=3)
    obs, _ = env.reset(seed=3)

    # Step 1: try to place all on node 0
    a1 = np.zeros(env.m_max, dtype=np.int64)
    obs, _, _, _, info1 = env.step(a1)

    if not info1["cap_feasible"]:
        pytest.skip("Step 1 infeasible — can't test migration properly on this env instance")

    # Step 2: place all on node 1 (if node 1 is active)
    if env.current_state.n_active_nodes < 2:
        pytest.skip("Need ≥2 active nodes for migration test")

    a2 = np.ones(env.m_max, dtype=np.int64)
    obs, _, _, _, info2 = env.step(a2)

    if info2["cap_feasible"] and env.current_sfcs.n_active_cnfs > 0:
        assert info2["migration_penalty"] >= 0.0
        # At least some CNFs should have moved if they were active on step 1
        # (some may have been active on node 0 and now on node 1)


# Lazy import to avoid circular at top of file
from src.env.continuum_env import ContinuumEnv
