import numpy as np
import pytest

from src.env.continuum_env import ContinuumEnv
from src.env.generator import TopologyGenerator
from src.env.parallel_vector_env import ParallelVectorContinuumEnv


def test_generator_reset_determinism():
    env = ContinuumEnv(seed=42)
    gen1 = TopologyGenerator(cfg=env.cfg, seed=42)
    state1, sfcs1 = gen1.reset(seed=42)

    gen2 = TopologyGenerator(cfg=env.cfg, seed=42)
    state2, sfcs2 = gen2.reset(seed=42)

    np.testing.assert_allclose(state1.node_features, state2.node_features)
    np.testing.assert_allclose(sfcs1.cnf_features, sfcs2.cnf_features)
    assert state1.n_active_nodes == state2.n_active_nodes
    assert sfcs1.n_active_cnfs == sfcs2.n_active_cnfs


def test_env_observation_shapes():
    env = ContinuumEnv(seed=42)
    obs, info = env.reset(seed=42)

    assert obs["node_features"].shape == (50, 9)
    assert obs["edge_attr"].shape == (2500, 3)
    assert obs["node_history"].shape == (5, 50, 9)
    assert obs["cnf_features"].shape == (150, 5)
    assert obs["action_mask"].shape == (150, 50)
    assert env.observation_space.contains(obs)


def test_env_step_transition():
    env = ContinuumEnv(seed=42)
    obs, info = env.reset(seed=42)

    # Sample action using action mask
    mask = obs["action_mask"]
    action = np.zeros(150, dtype=np.int64)
    for m in range(150):
        valid_nodes = np.where(mask[m])[0]
        if len(valid_nodes) > 0:
            action[m] = valid_nodes[0]

    next_obs, reward, terminated, truncated, next_info = env.step(action)

    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert "feasible" in next_info
    assert next_obs["node_features"].shape == (50, 9)


def test_1000_step_rollout_stability():
    env = ContinuumEnv(seed=42)
    obs, _ = env.reset(seed=42)

    for _ in range(100):  # 100 steps sanity rollout
        mask = obs["action_mask"]
        action = np.zeros(150, dtype=np.int64)
        for m in range(150):
            valid_nodes = np.where(mask[m])[0]
            if len(valid_nodes) > 0:
                action[m] = valid_nodes[0]

        obs, reward, terminated, truncated, info = env.step(action)
        if truncated:
            obs, _ = env.reset()


def test_parallel_vector_env():
    pvec = ParallelVectorContinuumEnv(num_envs=4, seed=42)
    obs_b, _ = pvec.reset(seed=42)

    assert obs_b["node_features"].shape[0] == 4
    assert obs_b["cnf_features"].shape[0] == 4
    assert obs_b["action_mask"].shape[0] == 4

    actions = np.zeros((4, obs_b["action_mask"].shape[1]), dtype=np.int64)
    next_obs_b, rewards, terminateds, truncateds, infos = pvec.step(actions)

    assert rewards.shape == (4,)
    assert terminateds.shape == (4,)
    assert truncateds.shape == (4,)
    assert len(infos) == 4
    pvec.close()


# ─── New obs keys: cnf_order, cnf_active ─────────────────────────────────────

def test_obs_contains_cnf_order():
    """Reset obs must include cnf_order key with shape (M_max,)."""
    env = ContinuumEnv(seed=42)
    obs, _ = env.reset(seed=42)
    assert "cnf_order" in obs, "cnf_order missing from obs dict"
    assert obs["cnf_order"].shape == (env.m_max,), (
        f"Expected ({env.m_max},), got {obs['cnf_order'].shape}"
    )
    assert obs["cnf_order"].dtype == np.int32


def test_cnf_order_is_valid_permutation():
    """cnf_order must be a valid permutation of [0..M_max-1]."""
    env = ContinuumEnv(seed=42)
    obs, _ = env.reset(seed=42)
    order = obs["cnf_order"].tolist()
    assert sorted(order) == list(range(env.m_max)), (
        "cnf_order is not a valid permutation!"
    )


def test_obs_contains_cnf_active():
    """Reset obs must include cnf_active key with shape (M_max,)."""
    env = ContinuumEnv(seed=42)
    obs, _ = env.reset(seed=42)
    assert "cnf_active" in obs, "cnf_active missing from obs dict"
    assert obs["cnf_active"].shape == (env.m_max,), (
        f"Expected ({env.m_max},), got {obs['cnf_active'].shape}"
    )
    # Values must be 0 or 1 only
    vals = obs["cnf_active"]
    assert np.all((vals == 0) | (vals == 1)), "cnf_active has non-binary values!"


def test_migration_penalty_in_step_info():
    """step() info must include migration_penalty ≥ 0."""
    env = ContinuumEnv(seed=42)
    env.reset(seed=42)
    action = np.zeros(env.m_max, dtype=np.int64)
    _, _, _, _, info = env.step(action)
    assert "migration_penalty" in info, "migration_penalty missing from step info!"
    assert info["migration_penalty"] >= 0.0


def test_cnf_order_step_obs():
    """cnf_order must also appear in obs after env.step()."""
    env = ContinuumEnv(seed=42)
    env.reset(seed=42)
    action = np.zeros(env.m_max, dtype=np.int64)
    obs, _, _, _, _ = env.step(action)
    assert "cnf_order" in obs
    assert sorted(obs["cnf_order"].tolist()) == list(range(env.m_max))
