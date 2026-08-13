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

    assert obs["node_features"].shape == (50, 6)
    assert obs["edge_attr"].shape == (2500, 3)
    assert obs["node_history"].shape == (5, 50, 6)
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
    assert next_obs["node_features"].shape == (50, 6)


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
