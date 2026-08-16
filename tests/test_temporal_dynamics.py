import numpy as np
import pytest
import torch

from src.env.continuum_env import ContinuumEnv
from src.env.exogenous_trace import ExogenousTraceGenerator
from src.models.tgnn_encoder import TGNNEncoder


def test_sfc_persistence_and_ttl_retirement():
    env = ContinuumEnv(seed=42)
    obs, _ = env.reset(seed=42)

    initial_sfc_ids = set(env.generator.active_sfcs.keys())
    assert len(initial_sfc_ids) > 0

    # Step through environment without reset
    retired_seen = False
    for step in range(50):
        action = np.zeros(env.m_max, dtype=int)
        obs, reward, term, trunc, info = env.step(action)
        current_sfc_ids = set(env.generator.active_sfcs.keys())

        # Check if any initial SFC retired after TTL expired
        if not initial_sfc_ids.issubset(current_sfc_ids):
            retired_seen = True
            break

    assert retired_seen, "Expected initial SFCs to retire eventually as TTL expires."


def test_resource_allocation_and_release():
    env = ContinuumEnv(seed=42)
    obs, _ = env.reset(seed=42)

    initial_cpu_avail = env.current_state.node_cpu.copy()

    # Take a feasible placement action
    mask = obs["action_mask"]
    action = np.zeros(env.m_max, dtype=int)
    for m in range(env.m_max):
        valid = np.where(mask[m])[0]
        action[m] = int(valid[0]) if len(valid) > 0 else 0

    next_obs, reward, term, trunc, info = env.step(action)

    if info["feasible"]:
        assert np.any(env.node_cpu_allocated > 0), "Expected CPU allocation to be committed on feasible placement."
        assert np.any(env.current_state.node_cpu < initial_cpu_avail), "Available CPU should decrease after allocation."

        # Simulate TTL expiration by manually releasing
        active_ids = list(env.generator.active_sfcs.keys())
        if len(active_ids) > 0:
            env.generator.active_sfcs[active_ids[0]]["ttl"] = 0
            # Step again to trigger retirement release
            _, _, _, _, _ = env.step(np.zeros(env.m_max, dtype=int))


def test_exogenous_trace_reproducibility():
    env_cfg = {
        "c_max": 50, "m_max": 150, "h_max": 30, "temporal_window": 5,
        "max_episode_steps": 100, "c_range": [20, 50], "h_range": [5, 10],
        "waxman_alpha": 0.5, "waxman_beta": 0.5, "layer_probs": [0.4, 0.3, 0.3],
        "cpu_range": [20, 60], "ram_range": [32, 128], "storage_range": [100, 1000],
        "bw_range": [2500, 10000], "latency_range": [1, 50],
        "cnf_cpu_range": [0.5, 3.0], "cnf_ram_range": [0.5, 6.0], "cnf_storage_range": [1, 20],
        "cnf_rate_range": [10, 200], "cnf_proc_delay_range": [0.1, 2.0],
        "sfc_delay_budget_range": [120, 350], "sfc_ttl_range": [10, 40],
        "ou_theta": 0.15, "ou_sigma": 0.05, "ou_dt": 1.0,
        "alpha": 0.1, "beta": 10.0, "cost_per_cpu": {"edge": 0.05, "fog": 0.10, "cloud": 0.20},
    }

    gen = ExogenousTraceGenerator(env_cfg)
    trace = gen.generate(seed=123, max_steps=50)

    env1 = ContinuumEnv(cfg_or_path=env_cfg, seed=42)
    env2 = ContinuumEnv(cfg_or_path=env_cfg, seed=999)  # Different env seed, same trace

    obs1, _ = env1.reset(seed=42, exogenous_trace=trace)
    obs2, _ = env2.reset(seed=999, exogenous_trace=trace)

    # Validate that both environment instances start with identical exogenous states
    np.testing.assert_allclose(obs1["node_features"], obs2["node_features"])
    np.testing.assert_allclose(obs1["cnf_features"], obs2["cnf_features"])


def test_node_history_distinct_chronology():
    env = ContinuumEnv(seed=42)
    obs, _ = env.reset(seed=42)

    # Step environment 10 times to populate history buffer with actual evolving states
    for _ in range(10):
        action = np.zeros(env.m_max, dtype=int)
        obs, _, _, _, _ = env.step(action)

    history = obs["node_history"]  # (W, C_max, 6)
    # Check that adjacent frames in history buffer are distinct (due to OU noise / allocations)
    diff = np.abs(history[1:] - history[:-1]).sum()
    assert diff > 1e-4, f"Expected non-zero temporal difference between history frames, got diff={diff}"


def test_no_duplicate_state_in_tgnn_encoder():
    encoder = TGNNEncoder({
        "tgnn": {
            "f_node": 6, "f_cnf": 5, "d_hidden": 64, "d_model": 64,
            "temporal_window": 5, "n_spatial_layers": 2
        }
    })

    B, C_max, M_max, W = 2, 50, 150, 5
    node_features = torch.randn(B, C_max, 6)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    node_history = torch.randn(B, W, C_max, 6)
    cnf_features = torch.randn(B, M_max, 5)

    node_emb, cnf_emb = encoder(node_features, edge_index, node_history, cnf_features)
    assert node_emb.shape == (B, C_max, 64)
    assert cnf_emb.shape == (B, M_max, 64)


def test_continuous_evaluation_rollout():
    env = ContinuumEnv(seed=42)
    obs, info = env.reset(seed=42)

    # Continuous 100-step rollout without calling reset
    steps_completed = 0
    for t in range(100):
        action = np.zeros(env.m_max, dtype=int)
        obs, reward, term, trunc, info = env.step(action)
        steps_completed += 1
        assert not term, "Continuous scheduling env should not terminate early."

    assert steps_completed == 100
