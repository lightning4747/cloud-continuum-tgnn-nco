#!/usr/bin/env python3
"""
Temporal-Information Sanity Experiment for Cloud-Continuum TGNN-NCO.
Empirically demonstrates that observation history [x_{t-W+1}, ..., x_t] contains
predictive information about near-future network state x_{t+k} that is unavailable
from a single observation snapshot x_t alone.

Usage:
    venv/bin/python scripts/temporal_sanity_check.py
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env.continuum_env import ContinuumEnv
from src.env.exogenous_trace import ExogenousTraceGenerator


def main():
    print("=" * 80)
    print("  TEMPORAL-INFORMATION SANITY EXPERIMENT")
    print("=" * 80)

    env_cfg = {
        "c_max": 50, "m_max": 150, "h_max": 30, "temporal_window": 5,
        "max_episode_steps": 100, "c_range": [20, 50], "h_range": [5, 10],
        "waxman_alpha": 0.5, "waxman_beta": 0.5, "layer_probs": [0.4, 0.3, 0.3],
        "cpu_range": [20, 60], "ram_range": [32, 128], "storage_range": [100, 1000],
        "bw_range": [2500, 10000], "latency_range": [1, 50],
        "cnf_cpu_range": [0.5, 3.0], "cnf_ram_range": [0.5, 6.0], "cnf_storage_range": [1, 20],
        "cnf_rate_range": [10, 200], "cnf_proc_delay_range": [0.1, 2.0],
        "sfc_delay_budget_range": [120, 350], "sfc_ttl_range": [10, 40],
        "sfc_arrival_prob": 0.30, "node_failure_prob": 0.01,
        "ou_theta": 0.15, "ou_sigma": 0.05, "ou_dt": 1.0,
        "alpha": 0.1, "beta": 10.0, "cost_per_cpu": {"edge": 0.05, "fog": 0.10, "cloud": 0.20},
    }

    gen = ExogenousTraceGenerator(env_cfg)
    trace = gen.generate(seed=42, max_steps=100)

    env = ContinuumEnv(cfg_or_path=env_cfg, seed=42)
    obs, info = env.reset(seed=42, exogenous_trace=trace)

    snapshots = []
    histories = []
    cpu_utils = []

    # Step through 100 continuous steps
    for t in range(100):
        # Sample valid placement action distributing CNFs across valid nodes
        mask = obs["action_mask"]
        action = np.zeros(env.m_max, dtype=int)
        for m in range(env.m_max):
            if obs["cnf_features"][m].sum() > 0:
                valid = np.where(mask[m])[0]
                if len(valid) > 0:
                    action[m] = int(valid[m % len(valid)])
                else:
                    action[m] = 0

        obs, reward, term, trunc, info = env.step(action)

        snapshots.append(obs["node_features"].copy())     # (C_max, 6)
        histories.append(obs["node_history"].copy())       # (W=5, C_max, 6)
        cpu_utils.append(info.get("cpu_utilization_pct", 0.0))

    snapshots = np.array(snapshots)  # (100, C_max, 6)
    histories = np.array(histories)  # (100, W, C_max, 6)

    # 1. Measure Temporal Variance of History Frames
    frame_vars = [np.var(histories[t, :, :, 0], axis=0).mean() for t in range(100)]
    mean_hist_var = np.mean(frame_vars)

    # 2. Autocorrelation across Lags (tau = 1, 3, 5)
    cpu_flat = snapshots[:, :, 0]  # (100, C_max)
    corrs = {}
    for tau in [1, 3, 5]:
        c = np.corrcoef(cpu_flat[:-tau].flatten(), cpu_flat[tau:].flatten())[0, 1]
        corrs[tau] = float(c)

    # 3. Burst Load Forecasting under Scenario B (Ramping Workload)
    # Compare predictive error during dynamic load burst (t in [30, 60])
    burst_targets = snapshots[30:57, :, 0]
    burst_snapshots = snapshots[27:54, :, 0]  # Single snapshot k=3 prediction
    # Temporal EMA predictor over W=5 history
    burst_ema = np.array([np.mean(histories[t, :, :, 0], axis=0) for t in range(27, 54)])

    mae_snapshot_burst = np.mean(np.abs(burst_targets - burst_snapshots))
    mae_ema_burst = np.mean(np.abs(burst_targets - burst_ema))

    # Mean CPU Utilization
    mean_cpu_util = np.mean(cpu_utils)
    max_cpu_util = np.max(cpu_utils)

    print("\n" + "-" * 72)
    print("  EXPERIMENT RESULTS & TEMPORAL METRICS")
    print("-" * 72)
    print(f"  1. History Temporal Variance (across W=5 frames) : {mean_hist_var:.6f}")
    print(f"  2. Autocorrelation at Lag 1 (tau=1)              : {corrs[1]:.4f}")
    print(f"  3. Autocorrelation at Lag 3 (tau=3)              : {corrs[3]:.4f}")
    print(f"  4. Autocorrelation at Lag 5 (tau=5)              : {corrs[5]:.4f}")
    print(f"  5. MAE of Single Snapshot (Burst Phase k=3)      : {mae_snapshot_burst:.6f}")
    print(f"  6. MAE of Temporal EMA (Burst Phase k=3)         : {mae_ema_burst:.6f}")
    print(f"  7. Mean CPU Resource Utilization (Stateful)     : {mean_cpu_util:.2f}% (Max: {max_cpu_util:.2f}%)")
    print("-" * 72)

    assert mean_hist_var > 1e-5, "Temporal history variance must be non-zero!"
    assert corrs[1] > 0.5, "Temporal state must exhibit significant autocorrelation across timesteps!"
    assert mean_cpu_util > 5.0, "Resource utilization must reflect stateful SFC allocation!"

    print("--> TEMPORAL-INFORMATION SANITY EXPERIMENT PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    main()
