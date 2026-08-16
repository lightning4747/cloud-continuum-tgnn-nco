#!/usr/bin/env python3
"""
Empirical Load Parameter Sweeper for TGNN-NCO Environment Configuration.
Sweeps multi-dimensional load parameters (SFC count, CNF demands, node capacities, link bandwidths, traffic rates)
using 500-step random valid action sampling until an empirical FeasRate > 50% is achieved.

Reports exact breakdown of:
- Empirical FeasRate (%)
- Capacity violations (CPU, RAM, Storage)
- Link Bandwidth violations & path failures
- Mean raw reward, deployment cost, and latency penalty
"""

import copy
import sys
from pathlib import Path
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env.continuum_env import ContinuumEnv


def evaluate_config(cfg: dict, n_steps: int = 500, seed: int = 42) -> dict:
    env = ContinuumEnv(cfg_or_path=cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    
    feasible_count = 0
    cpu_viol = 0
    ram_viol = 0
    stor_viol = 0
    bw_viol = 0
    no_path_count = 0
    
    raw_rewards = []
    costs = []
    latencies = []

    for step in range(n_steps):
        mask = env.build_action_mask()
        action = np.zeros(env.m_max, dtype=int)
        for m in range(env.m_max):
            valid = np.where(mask[m])[0]
            action[m] = int(np.random.choice(valid)) if len(valid) > 0 else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        if info["feasible"]:
            feasible_count += 1
            
        cap_det = info.get("cap_details", {})
        bw_det = info.get("bw_details", {})
        
        cpu_viol += cap_det.get("cpu_violations", 0)
        ram_viol += cap_det.get("ram_violations", 0)
        stor_viol += cap_det.get("stor_violations", 0)
        bw_viol += bw_det.get("bw_violations", 0)
        if bw_det.get("no_path", False):
            no_path_count += 1

        raw_rewards.append(reward)
        costs.append(info.get("deployment_cost", 0.0))
        latencies.append(info.get("latency_penalty", 0.0))

        if terminated or truncated:
            obs, _ = env.reset()

    feas_rate = (feasible_count / n_steps) * 100.0
    return {
        "feas_rate": feas_rate,
        "cpu_viol": cpu_viol,
        "ram_viol": ram_viol,
        "stor_viol": stor_viol,
        "bw_viol": bw_viol,
        "no_path_count": no_path_count,
        "mean_raw_reward": float(np.mean(raw_rewards)),
        "mean_cost": float(np.mean(costs)),
        "mean_latency": float(np.mean(latencies)),
    }


def main():
    with open("configs/env_config.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)

    print("=" * 80)
    print("  TGNN-NCO EMPIRICAL MULTI-DIMENSIONAL LOAD PARAMETER SWEEP")
    print("=" * 80)

    # Candidate load parameter profiles to evaluate empirically
    candidates = [
        {
            "name": "Profile 6 — Moderate BW & Rate (h=[5,10], cnf_rate=[20,250], bw=[1000,10000])",
            "h_range": [5, 10],
            "cpu_range": [20, 60],
            "ram_range": [32, 128],
            "cnf_cpu_range": [0.5, 3.0],
            "cnf_ram_range": [0.5, 6.0],
            "cnf_rate_range": [20, 250],
            "bw_range": [1000, 10000],
            "sfc_delay_budget_range": [100, 300],
            "alpha": 0.1,
        },
        {
            "name": "Profile 7 — Calibrated Research Workload (h=[5,9], cnf_rate=[10,150], bw=[2000,10000])",
            "h_range": [5, 9],
            "cpu_range": [20, 60],
            "ram_range": [32, 128],
            "cnf_cpu_range": [0.5, 3.0],
            "cnf_ram_range": [0.5, 6.0],
            "cnf_rate_range": [10, 150],
            "bw_range": [2000, 10000],
            "sfc_delay_budget_range": [100, 350],
            "alpha": 0.1,
        },
        {
            "name": "Profile 8 — High Bandwidth (h=[6,12], cnf_rate=[10,200], bw=[3000,10000])",
            "h_range": [6, 12],
            "cpu_range": [20, 60],
            "ram_range": [32, 128],
            "cnf_cpu_range": [0.5, 3.0],
            "cnf_ram_range": [0.5, 6.0],
            "cnf_rate_range": [10, 200],
            "bw_range": [3000, 10000],
            "sfc_delay_budget_range": [100, 350],
            "alpha": 0.1,
        },
        {
            "name": "Profile 9 — Target Benchmark (h=[5,10], cnf_rate=[10,200], bw=[2500,10000])",
            "h_range": [5, 10],
            "cpu_range": [20, 60],
            "ram_range": [32, 128],
            "cnf_cpu_range": [0.5, 3.0],
            "cnf_ram_range": [0.5, 6.0],
            "cnf_rate_range": [10, 200],
            "bw_range": [2500, 10000],
            "sfc_delay_budget_range": [120, 350],
            "alpha": 0.1,
        },
    ]

    best_cand = None
    target_met_candidates = []

    for cand in candidates:
        cfg = copy.deepcopy(base_cfg)
        for k, v in cand.items():
            if k != "name":
                cfg[k] = v

        res = evaluate_config(cfg, n_steps=500, seed=42)
        
        print(f"\nEvaluating: {cand['name']}")
        print(f"  FeasRate (random valid actions) : {res['feas_rate']:.1f}%")
        print(f"  CPU Violations (total/500 steps): {res['cpu_viol']}")
        print(f"  RAM Violations (total/500 steps): {res['ram_viol']}")
        print(f"  BW Violations  (total/500 steps): {res['bw_viol']}")
        print(f"  Path Failures  (total/500 steps): {res['no_path_count']}")
        print(f"  Mean Raw Reward                 : {res['mean_raw_reward']:.2f}")
        print(f"  Mean Deployment Cost            : {res['mean_cost']:.2f}")
        print(f"  Mean Latency Penalty            : {res['mean_latency']:.2f}")

        if res['feas_rate'] >= 50.0:
            target_met_candidates.append((cand, res))

    print("\n" + "=" * 80)
    if target_met_candidates:
        print(f"  FOUND {len(target_met_candidates)} PROFILE(S) MEETING FeasRate >= 50% TARGET!")
        for cand, res in target_met_candidates:
            print(f"  --> {cand['name']}: FeasRate = {res['feas_rate']:.1f}%")
    else:
        print("  NO CANDIDATE MET FeasRate >= 50%. FURTHER SWEEP REQUIRED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
