import argparse
import os
import numpy as np
import yaml

from src.baselines.greedy import GreedyFFD, GreedyLatencyAware
from src.env.continuum_env import ContinuumEnv


def main():
    parser = argparse.ArgumentParser(description="Evaluate Placement Solvers")
    parser.add_argument("--config", type=str, default="configs/evaluation_config.yaml", help="Path to eval config")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of test episodes")
    args = parser.parse_args()

    env = ContinuumEnv(seed=123)
    solvers = {
        "GreedyFFD": GreedyFFD(),
        "GreedyLatencyAware": GreedyLatencyAware(),
    }

    results = {name: [] for name in solvers}

    print(f"Starting evaluation across {args.n_episodes} episodes...")
    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=123 + ep)
        state = env.current_state
        sfcs = env.current_sfcs

        for name, solver in solvers.items():
            placement, info = solver.solve(state, sfcs)
            results[name].append(info["feasible"])

    print("\n--- Evaluation Results ---")
    for name, feas_list in results.items():
        feas_rate = (sum(feas_list) / len(feas_list)) * 100.0
        print(f"Solver: {name:20s} | Feasibility Rate: {feas_rate:6.2f}%")


if __name__ == "__main__":
    main()
