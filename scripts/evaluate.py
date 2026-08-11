import argparse
import copy
import sys
from pathlib import Path
import yaml

# Ensure project root is in sys.path for absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.greedy import GreedyFFD, GreedyLatencyAware
from src.env.continuum_env import ContinuumEnv


def main():
    parser = argparse.ArgumentParser(description="Evaluate Placement Solvers")
    parser.add_argument("--config", type=str, default="configs/evaluation_config.yaml", help="Path to evaluation protocol config")
    parser.add_argument("--env-config", type=str, default="configs/env_config.yaml", help="Path to environment config")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of test episodes")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        eval_cfg = yaml.safe_load(f)

    n_episodes = args.n_episodes if args.n_episodes != 10 else eval_cfg.get("n_test_episodes", 10)
    seed = eval_cfg.get("eval_seed", 123)

    env = ContinuumEnv(cfg_or_path=args.env_config, seed=seed)
    solvers = {
        "GreedyFFD": GreedyFFD(),
        "GreedyLatencyAware": GreedyLatencyAware(),
    }

    results = {name: [] for name in solvers}

    print(f"Starting evaluation across {n_episodes} episodes using env config '{args.env_config}'...")
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        state = env.current_state
        sfcs = env.current_sfcs

        for name, solver in solvers.items():
            state_copy = copy.deepcopy(state)
            sfcs_copy = copy.deepcopy(sfcs)
            placement, info = solver.solve(state_copy, sfcs_copy)
            results[name].append(info["feasible"])

    print("\n--- Evaluation Results ---")
    for name, feas_list in results.items():
        feas_rate = (sum(feas_list) / len(feas_list)) * 100.0
        print(f"Solver: {name:20s} | Feasibility Rate: {feas_rate:6.2f}%")


if __name__ == "__main__":
    main()
