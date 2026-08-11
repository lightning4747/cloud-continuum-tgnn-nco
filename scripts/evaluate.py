import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import yaml

# Ensure project root is in sys.path for absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.greedy import GreedyFFD, GreedyLatencyAware
from src.env.continuum_env import ContinuumEnv
from src.models.actor_critic import ActorCritic


def obs_to_tensors(obs: dict, edge_index_np: np.ndarray, device: torch.device):
    """
    Converts single observation dict to PyTorch tensors with batch dimension B=1.
    """
    node_f = torch.from_numpy(obs["node_features"]).unsqueeze(0).float().to(device)
    edge_i = torch.from_numpy(edge_index_np).long().to(device)
    node_h = torch.from_numpy(obs["node_history"]).unsqueeze(0).float().to(device)
    cnf_f = torch.from_numpy(obs["cnf_features"]).unsqueeze(0).float().to(device)
    mask = torch.from_numpy(obs["action_mask"]).unsqueeze(0).bool().to(device)
    return node_f, edge_i, node_h, cnf_f, mask


def main():
    parser = argparse.ArgumentParser(description="Evaluate Placement Solvers & Saved TGNN Checkpoints")
    parser.add_argument("--config", type=str, default="configs/evaluation_config.yaml", help="Path to evaluation protocol config")
    parser.add_argument("--env-config", type=str, default="configs/env_config.yaml", help="Path to environment config")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml", help="Path to model config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained PyTorch model checkpoint (.pt)")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--output-json", type=str, default="results/evaluation_results.json", help="Output path for empirical metrics JSON")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        eval_cfg = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_cfg = yaml.safe_load(f)

    n_episodes = args.n_episodes if args.n_episodes != 10 else eval_cfg.get("n_test_episodes", 10)
    seed = eval_cfg.get("eval_seed", 123)

    env = ContinuumEnv(cfg_or_path=args.env_config, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Trained TGNN Policy Checkpoint if provided
    tgnn_model = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading trained TGNN policy checkpoint from '{args.checkpoint}' on {device}...")
        tgnn_model = ActorCritic(model_cfg).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        tgnn_model.load_state_dict(ckpt.get("model_state", ckpt))
        tgnn_model.eval()

    solvers = {
        "GreedyFFD": GreedyFFD(),
        "GreedyLatencyAware": GreedyLatencyAware(),
    }

    metrics_store = {
        name: {
            "feasibility_rate": 0.0,
            "mean_deployment_cost": 0.0,
            "mean_e2e_latency": 0.0,
            "inference_time_ms": 0.0,
            "records": [],
        }
        for name in solvers.keys()
    }

    if tgnn_model is not None:
        metrics_store["TGNN-NCO"] = {
            "feasibility_rate": 0.0,
            "mean_deployment_cost": 0.0,
            "mean_e2e_latency": 0.0,
            "inference_time_ms": 0.0,
            "records": [],
        }

    print(f"\nStarting evaluation across {n_episodes} test episodes on {device}...")

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        state = env.current_state
        sfcs = env.current_sfcs
        edge_index_np = state.edge_index

        # 1. Evaluate Heuristic Baseline Solvers
        for name, solver in solvers.items():
            state_copy = copy.deepcopy(state)
            sfcs_copy = copy.deepcopy(sfcs)
            t0 = time.perf_counter()
            placement, info = solver.solve(state_copy, sfcs_copy)
            t_ms = (time.perf_counter() - t0) * 1000.0

            # Evaluate placement on environment step
            _, cost, lat_pen, e2e_lats = env._compute_deployment_cost(placement), env._compute_deployment_cost(placement), 0.0, {}
            lat_pen, e2e_lats = env._compute_latency_penalty(placement)

            metrics_store[name]["records"].append({
                "feasible": info["feasible"],
                "cost": cost,
                "mean_latency": float(np.mean(list(e2e_lats.values()))) if e2e_lats else 0.0,
                "time_ms": t_ms,
            })

        # 2. Evaluate Trained TGNN-NCO Model if loaded
        if tgnn_model is not None:
            node_f, edge_i, node_h, cnf_f, mask = obs_to_tensors(obs, edge_index_np, device)
            t0 = time.perf_counter()
            with torch.no_grad():
                actions, _, _, _ = tgnn_model.get_action_and_value(node_f, edge_i, node_h, cnf_f, action_mask=mask)
            t_ms = (time.perf_counter() - t0) * 1000.0

            action_np = actions.squeeze(0).cpu().numpy()
            _, reward, term, trunc, info = env.step(action_np)

            metrics_store["TGNN-NCO"]["records"].append({
                "feasible": info["feasible"],
                "cost": info["deployment_cost"],
                "mean_latency": info["mean_e2e_latency"],
                "time_ms": t_ms,
            })

    # Compile Aggregate Empirical Summary
    summary = {}
    print("\n" + "=" * 75)
    print(f"  EMPIRICAL BENCHMARK EVALUATION RESULTS ({n_episodes} EPISODES)")
    print("=" * 75)
    print(f"{'Solver Name':20s} | {'Feas. Rate (%)':15s} | {'Cost ($)':10s} | {'Latency (ms)':12s} | {'Speed (ms)':10s}")
    print("-" * 75)

    for name, data in metrics_store.items():
        records = data["records"]
        feas_rate = float(np.mean([r["feasible"] for r in records])) * 100.0
        mean_cost = float(np.mean([r["cost"] for r in records]))
        mean_lat = float(np.mean([r["mean_latency"] for r in records]))
        mean_time = float(np.mean([r["time_ms"] for r in records]))

        summary[name] = {
            "feasibility_rate": feas_rate,
            "mean_deployment_cost": mean_cost,
            "mean_e2e_latency": mean_lat,
            "inference_time_ms": mean_time,
        }

        print(f"{name:20s} | {feas_rate:14.2f}% | {mean_cost:10.2f} | {mean_lat:12.2f} | {mean_time:9.2f}ms")

    print("=" * 75)

    # Save to JSON file
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Empirical evaluation metrics saved to '{args.output_json}'.\n")


if __name__ == "__main__":
    main()
