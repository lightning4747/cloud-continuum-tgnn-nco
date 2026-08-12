import argparse
import copy
import csv
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
from src.baselines.minlp_solver import MINLPSolver
from src.baselines.static_gnn import StaticGNNActorCritic
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
    parser = argparse.ArgumentParser(description="Multi-Scenario Evaluation Engine per spec/task.md Section 3.2")
    parser.add_argument("--config", type=str, default="configs/evaluation_config.yaml", help="Path to evaluation protocol config")
    parser.add_argument("--env-config", type=str, default="configs/env_config.yaml", help="Path to environment config")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml", help="Path to model config")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/tgnn_ppo_step_200704.pt", help="Path to trained TGNN-NCO checkpoint (.pt)")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of test episodes per scenario")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save CSV and JSON results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "figures"), exist_ok=True)

    with open(args.config, "r") as f:
        eval_cfg = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_episodes = args.n_episodes
    seed = eval_cfg.get("eval_seed", 123)

    # 1. Load Trained TGNN-NCO Model Checkpoint
    tgnn_model = None
    if os.path.exists(args.checkpoint):
        print(f"--> Loading trained TGNN-NCO policy checkpoint from '{args.checkpoint}' on {device}...")
        tgnn_model = ActorCritic(model_cfg).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        tgnn_model.load_state_dict(ckpt.get("model_state", ckpt))
        tgnn_model.eval()
    else:
        print(f"WARNING: Checkpoint '{args.checkpoint}' not found. Evaluating baselines only.")

    # 2. Instantiate Baseline Solvers
    solvers = {
        "GreedyFFD": GreedyFFD(),
        "GreedyLatencyAware": GreedyLatencyAware(),
        "Static-GNN": StaticGNNActorCritic(model_cfg).to(device),
        "MINLP": MINLPSolver(timeout=10),
    }
    if tgnn_model is not None:
        solvers["TGNN-NCO"] = tgnn_model

    scenarios = {
        "In-Distribution": {"c_range": [20, 50], "m_range": [40, 150]},
        "OOD Scalability": {"c_range": [40, 70], "m_range": [100, 200]},
        "Constraint-Tight": {"resource_utilization_factor": 0.95},
        "MINLP Small-Instance": {"c_range": [10, 15], "m_range": [20, 30]},
    }

    summary_records = []

    print("\n" + "=" * 85)
    print(f"  STARTING MULTI-SCENARIO BENCHMARK EVALUATION ({n_episodes} EPISODES / SCENARIO)")
    print("=" * 85)

    for sc_name, sc_override in scenarios.items():
        print(f"\n--- Running Scenario: {sc_name} ---")

        # Load environment with scenario overrides
        env = ContinuumEnv(cfg_or_path=args.env_config, seed=seed)
        if "c_range" in sc_override:
            env.cfg["c_range"] = sc_override["c_range"]
        if "m_range" in sc_override:
            env.cfg["m_range"] = sc_override["m_range"]

        sc_metrics = {s_name: [] for s_name in solvers.keys()}

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            state = env.current_state
            sfcs = env.current_sfcs
            edge_index_np = state.edge_index

            for s_name, solver in solvers.items():
                state_copy = copy.deepcopy(state)
                sfcs_copy = copy.deepcopy(sfcs)

                if s_name == "TGNN-NCO":
                    node_f, edge_i, node_h, cnf_f, mask = obs_to_tensors(obs, edge_index_np, device)
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        actions, _, _, _ = solver.get_action_and_value(node_f, edge_i, node_h, cnf_f, action_mask=mask)
                    t_ms = (time.perf_counter() - t0) * 1000.0
                    action_np = actions.squeeze(0).cpu().numpy()
                    _, reward, term, trunc, info = env.step(action_np)

                    sc_metrics[s_name].append({
                        "feasible": info["feasible"],
                        "cost": info["deployment_cost"],
                        "mean_latency": info["mean_e2e_latency"],
                        "time_ms": t_ms,
                    })

                elif s_name == "Static-GNN":
                    node_f, edge_i, node_h, cnf_f, mask = obs_to_tensors(obs, edge_index_np, device)
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        actions, _, _, _ = solver.get_action_and_value(node_f, edge_i, node_h, cnf_f, action_mask=mask)
                    t_ms = (time.perf_counter() - t0) * 1000.0
                    action_np = actions.squeeze(0).cpu().numpy()
                    _, reward, term, trunc, info = env.step(action_np)

                    sc_metrics[s_name].append({
                        "feasible": info["feasible"],
                        "cost": info["deployment_cost"],
                        "mean_latency": info["mean_e2e_latency"],
                        "time_ms": t_ms,
                    })

                else:
                    t0 = time.perf_counter()
                    placement, info = solver.solve(state_copy, sfcs_copy)
                    t_ms = (time.perf_counter() - t0) * 1000.0

                    if placement is not None:
                        cost = env._compute_deployment_cost(placement)
                        lat_pen, e2e_lats = env._compute_latency_penalty(placement)
                        mean_lat = float(np.mean(list(e2e_lats.values()))) if e2e_lats else 0.0
                    else:
                        cost = 5000.0
                        mean_lat = 500.0

                    sc_metrics[s_name].append({
                        "feasible": info["feasible"],
                        "cost": cost,
                        "mean_latency": mean_lat,
                        "time_ms": t_ms,
                    })

        # Scenario Summary Printing
        print(f"{'Solver':20s} | {'FeasRate (%)':14s} | {'Cost ($)':10s} | {'Latency (ms)':12s} | {'Time (ms)':10s}")
        print("-" * 75)
        for s_name, recs in sc_metrics.items():
            feas_rate = float(np.mean([r["feasible"] for r in recs])) * 100.0
            mean_cost = float(np.mean([r["cost"] for r in recs]))
            mean_lat = float(np.mean([r["mean_latency"] for r in recs]))
            mean_time = float(np.mean([r["time_ms"] for r in recs]))

            print(f"{s_name:20s} | {feas_rate:13.2f}% | {mean_cost:10.2f} | {mean_lat:12.2f} | {mean_time:9.2f}ms")

            summary_records.append({
                "scenario": sc_name,
                "solver": s_name,
                "feasibility_rate": feas_rate,
                "mean_deployment_cost": mean_cost,
                "mean_e2e_latency": mean_lat,
                "inference_time_ms": mean_time,
            })

    # Save to CSV
    csv_path = os.path.join(args.results_dir, "summary_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "solver", "feasibility_rate", "mean_deployment_cost", "mean_e2e_latency", "inference_time_ms"])
        writer.writeheader()
        writer.writerows(summary_records)

    # Save to JSON
    json_path = os.path.join(args.results_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(summary_records, f, indent=2)

    print("\n" + "=" * 85)
    print(f"--> Benchmark evaluation complete! Results saved to '{csv_path}' and '{json_path}'")
    print("=" * 85)


if __name__ == "__main__":
    main()
