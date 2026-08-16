#!/usr/bin/env python3
"""
Multi-Scenario Exogenous Benchmark Evaluation Engine for Cloud-Continuum Placement Policies.
Features continuous multi-step rollouts (T=100) per episode across 6 controlled temporal stress scenarios
using deterministic ExogenousTrace objects for exact cross-solver environment reproducibility.

Usage:
    python scripts/evaluate.py --smoke-test
    python scripts/evaluate.py --n-episodes 10 --episode-steps 100
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.flat_rl import FlatRLActorCritic
from src.baselines.greedy import GreedyFFD, GreedyLatencyAware
from src.baselines.static_gnn import StaticGNNActorCritic
from src.env.continuum_env import ContinuumEnv
from src.env.exogenous_trace import ExogenousTraceGenerator
from src.models.actor_critic import ActorCritic


class RandomValidSolver:
    """
    Baseline solver that samples uniformly random valid placement nodes per active CNF.
    """
    def get_action(self, action_mask: np.ndarray, m_max: int) -> np.ndarray:
        action = np.zeros(m_max, dtype=int)
        for m in range(m_max):
            valid = np.where(action_mask[m])[0]
            action[m] = int(np.random.choice(valid)) if len(valid) > 0 else 0
        return action


def obs_to_tensors(obs: dict, edge_index_np: np.ndarray, device: torch.device, disable_mask: bool = False):
    node_f = torch.from_numpy(obs["node_features"]).unsqueeze(0).float().to(device)
    edge_i = torch.from_numpy(edge_index_np).long().to(device)
    node_h = torch.from_numpy(obs["node_history"]).unsqueeze(0).float().to(device)
    cnf_f = torch.from_numpy(obs["cnf_features"]).unsqueeze(0).float().to(device)
    if disable_mask:
        mask = torch.ones_like(torch.from_numpy(obs["action_mask"])).unsqueeze(0).bool().to(device)
    else:
        mask = torch.from_numpy(obs["action_mask"]).unsqueeze(0).bool().to(device)
    return node_f, edge_i, node_h, cnf_f, mask


def main():
    parser = argparse.ArgumentParser(description="Multi-Scenario Exogenous Temporal Evaluation Engine")
    parser.add_argument("--config", type=str, default="configs/evaluation_config.yaml")
    parser.add_argument("--env-config", type=str, default="configs/env_config.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/tgnn_ppo_step_200704.pt")
    parser.add_argument("--static-gnn-checkpoint", type=str, default=None)
    parser.add_argument("--flat-rl-checkpoint", type=str, default=None)
    parser.add_argument("--nomask-checkpoint", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=10, help="Episodes per scenario")
    parser.add_argument("--episode-steps", type=int, default=100, help="Steps per continuous episode horizon T")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast 2-episode x 10-step sanity evaluation")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_episodes = 2
        args.episode_steps = 10
        print("--> Running in SMOKE-TEST mode (2 episodes x 10 steps per scenario)")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "figures"), exist_ok=True)

    with open(args.config, "r") as f:
        eval_cfg = yaml.safe_load(f)
    with open(args.env_config, "r") as f:
        env_cfg = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_episodes = args.n_episodes
    episode_steps = args.episode_steps
    seed_base = eval_cfg.get("eval_seed", 123)

    # 1. Initialize Benchmark Solvers
    solvers = {
        "RandomValid": RandomValidSolver(),
        "GreedyFFD": GreedyFFD(),
        "GreedyLatencyAware": GreedyLatencyAware(),
    }

    # Neural Policy Solvers
    if os.path.exists(args.checkpoint):
        print(f"--> Loading TGNN-NCO checkpoint from '{args.checkpoint}'...")
        tgnn_model = ActorCritic(model_cfg).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        tgnn_model.load_state_dict(ckpt.get("model_state", ckpt))
        tgnn_model.eval()
        solvers["TGNN-NCO"] = tgnn_model

    if args.static_gnn_checkpoint and os.path.exists(args.static_gnn_checkpoint):
        print(f"--> Loading Static-GNN checkpoint from '{args.static_gnn_checkpoint}'...")
        sgnn_model = StaticGNNActorCritic(model_cfg).to(device)
        ckpt = torch.load(args.static_gnn_checkpoint, map_location=device)
        sgnn_model.load_state_dict(ckpt.get("model_state", ckpt))
        sgnn_model.eval()
        solvers["Static-GNN"] = sgnn_model
    else:
        solvers["Static-GNN"] = StaticGNNActorCritic(model_cfg).to(device).eval()

    if args.flat_rl_checkpoint and os.path.exists(args.flat_rl_checkpoint):
        print(f"--> Loading Flat-RL checkpoint from '{args.flat_rl_checkpoint}'...")
        frl_model = FlatRLActorCritic(model_cfg).to(device)
        ckpt = torch.load(args.flat_rl_checkpoint, map_location=device)
        frl_model.load_state_dict(ckpt.get("model_state", ckpt))
        frl_model.eval()
        solvers["Flat-RL"] = frl_model
    else:
        solvers["Flat-RL"] = FlatRLActorCritic(model_cfg).to(device).eval()

    if args.nomask_checkpoint and os.path.exists(args.nomask_checkpoint):
        print(f"--> Loading No-Mask checkpoint from '{args.nomask_checkpoint}'...")
        nomask_model = ActorCritic(model_cfg).to(device)
        ckpt = torch.load(args.nomask_checkpoint, map_location=device)
        nomask_model.load_state_dict(ckpt.get("model_state", ckpt))
        nomask_model.eval()
        solvers["No-Mask"] = nomask_model

    # 2. Define 6 Controlled Temporal Stress Regimes
    scenarios = {
        "A_stable_workload": {"scenario": "A_stable"},
        "B_load_burst": {"scenario": "B_load_burst"},
        "C_node_failure": {"scenario": "C_node_failure", "node_failure_prob": 0.05},
        "D_link_degradation": {"scenario": "D_link_degradation"},
        "E_sfc_churn": {"scenario": "E_sfc_churn", "sfc_arrival_prob": 0.50},
        "F_recovery": {"scenario": "F_recovery"},
    }

    summary_records = []
    trace_gen = ExogenousTraceGenerator(env_cfg)

    print("\n" + "=" * 90)
    print(f"  STARTING EXOGENOUS TEMPORAL BENCHMARK EVALUATION ({n_episodes} EPISODES x {episode_steps} STEPS)")
    print("=" * 90)

    for sc_name, sc_override in scenarios.items():
        print(f"\n--- Running Temporal Scenario: {sc_name} ---")

        sc_metrics = {s_name: [] for s_name in solvers.keys()}

        for ep in range(n_episodes):
            ep_seed = seed_base + ep * 100
            # Pre-generate deterministic ExogenousTrace for this episode & scenario
            trace = trace_gen.generate(seed=ep_seed, max_steps=episode_steps, scenario_override=sc_override)

            for s_name, solver in solvers.items():
                env = ContinuumEnv(cfg_or_path=env_cfg, seed=ep_seed)
                env.max_steps = episode_steps
                obs, _ = env.reset(seed=ep_seed, exogenous_trace=trace)

                feas_count = 0
                costs = []
                latencies = []
                rewards = []
                infer_times = []
                cpu_utils = []
                cap_viols = 0
                bw_viols = 0

                # Continuous Multi-step Rollout Loop (T steps without reset)
                for t in range(episode_steps):
                    edge_index_np = env.current_state.edge_index

                    if isinstance(solver, torch.nn.Module):
                        disable_m = (s_name == "No-Mask")
                        node_f, edge_i, node_h, cnf_f, mask = obs_to_tensors(obs, edge_index_np, device, disable_mask=disable_m)
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            actions, _, _, _ = solver.get_action_and_value(node_f, edge_i, node_h, cnf_f, action_mask=mask)
                        t_ms = (time.perf_counter() - t0) * 1000.0
                        action_np = actions.squeeze(0).cpu().numpy()
                        obs, reward, term, trunc, info = env.step(action_np)

                    elif isinstance(solver, RandomValidSolver):
                        t0 = time.perf_counter()
                        action_np = solver.get_action(obs["action_mask"], env.m_max)
                        t_ms = (time.perf_counter() - t0) * 1000.0
                        obs, reward, term, trunc, info = env.step(action_np)

                    else:
                        # Heuristic baselines (GreedyFFD, GreedyLatencyAware)
                        t0 = time.perf_counter()
                        placement, info_sol = solver.solve(copy.deepcopy(env.current_state), copy.deepcopy(env.current_sfcs))
                        t_ms = (time.perf_counter() - t0) * 1000.0
                        if placement is not None:
                            action_np = np.argmax(placement, axis=1)
                        else:
                            action_np = np.zeros(env.m_max, dtype=int)
                        obs, reward, term, trunc, info = env.step(action_np)

                    if info["feasible"]:
                        feas_count += 1
                    costs.append(info["deployment_cost"])
                    latencies.append(info["mean_e2e_latency"])
                    rewards.append(reward)
                    infer_times.append(t_ms)
                    cpu_utils.append(info.get("cpu_utilization_pct", 0.0))

                    cap_det = info.get("cap_details", {})
                    bw_det = info.get("bw_details", {})
                    cap_viols += cap_det.get("cpu_violations", 0) + cap_det.get("ram_violations", 0) + cap_det.get("stor_violations", 0)
                    bw_viols += bw_det.get("bw_violations", 0)

                sc_metrics[s_name].append({
                    "feas_rate": (feas_count / episode_steps) * 100.0,
                    "cost": float(np.mean(costs)),
                    "latency": float(np.mean(latencies)),
                    "reward": float(np.mean(rewards)),
                    "infer_time_ms": float(np.mean(infer_times)),
                    "cpu_util_pct": float(np.mean(cpu_utils)),
                    "cap_violations": cap_viols,
                    "bw_violations": bw_viols,
                })

        # Scenario Summary Table
        print(f"\n{'Solver':18s} | {'FeasRate (%)':12s} | {'Cost ($)':9s} | {'Latency (ms)':12s} | {'Reward':9s} | {'CPU Util (%)':12s} | {'Time (ms)':9s}")
        print("-" * 90)
        for s_name, recs in sc_metrics.items():
            mean_feas = float(np.mean([r["feas_rate"] for r in recs]))
            mean_cost = float(np.mean([r["cost"] for r in recs]))
            mean_lat = float(np.mean([r["latency"] for r in recs]))
            mean_rew = float(np.mean([r["reward"] for r in recs]))
            mean_util = float(np.mean([r["cpu_util_pct"] for r in recs]))
            mean_time = float(np.mean([r["infer_time_ms"] for r in recs]))

            print(f"{s_name:18s} | {mean_feas:11.2f}% | {mean_cost:9.2f} | {mean_lat:12.2f} | {mean_rew:9.2f} | {mean_util:11.2f}% | {mean_time:8.2f}ms")

            summary_records.append({
                "scenario": sc_name,
                "solver": s_name,
                "feasibility_rate": mean_feas,
                "mean_deployment_cost": mean_cost,
                "mean_e2e_latency": mean_lat,
                "mean_reward": mean_rew,
                "cpu_utilization_pct": mean_util,
                "inference_time_ms": mean_time,
            })

    # Save Results
    csv_path = os.path.join(args.results_dir, "summary_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scenario", "solver", "feasibility_rate", "mean_deployment_cost",
            "mean_e2e_latency", "mean_reward", "cpu_utilization_pct", "inference_time_ms"
        ])
        writer.writeheader()
        writer.writerows(summary_records)

    json_path = os.path.join(args.results_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(summary_records, f, indent=2)

    print("\n" + "=" * 90)
    print(f"--> Exogenous Benchmark Evaluation Complete! Results saved to '{csv_path}' and '{json_path}'")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
