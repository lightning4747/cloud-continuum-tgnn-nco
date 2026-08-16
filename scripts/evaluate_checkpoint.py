#!/usr/bin/env python3
"""
Consolidated 4-Model Benchmark Evaluation Script.
Evaluates:
  1. Flagship TGNN-NCO (checkpoints/tgnn_ppo_final.pt)
  2. No-Mask TGNN (checkpoints/frozen_baselines/nomask_ppo_baseline.pt)
  3. Static-GNN (checkpoints/frozen_baselines/static_gnn_ppo_final.pt)
  4. Flat-RL Baseline (checkpoints/frozen_baselines/flat_rl_ppo_baseline.pt)
  5. Random Valid Action Sampling

Across 100 test episodes using the fixed project evaluation seed (eval_seed=123).

Computes:
- Feasibility Rate (%)
- Mean & Std Raw Reward
- Mean Deployment Cost ($)
- Mean Latency Penalty (ms) & Mean End-to-End Latency (ms)
- Constraint Violations Breakdown (CPU, RAM, Storage, Bandwidth)
- Inference Time per step (ms)
"""

import argparse
import copy
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.flat_rl import FlatRLActorCritic
from src.baselines.static_gnn import StaticGNNActorCritic
from src.models.actor_critic import ActorCritic
from src.env.continuum_env import ContinuumEnv
from scripts.evaluate import obs_to_tensors


def run_evaluation(
    env: ContinuumEnv,
    model: torch.nn.Module | None,
    n_episodes: int = 100,
    seed: int = 123,
    device: torch.device = torch.device("cpu"),
    disable_mask: bool = False,
) -> dict:
    feasible_list = []
    raw_rewards_list = []
    costs_list = []
    lat_penalties_list = []
    e2e_latencies_list = []
    times_ms_list = []

    cpu_viol_total = 0
    ram_viol_total = 0
    stor_viol_total = 0
    bw_viol_total = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        edge_index_np = env.current_state.edge_index

        if model is not None:
            node_f, edge_i, node_h, cnf_f, mask = obs_to_tensors(obs, edge_index_np, device, disable_mask=disable_mask)
            t0 = time.perf_counter()
            with torch.no_grad():
                actions, _, _, _ = model.get_action_and_value(node_f, edge_i, node_h, cnf_f, action_mask=mask)
            t_ms = (time.perf_counter() - t0) * 1000.0
            action_np = actions.squeeze(0).cpu().numpy()
        else:
            t0 = time.perf_counter()
            mask_np = obs["action_mask"]
            action_np = np.zeros(env.m_max, dtype=int)
            for m in range(env.m_max):
                valid = np.where(mask_np[m])[0]
                action_np[m] = int(np.random.choice(valid)) if len(valid) > 0 else 0
            t_ms = (time.perf_counter() - t0) * 1000.0

        _, reward, terminated, truncated, info = env.step(action_np)

        feasible_list.append(info["feasible"])
        raw_rewards_list.append(reward)
        costs_list.append(info["deployment_cost"])
        lat_penalties_list.append(info["latency_penalty"])
        e2e_latencies_list.append(info["mean_e2e_latency"])
        times_ms_list.append(t_ms)

        cap_det = info.get("cap_details", {})
        bw_det = info.get("bw_details", {})
        cpu_viol_total += cap_det.get("cpu_violations", 0)
        ram_viol_total += cap_det.get("ram_violations", 0)
        stor_viol_total += cap_det.get("stor_violations", 0)
        bw_viol_total += bw_det.get("bw_violations", 0)

    raw_rewards = np.array(raw_rewards_list)
    costs = np.array(costs_list)
    lat_penalties = np.array(lat_penalties_list)

    return {
        "feas_rate": float(np.mean(feasible_list)) * 100.0,
        "mean_raw_reward": float(np.mean(raw_rewards)),
        "std_raw_reward": float(np.std(raw_rewards)),
        "mean_cost": float(np.mean(costs)),
        "mean_lat_penalty": float(np.mean(lat_penalties)),
        "mean_e2e_latency": float(np.mean(e2e_latencies_list)),
        "cpu_viol": cpu_viol_total,
        "ram_viol": ram_viol_total,
        "stor_viol": stor_viol_total,
        "bw_viol": bw_viol_total,
        "mean_time_ms": float(np.mean(times_ms_list)),
    }


def load_model(model_cls, ckpt_path: str, base_cfg: dict, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}'")
    
    cfg = copy.deepcopy(base_cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state", ckpt)

    # Detect d_model from linear/embedding weights
    d_detected = None
    for key in ["encoder.spatial_gnn.gnn_layers.0.conv.lin_src.weight", "encoder.node_mlp.0.weight", "logit_proj.0.weight"]:
        if key in state_dict:
            d_detected = state_dict[key].shape[0]
            break
    if d_detected is None:
        for k, v in state_dict.items():
            if "weight" in k and len(v.shape) == 2 and v.shape[0] in [256, 384, 512]:
                d_detected = v.shape[0]
                break

    if d_detected:
        print(f"--> Auto-detected d_model = {d_detected} for '{ckpt_path}'")
        cfg["tgnn"]["d_model"] = d_detected
        cfg["tgnn"]["d_hidden"] = d_detected
        cfg["actor_critic"]["d_model"] = d_detected

    model = model_cls(cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgnn-checkpoint", type=str, default="checkpoints/tgnn_ppo_final.pt")
    parser.add_argument("--nomask-checkpoint", type=str, default="checkpoints/frozen_baselines/nomask_ppo_baseline.pt")
    parser.add_argument("--static-gnn-checkpoint", type=str, default="checkpoints/frozen_baselines/static_gnn_ppo_final.pt")
    parser.add_argument("--flat-rl-checkpoint", type=str, default="checkpoints/frozen_baselines/flat_rl_ppo_baseline.pt")
    parser.add_argument("--env-config", type=str, default="configs/env_config.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    with open(args.model_config, "r") as f:
        model_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ContinuumEnv(cfg_or_path=args.env_config, seed=args.seed)

    print("=" * 130)
    print(f"  CONSOLIDATED BENCHMARK EVALUATION — Flagship TGNN-NCO vs No-Mask vs Static-GNN vs Flat-RL vs Random")
    print(f"  Episodes: {args.n_episodes} | Seed: {args.seed} | Device: {device}")
    print("=" * 130)

    # 1. Load Flagship TGNN-NCO
    tgnn_res = None
    if os.path.exists(args.tgnn_checkpoint):
        print(f"\n[1/5] Loading Flagship TGNN-NCO from '{args.tgnn_checkpoint}'...")
        tgnn_model = load_model(ActorCritic, args.tgnn_checkpoint, model_cfg, device)
        tgnn_res = run_evaluation(env, tgnn_model, n_episodes=args.n_episodes, seed=args.seed, device=device, disable_mask=False)

    # 2. Load No-Mask TGNN
    nomask_res = None
    if os.path.exists(args.nomask_checkpoint):
        print(f"\n[2/5] Loading Trained No-Mask TGNN from '{args.nomask_checkpoint}'...")
        nomask_model = load_model(ActorCritic, args.nomask_checkpoint, model_cfg, device)
        nomask_res = run_evaluation(env, nomask_model, n_episodes=args.n_episodes, seed=args.seed, device=device, disable_mask=True)

    # 3. Load Static-GNN
    print(f"\n[3/5] Loading Trained Static-GNN from '{args.static_gnn_checkpoint}'...")
    sgnn_model = load_model(StaticGNNActorCritic, args.static_gnn_checkpoint, model_cfg, device)
    sgnn_res = run_evaluation(env, sgnn_model, n_episodes=args.n_episodes, seed=args.seed, device=device)

    # 4. Load Frozen Flat-RL Baseline
    print(f"\n[4/5] Loading Frozen Flat-RL Baseline from '{args.flat_rl_checkpoint}'...")
    frl_model = load_model(FlatRLActorCritic, args.flat_rl_checkpoint, model_cfg, device)
    frl_res = run_evaluation(env, frl_model, n_episodes=args.n_episodes, seed=args.seed, device=device)

    # 5. Random Valid Action Sampling
    print(f"\n[5/5] Running Pure Random-Valid Action Sampling Baseline...")
    random_res = run_evaluation(env, None, n_episodes=args.n_episodes, seed=args.seed, device=device)

    # 6. Print Consolidated Comparison Table
    print("\n" + "=" * 135)
    header = f"{'Metric':<30} | {'TGNN-NCO (Proposed)':<20} | {'No-Mask TGNN':<16} | {'Static-GNN':<16} | {'Flat-RL Baseline':<16} | {'Random Valid':<14}"
    print(header)
    print("-" * 135)

    tg_feas = f"{tgnn_res['feas_rate']:>19.2f}%" if tgnn_res else "N/A"
    nm_feas = f"{nomask_res['feas_rate']:>15.2f}%" if nomask_res else "N/A"
    print(f"{'Feasibility Rate (%)':<30} | {tg_feas:<20} | {nm_feas:<16} | {sgnn_res['feas_rate']:>15.2f}% | {frl_res['feas_rate']:>15.2f}% | {random_res['feas_rate']:>13.2f}%")

    tg_raw = f"{tgnn_res['mean_raw_reward']:>20.2f}" if tgnn_res else "N/A"
    nm_raw = f"{nomask_res['mean_raw_reward']:>16.2f}" if nomask_res else "N/A"
    print(f"{'Mean Raw Reward':<30} | {tg_raw:<20} | {nm_raw:<16} | {sgnn_res['mean_raw_reward']:>16.2f} | {frl_res['mean_raw_reward']:>16.2f} | {random_res['mean_raw_reward']:>14.2f}")

    tg_std = f"{tgnn_res['std_raw_reward']:>20.2f}" if tgnn_res else "N/A"
    nm_std = f"{nomask_res['std_raw_reward']:>16.2f}" if nomask_res else "N/A"
    print(f"{'Std Raw Reward':<30} | {tg_std:<20} | {nm_std:<16} | {sgnn_res['std_raw_reward']:>16.2f} | {frl_res['std_raw_reward']:>16.2f} | {random_res['std_raw_reward']:>14.2f}")

    tg_cost = f"{tgnn_res['mean_cost']:>20.2f}" if tgnn_res else "N/A"
    nm_cost = f"{nomask_res['mean_cost']:>16.2f}" if nomask_res else "N/A"
    print(f"{'Mean Deployment Cost ($)':<30} | {tg_cost:<20} | {nm_cost:<16} | {sgnn_res['mean_cost']:>16.2f} | {frl_res['mean_cost']:>16.2f} | {random_res['mean_cost']:>14.2f}")

    tg_lat = f"{tgnn_res['mean_lat_penalty']:>20.2f}" if tgnn_res else "N/A"
    nm_lat = f"{nomask_res['mean_lat_penalty']:>16.2f}" if nomask_res else "N/A"
    print(f"{'Mean Latency Penalty (ms)':<30} | {tg_lat:<20} | {nm_lat:<16} | {sgnn_res['mean_lat_penalty']:>16.2f} | {frl_res['mean_lat_penalty']:>16.2f} | {random_res['mean_lat_penalty']:>14.2f}")

    tg_e2e = f"{tgnn_res['mean_e2e_latency']:>20.2f}" if tgnn_res else "N/A"
    nm_e2e = f"{nomask_res['mean_e2e_latency']:>16.2f}" if nomask_res else "N/A"
    print(f"{'Mean End-to-End Latency (ms)':<30} | {tg_e2e:<20} | {nm_e2e:<16} | {sgnn_res['mean_e2e_latency']:>16.2f} | {frl_res['mean_e2e_latency']:>16.2f} | {random_res['mean_e2e_latency']:>14.2f}")

    tg_cpu = f"{tgnn_res['cpu_viol']:>20d}" if tgnn_res else "N/A"
    nm_cpu = f"{nomask_res['cpu_viol']:>16d}" if nomask_res else "N/A"
    print(f"{'CPU Violations (total)':<30} | {tg_cpu:<20} | {nm_cpu:<16} | {sgnn_res['cpu_viol']:>16d} | {frl_res['cpu_viol']:>16d} | {random_res['cpu_viol']:>14d}")

    tg_ram = f"{tgnn_res['ram_viol']:>20d}" if tgnn_res else "N/A"
    nm_ram = f"{nomask_res['ram_viol']:>16d}" if nomask_res else "N/A"
    print(f"{'RAM Violations (total)':<30} | {tg_ram:<20} | {nm_ram:<16} | {sgnn_res['ram_viol']:>16d} | {frl_res['ram_viol']:>16d} | {random_res['ram_viol']:>14d}")

    tg_stor = f"{tgnn_res['stor_viol']:>20d}" if tgnn_res else "N/A"
    nm_stor = f"{nomask_res['stor_viol']:>16d}" if nomask_res else "N/A"
    print(f"{'Storage Violations (total)':<30} | {tg_stor:<20} | {nm_stor:<16} | {sgnn_res['stor_viol']:>16d} | {frl_res['stor_viol']:>16d} | {random_res['stor_viol']:>14d}")

    tg_bw = f"{tgnn_res['bw_viol']:>20d}" if tgnn_res else "N/A"
    nm_bw = f"{nomask_res['bw_viol']:>16d}" if nomask_res else "N/A"
    print(f"{'Bandwidth Violations (total)':<30} | {tg_bw:<20} | {nm_bw:<16} | {sgnn_res['bw_viol']:>16d} | {frl_res['bw_viol']:>16d} | {random_res['bw_viol']:>14d}")

    tg_time = f"{tgnn_res['mean_time_ms']:>18.2f}ms" if tgnn_res else "N/A"
    nm_time = f"{nomask_res['mean_time_ms']:>14.2f}ms" if nomask_res else "N/A"
    print(f"{'Inference Time per step (ms)':<30} | {tg_time:<20} | {nm_time:<16} | {sgnn_res['mean_time_ms']:>14.2f}ms | {frl_res['mean_time_ms']:>14.2f}ms | {random_res['mean_time_ms']:>14.2f}ms")
    print("=" * 135)


if __name__ == "__main__":
    main()
