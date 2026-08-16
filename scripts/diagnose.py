#!/usr/bin/env python3
"""
Empirical Diagnostic Script for TGNN-NCO Training Failures.
Runs entirely on CPU. Answers 6 empirical questions without modifying training code.

Usage:
    python scripts/diagnose.py
    python scripts/diagnose.py --model flat_rl
    python scripts/diagnose.py --checkpoint checkpoints/flat_rl_ppo_step_45056.pt
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env.continuum_env import ContinuumEnv
from src.env.vector_env import VectorContinuumEnv
from src.models.actor_critic import ActorCritic
from src.baselines.flat_rl import FlatRLActorCritic
from src.baselines.static_gnn import StaticGNNActorCritic
from scripts.train import obs_batch_to_tensors, compute_gae_vectorized


def SECTION(title):
    print(f"\n{'='*72}\n  {title}\n{'='*72}")

def ROW(key, val):
    print(f"  {key:<48} {val}")


# ── Q1: Random-Valid-Action Feasibility Baseline ──────────────────────────────

def q1_random_valid_action_baseline(env: ContinuumEnv, n_steps: int = 500):
    SECTION("Q1 — Random-Valid-Action Feasibility Baseline")
    obs, _ = env.reset(seed=42)
    feasible_count = 0
    cap_violations_total = 0
    bw_violations_total = 0
    last_info = {}

    for step in range(n_steps):
        mask = env.build_action_mask()  # (M_max, C_max) — individual per-CNF capacity check

        action = np.zeros(env.m_max, dtype=int)
        for m in range(env.m_max):
            valid_nodes = np.where(mask[m])[0]
            action[m] = int(np.random.choice(valid_nodes))

        obs, reward, terminated, truncated, info = env.step(action)
        last_info = info

        if info["feasible"]:
            feasible_count += 1
        cap_det = info.get("cap_details", {})
        bw_det = info.get("bw_details", {})
        cap_violations_total += cap_det.get("cpu_violations", 0) + cap_det.get("ram_violations", 0) + cap_det.get("stor_violations", 0)
        bw_violations_total += bw_det.get("bw_violations", 0)

        if terminated or truncated:
            obs, _ = env.reset()

    feas_rate = feasible_count / n_steps * 100.0
    ROW("Steps run:", n_steps)
    ROW("Feasible steps:", feasible_count)
    ROW("FeasRate (random-valid-action):", f"{feas_rate:.1f}%")
    ROW("Total capacity violations (across all steps):", cap_violations_total)
    ROW("Total BW violations (across all steps):", bw_violations_total)
    ROW("Last step cap_details:", last_info.get("cap_details", "N/A"))
    ROW("Last step bw_details:", last_info.get("bw_details", "N/A"))
    ROW("Last step n_active_cnfs:", env.current_sfcs.n_active_cnfs)
    ROW("Last step n_active_nodes:", env.current_state.n_active_nodes)
    return feas_rate


# ── Q2: Raw Reward Statistics ─────────────────────────────────────────────────

def q2_raw_reward_stats(env: ContinuumEnv, n_steps: int = 200):
    SECTION("Q2 — Raw Reward Statistics (Before VecRewardScaler)")
    obs, _ = env.reset(seed=42)
    rewards, costs, latencies, betas_applied = [], [], [], []

    for step in range(n_steps):
        mask = env.build_action_mask()
        action = np.array([
            int(np.random.choice(np.where(mask[m])[0]))
            for m in range(env.m_max)
        ])
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        costs.append(info["deployment_cost"])
        latencies.append(info["latency_penalty"])
        betas_applied.append(env.cfg["beta"] if not info["feasible"] else 0.0)
        if terminated or truncated:
            obs, _ = env.reset()

    rewards = np.array(rewards)
    ROW("Raw reward — mean:", f"{rewards.mean():.4f}")
    ROW("Raw reward — std:", f"{rewards.std():.6f}")
    ROW("Raw reward — min:", f"{rewards.min():.4f}")
    ROW("Raw reward — max:", f"{rewards.max():.4f}")
    ROW("Deployment cost — mean:", f"{np.mean(costs):.4f}")
    ROW("Latency penalty — mean:", f"{np.mean(latencies):.4f}")
    ROW("β applied (when infeasible) — mean:", f"{np.mean(betas_applied):.4f}")
    ROW("Reward is near-constant (std < 0.01)?", str(rewards.std() < 0.01))
    return rewards


# ── Q3: Action Mask Statistics ────────────────────────────────────────────────

def q3_action_mask_stats(env: ContinuumEnv, n_steps: int = 100):
    SECTION("Q3 — Action Mask Validity Statistics")
    obs, _ = env.reset(seed=42)
    fill_rates, zero_mask_counts, n_active_cnfs_list, n_active_nodes_list = [], [], [], []

    for step in range(n_steps):
        mask = env.build_action_mask()  # (M_max, C_max)
        active = env.current_sfcs.cnf_active  # (M_eff,) — may exceed m_max
        min_len = min(len(mask), len(active))
        active_mask = mask[:min_len][active[:min_len]]  # (n_active, C_max)

        n_active = active.sum()
        n_active_nodes = env.current_state.n_active_nodes
        n_active_cnfs_list.append(n_active)
        n_active_nodes_list.append(n_active_nodes)

        if n_active > 0:
            fill_rate = active_mask.sum(-1) / env.c_max
            fill_rates.append(fill_rate.mean())
            zero_mask_counts.append((active_mask.sum(-1) == 0).sum())
        else:
            fill_rates.append(0.0)
            zero_mask_counts.append(0)

        action = np.array([
            int(np.random.choice(np.where(mask[m])[0]))
            for m in range(env.m_max)
        ])
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    ROW("n_active_cnfs — mean:", f"{np.mean(n_active_cnfs_list):.1f}  (range: {np.min(n_active_cnfs_list)}–{np.max(n_active_cnfs_list)})")
    ROW("n_active_nodes — mean:", f"{np.mean(n_active_nodes_list):.1f}  (range: {np.min(n_active_nodes_list)}–{np.max(n_active_nodes_list)})")
    ROW("Mean fill rate per active CNF:", f"{np.mean(fill_rates)*100:.1f}% of C_max={env.c_max} nodes")
    ROW("CNFs with ZERO individually-valid nodes/step:", f"{np.mean(zero_mask_counts):.2f}")

    # Average CNFs per node if placed uniformly
    avg_cnf_per_node = np.mean(n_active_cnfs_list) / max(np.mean(n_active_nodes_list), 1)
    ROW("Avg CNFs per node (if spread uniformly):", f"{avg_cnf_per_node:.1f}")

    # Estimate aggregate demand
    # Sample demand estimate from current state
    if env.current_sfcs.n_active_cnfs > 0:
        active_idx = np.where(env.current_sfcs.cnf_active)[0]
        avg_cnf_cpu = env.current_sfcs.cnf_cpu[active_idx].mean()
        avg_cnf_ram = env.current_sfcs.cnf_ram[active_idx].mean()
        avg_node_cpu = env.current_state.node_cpu[:env.current_state.n_active_nodes].mean()
        avg_node_ram = env.current_state.node_ram[:env.current_state.n_active_nodes].mean()
        ROW("Avg CNF CPU demand:", f"{avg_cnf_cpu:.2f} cores")
        ROW("Avg node CPU capacity:", f"{avg_node_cpu:.2f} cores")
        ROW("Estimated aggregate CPU load if uniform (cnfs*avg_cpu / nodes*avg_cap):",
            f"{avg_cnf_per_node * avg_cnf_cpu / max(avg_node_cpu, 1e-5):.2f}x node capacity")
        ROW("Estimated aggregate RAM load:",
            f"{avg_cnf_per_node * avg_cnf_ram / max(avg_node_ram, 1e-5):.2f}x node capacity")


# ── Q4: Advantage Statistics After Mini Rollout ───────────────────────────────

def q4_advantage_stats(vec_env: VectorContinuumEnv, model: ActorCritic, device: torch.device, n_steps: int = 32):
    SECTION("Q4 — GAE Advantage Statistics (32-step rollout, random-valid policy)")
    # Reset first so current_state is populated
    batched_obs, _ = vec_env.reset(seed=42)
    edge_index_np = vec_env.envs[0].current_state.edge_index

    rewards_list, values_list, dones_list, raw_rewards_log = [], [], [], []

    for step in range(n_steps):
        nf, ei, nh, cf, mask = obs_batch_to_tensors(
            batched_obs, edge_index_np, device, C_max=50, M_max=150
        )
        with torch.no_grad():
            _, _, _, value = model.get_action_and_value(nf, ei, nh, cf, action_mask=mask)

        # Random-valid actions
        am_np = batched_obs["action_mask"]
        B = am_np.shape[0]
        M = min(am_np.shape[1], 150)
        actions_np = np.zeros((B, 150), dtype=int)
        for b in range(B):
            for m in range(M):
                valid = np.where(am_np[b, m])[0]
                actions_np[b, m] = int(np.random.choice(valid)) if len(valid) > 0 else 0

        batched_obs, scaled_rewards, terminateds, truncateds, info_list = vec_env.step(actions_np)
        dones = terminateds | truncateds

        rewards_list.append(scaled_rewards)
        values_list.append(value.squeeze(-1).cpu().numpy())
        dones_list.append(dones)

    rewards_mat = np.array(rewards_list)
    values_mat = np.array(values_list)
    dones_mat = np.array(dones_list)

    nf, ei, nh, cf, _ = obs_batch_to_tensors(batched_obs, edge_index_np, device)
    with torch.no_grad():
        next_vals = model.get_value(nf, ei, nh, cf).squeeze(-1).cpu().numpy()

    advantages, returns = compute_gae_vectorized(
        rewards_mat, values_mat, next_vals, dones_mat, gamma=0.99, gae_lambda=0.95
    )
    adv_flat = advantages.reshape(-1).numpy()
    ret_flat = returns.reshape(-1).numpy()

    ROW("Scaled reward — mean:", f"{rewards_mat.mean():.6f}")
    ROW("Scaled reward — std:", f"{rewards_mat.std():.8f}")
    ROW("Advantage — mean:", f"{adv_flat.mean():.6f}")
    ROW("Advantage — std:", f"{adv_flat.std():.6f}")
    ROW("Advantage — min:", f"{adv_flat.min():.6f}")
    ROW("Advantage — max:", f"{adv_flat.max():.6f}")
    ROW("Near-zero advantages |adv| < 1e-3:",
        f"{(np.abs(adv_flat) < 1e-3).sum()} / {len(adv_flat)} ({(np.abs(adv_flat) < 1e-3).mean()*100:.1f}%)")
    ROW("Returns — mean:", f"{ret_flat.mean():.6f}")
    ROW("Value predictions — mean:", f"{values_mat.mean():.6f}")
    return advantages, returns


# ── Q5: Policy Distribution Inspection ───────────────────────────────────────

def q5_policy_distribution(vec_env: VectorContinuumEnv, model: ActorCritic, device: torch.device):
    SECTION("Q5 — Policy Probability Distribution (env 0, first 5 active CNFs)")
    edge_index_np = vec_env.envs[0].current_state.edge_index
    batched_obs, _ = vec_env.reset(seed=42)
    nf, ei, nh, cf, mask = obs_batch_to_tensors(batched_obs, edge_index_np, device)

    model.eval()
    with torch.no_grad():
        dist, _ = model.forward(nf, ei, nh, cf, action_mask=mask)

    # Per-CNF entropy
    probs = dist.probs[0].cpu().numpy()   # (M_max, C_max) for env 0
    entropies = dist.entropy()[0].cpu().numpy()  # (M_max,)
    active = vec_env.envs[0].current_sfcs.cnf_active  # (M_max,)

    active_idx = np.where(active)[0][:5]
    for m in active_idx:
        if m < probs.shape[0]:
            p = probs[m]
            top3 = np.argsort(p)[::-1][:3]
            ROW(f"  CNF {m:3d} — top-3 nodes (prob):",
                ", ".join(f"node_{top3[i]}({p[top3[i]]:.4f})" for i in range(3)))

    active_mask = active[:len(entropies)]
    active_entropies = entropies[active_mask]
    eff_nodes = np.exp(active_entropies)

    ROW("", "")
    ROW("Per-active-CNF entropy — mean:", f"{active_entropies.mean():.4f} nats")
    ROW("Per-active-CNF entropy — min:", f"{active_entropies.min():.4f} nats")
    ROW("Per-active-CNF entropy — max:", f"{active_entropies.max():.4f} nats")
    ROW("Effective node count (exp(H)) — mean:", f"{eff_nodes.mean():.2f} / {vec_env.c_max} nodes")
    ROW("Summed entropy (= logged |Entropy|):", f"{active_entropies.sum():.3f}")
    ROW("Max possible entropy (log(C_max)):", f"{np.log(vec_env.c_max):.4f} nats per CNF")


# ── Q6: Gradient Norm Analysis ────────────────────────────────────────────────

def q6_gradient_norm(vec_env: VectorContinuumEnv, model: ActorCritic, device: torch.device, optimizer):
    SECTION("Q6 — Policy Gradient Norms (1 PPO mini-batch step)")
    edge_index_np = vec_env.envs[0].current_state.edge_index
    batched_obs, _ = vec_env.reset(seed=42)
    nf, ei, nh, cf, mask = obs_batch_to_tensors(batched_obs, edge_index_np, device)

    model.train()
    _, log_prob, entropy, value = model.get_action_and_value(nf, ei, nh, cf, action_mask=mask)

    # Synthetic non-zero advantage of 1.0 to isolate gradient flow
    fake_advantage = torch.ones_like(log_prob)
    policy_loss = -(log_prob * fake_advantage).mean()
    entropy_loss = -entropy.mean()
    loss = policy_loss + 0.05 * entropy_loss

    optimizer.zero_grad()
    loss.backward()

    # Total norm
    total_norm_sq = sum(p.grad.data.norm(2).item() ** 2
                        for p in model.parameters() if p.grad is not None)
    total_norm = total_norm_sq ** 0.5

    ROW("Total gradient norm (synthetic adv=1):", f"{total_norm:.6f}")
    ROW("Policy loss:", f"{policy_loss.item():.6f}")
    ROW("Entropy loss:", f"{entropy_loss.item():.6f}")

    # Per-module group
    groups = {"encoder": 0.0, "cross_attn": 0.0, "logit_proj": 0.0, "critic": 0.0}
    other = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        pnorm_sq = p.grad.data.norm(2).item() ** 2
        matched = False
        for key in groups:
            if key in name:
                groups[key] += pnorm_sq
                matched = True
                break
        if not matched:
            other += pnorm_sq

    for key, v in groups.items():
        ROW(f"  Grad norm — {key}:", f"{v**0.5:.6f}")
    ROW("  Grad norm — other:", f"{other**0.5:.6f}")

    # KL ratio proxy
    with torch.no_grad():
        _, new_log_prob, _, _ = model.get_action_and_value(nf, ei, nh, cf, action_mask=mask)
    kl_approx = (log_prob.detach() - new_log_prob).mean().item()
    ratio = torch.exp(new_log_prob - log_prob.detach()).mean().item()
    ROW("Approx KL (before vs after 1 step):", f"{kl_approx:.6f}")
    ROW("Mean ratio r(θ):", f"{ratio:.6f}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TGNN-NCO Empirical Diagnostic Script")
    parser.add_argument("--model", default="flat_rl", choices=["tgnn", "static_gnn", "flat_rl"])
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint to load")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--env-config", default="configs/env_config.yaml")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  TGNN-NCO Empirical Diagnostic — model={args.model}")
    if args.checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*72}")

    with open(args.config) as f:
        model_cfg = yaml.safe_load(f)

    device = torch.device("cpu")

    single_env = ContinuumEnv(cfg_or_path=args.env_config, seed=42)
    vec_env = VectorContinuumEnv(num_envs=2, cfg_or_path=args.env_config, seed=42)

    MODEL_MAP = {
        "tgnn": ActorCritic,
        "flat_rl": FlatRLActorCritic,
        "static_gnn": StaticGNNActorCritic,
    }
    model = MODEL_MAP[args.model](model_cfg).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"\n  Loaded checkpoint: {args.checkpoint}")
    else:
        print("\n  [INFO] No checkpoint provided — using randomly initialized model.")

    # Run diagnostics
    q1_random_valid_action_baseline(single_env, n_steps=500)
    q2_raw_reward_stats(single_env, n_steps=200)
    q3_action_mask_stats(single_env, n_steps=100)
    q4_advantage_stats(vec_env, model, device, n_steps=32)
    q5_policy_distribution(vec_env, model, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    q6_gradient_norm(vec_env, model, device, optimizer)

    print(f"\n{'='*72}")
    print(f"  DIAGNOSTIC COMPLETE")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
