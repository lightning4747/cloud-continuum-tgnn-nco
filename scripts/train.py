import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.env.parallel_vector_env import ParallelVectorContinuumEnv
from src.env.vector_env import VectorContinuumEnv
from src.models.actor_critic import ActorCritic
from src.utils.logger import TrainingLogger
from src.utils.seed import set_seed


def obs_batch_to_tensors(batched_obs: dict, edge_index_np: np.ndarray, device: torch.device):
    """
    Converts batched observation dict arrays into PyTorch Tensors on target device
    using pinned memory and non-blocking CUDA transfers.
    """
    if device.type == "cuda":
        node_feats = torch.from_numpy(batched_obs["node_features"]).float().pin_memory().to(device, non_blocking=True)
        edge_idx = torch.from_numpy(edge_index_np).long().pin_memory().to(device, non_blocking=True)
        node_hist = torch.from_numpy(batched_obs["node_history"]).float().pin_memory().to(device, non_blocking=True)
        cnf_feats = torch.from_numpy(batched_obs["cnf_features"]).float().pin_memory().to(device, non_blocking=True)
        action_mask = torch.from_numpy(batched_obs["action_mask"]).bool().pin_memory().to(device, non_blocking=True)
    else:
        node_feats = torch.from_numpy(batched_obs["node_features"]).float().to(device)
        edge_idx = torch.from_numpy(edge_index_np).long().to(device)
        node_hist = torch.from_numpy(batched_obs["node_history"]).float().to(device)
        cnf_feats = torch.from_numpy(batched_obs["cnf_features"]).float().to(device)
        action_mask = torch.from_numpy(batched_obs["action_mask"]).bool().to(device)

    return node_feats, edge_idx, node_hist, cnf_feats, action_mask


def compute_gae_vectorized(rewards_matrix, values_matrix, next_values, dones_matrix, gamma=0.99, gae_lambda=0.95):
    """
    Computes Generalized Advantage Estimation (GAE) across vectorized env rollouts.
    rewards_matrix: (N_steps, num_envs)
    values_matrix: (N_steps, num_envs)
    next_values: (num_envs,)
    """
    n_steps, num_envs = rewards_matrix.shape
    advantages = np.zeros((n_steps, num_envs), dtype=np.float32)
    gae = np.zeros(num_envs, dtype=np.float32)

    values_ext = np.vstack([values_matrix, next_values[None, :]])

    for step in reversed(range(n_steps)):
        delta = rewards_matrix[step] + gamma * values_ext[step + 1] * (1.0 - dones_matrix[step].astype(float)) - values_ext[step]
        gae = delta + gamma * gae_lambda * (1.0 - dones_matrix[step].astype(float)) * gae
        advantages[step] = gae

    returns = advantages + values_matrix
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Train High-Throughput TGNN-NCO PPO Placement Policy")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to model config")
    parser.add_argument("--env-config", type=str, default="configs/env_config.yaml", help="Path to env config")
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel vectorized environments")
    parser.add_argument("--batch-size", type=int, default=512, help="PPO mini-batch size for GPU optimization")
    parser.add_argument("--d-model", type=int, default=256, help="Embedding dimension (d_model)")
    parser.add_argument("--n-steps", type=int, default=2048, help="Rollout steps per environment")
    parser.add_argument("--max-steps", type=int, default=2000000, help="Max total timesteps")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cuda:0, cpu)")
    parser.add_argument("--use-amp", action="store_true", default=True, help="Use Automatic Mixed Precision (AMP)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable multiprocess env vectorization")
    parser.add_argument("--dry-run", action="store_true", help="Dry run test mode")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        model_cfg = yaml.safe_load(f)
    with open(args.env_config, "r") as f:
        env_cfg = yaml.safe_load(f)

    # CLI Overrides
    if args.d_model:
        model_cfg["tgnn"]["d_model"] = args.d_model
        model_cfg["tgnn"]["d_hidden"] = args.d_model
        model_cfg["actor_critic"]["d_model"] = args.d_model

    seed = model_cfg["training"]["seed"]
    set_seed(seed)

    # Device Resolution
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device

    device = torch.device(device_str)
    use_amp = args.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    num_envs = 2 if args.dry_run else args.num_envs
    n_steps = 10 if args.dry_run else args.n_steps
    batch_size = 4 if args.dry_run else args.batch_size
    n_epochs = 1 if args.dry_run else model_cfg["ppo"]["n_epochs"]
    total_timesteps = 20 if args.dry_run else args.max_steps

    print("=" * 70)
    print(f"  TGNN-NCO High-Throughput PPO Training Engine")
    print(f"  Target Device       : {device_str.upper()}")
    print(f"  Automatic Precision : {'AMP FP16' if use_amp else 'FP32'}")
    print(f"  Parallel Multiprocess: {'Disabled' if args.no_parallel or args.dry_run else f'Enabled ({num_envs} CPU workers)'}")
    print(f"  PPO Mini-Batch      : {batch_size}")
    print(f"  Embedding d_model   : {model_cfg['tgnn']['d_model']}")
    print(f"  Rollout Steps/Env   : {n_steps}")
    print(f"  Total Rollout/Update: {num_envs * n_steps:,} transitions")
    if device.type == "cuda":
        print(f"  GPU Name            : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Available      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 70)

    # Environment Initialization
    if args.no_parallel or args.dry_run:
        vec_env = VectorContinuumEnv(num_envs=num_envs, cfg_or_path=env_cfg, seed=seed)
    else:
        vec_env = ParallelVectorContinuumEnv(num_envs=num_envs, cfg_or_path=env_cfg, seed=seed)

    batched_obs, _ = vec_env.reset(seed=seed)
    if hasattr(vec_env, "envs"):
        edge_index_np = vec_env.envs[0].current_state.edge_index
    else:
        # Padded edge index for active nodes
        edge_index_np = np.array([[0, 1], [1, 0]], dtype=np.int64)

    model = ActorCritic(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(model_cfg["ppo"]["learning_rate"]))
    logger = TrainingLogger(log_dir="runs/")

    global_step = 0

    try:
        while global_step < total_timesteps:
            t_start = time.perf_counter()

            # Rollout Tensors Storage
            obs_node_feats_list = []
            obs_node_hist_list = []
            obs_cnf_feats_list = []
            obs_action_mask_list = []

            actions_list = []
            log_probs_list = []
            rewards_list = []
            values_list = []
            dones_list = []
            feasibility_list = []

            # 1. Parallel Rollout Collection Loop
            for step in range(n_steps):
                global_step += num_envs
                node_f, edge_i, node_h, cnf_f, mask = obs_batch_to_tensors(batched_obs, edge_index_np, device)

                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        actions, log_prob, entropy, value = model.get_action_and_value(
                            node_f, edge_i, node_h, cnf_f, action_mask=mask
                        )

                actions_np = actions.cpu().numpy()  # (num_envs, M_max)
                next_batched_obs, rewards, terminateds, truncateds, info_list = vec_env.step(actions_np)
                dones = terminateds | truncateds

                obs_node_feats_list.append(node_f)
                obs_node_hist_list.append(node_h)
                obs_cnf_feats_list.append(cnf_f)
                obs_action_mask_list.append(mask)

                actions_list.append(actions)
                log_probs_list.append(log_prob)
                rewards_list.append(rewards)
                values_list.append(value.squeeze(-1).cpu().numpy())
                dones_list.append(dones)

                for info in info_list:
                    feasibility_list.append(info["feasible"])

                batched_obs = next_batched_obs

            # 2. Vectorized GAE Advantage Computation
            with torch.no_grad():
                node_f_next, edge_i_next, node_h_next, cnf_f_next, _ = obs_batch_to_tensors(batched_obs, edge_index_np, device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    next_vals = model.get_value(node_f_next, edge_i_next, node_h_next, cnf_f_next).squeeze(-1).cpu().numpy()

            rewards_mat = np.array(rewards_list, dtype=np.float32)  # (N_steps, num_envs)
            values_mat = np.array(values_list, dtype=np.float32)    # (N_steps, num_envs)
            dones_mat = np.array(dones_list, dtype=bool)            # (N_steps, num_envs)

            advantages_tensor, returns_tensor = compute_gae_vectorized(
                rewards_mat, values_mat, next_vals, dones_mat,
                gamma=float(model_cfg["ppo"]["gamma"]), gae_lambda=float(model_cfg["ppo"]["gae_lambda"])
            )

            # Flatten rollout dimensions (N_steps * num_envs, ...)
            total_samples = n_steps * num_envs
            flat_node_feats = torch.cat(obs_node_feats_list, dim=0)
            flat_node_hist = torch.cat(obs_node_hist_list, dim=0)
            flat_cnf_feats = torch.cat(obs_cnf_feats_list, dim=0)
            flat_action_masks = torch.cat(obs_action_mask_list, dim=0)
            flat_actions = torch.cat(actions_list, dim=0)
            flat_old_log_probs = torch.cat(log_probs_list, dim=0)

            flat_advantages = advantages_tensor.reshape(-1)
            flat_returns = returns_tensor.reshape(-1)

            # Advantage Normalization
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
            flat_advantages = flat_advantages.to(device)
            flat_returns = flat_returns.to(device)

            edge_i_shared = torch.tensor(edge_index_np, dtype=torch.long, device=device)

            # 3. PPO GPU Batched Mini-Batch Optimization with AMP
            for epoch in range(n_epochs):
                indices = np.arange(total_samples)
                np.random.shuffle(indices)

                for start in range(0, total_samples, batch_size):
                    end = start + batch_size
                    mb_idx = indices[start:end]

                    mb_node_f = flat_node_feats[mb_idx]
                    mb_node_h = flat_node_hist[mb_idx]
                    mb_cnf_f = flat_cnf_feats[mb_idx]
                    mb_mask = flat_action_masks[mb_idx]
                    mb_act = flat_actions[mb_idx]
                    mb_old_lp = flat_old_log_probs[mb_idx]
                    mb_adv = flat_advantages[mb_idx]
                    mb_ret = flat_returns[mb_idx]

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        _, new_log_prob, new_entropy, new_value = model.get_action_and_value(
                            mb_node_f, edge_i_shared, mb_node_h, mb_cnf_f, action_mask=mb_mask, action=mb_act
                        )

                        ratio = torch.exp(new_log_prob - mb_old_lp)
                        surr1 = ratio * mb_adv
                        surr2 = torch.clamp(ratio, 1.0 - float(model_cfg["ppo"]["clip_epsilon"]), 1.0 + float(model_cfg["ppo"]["clip_epsilon"])) * mb_adv

                        policy_loss = -torch.min(surr1, surr2).mean()
                        value_loss = F.mse_loss(new_value.squeeze(-1), mb_ret)
                        entropy_loss = -new_entropy.mean()

                        loss = (policy_loss
                                + float(model_cfg["ppo"]["value_loss_coef"]) * value_loss
                                + float(model_cfg["ppo"]["entropy_coef"]) * entropy_loss)

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(model_cfg["ppo"]["max_grad_norm"]))
                    scaler.step(optimizer)
                    scaler.update()

            # 4. Metrics & Performance Reporting
            t_elapsed = time.perf_counter() - t_start
            fps = total_samples / t_elapsed
            mean_reward = float(np.mean(rewards_mat))
            feas_rate = float(np.mean(feasibility_list)) * 100.0

            vram_allocated_gb = 0.0
            if device.type == "cuda":
                vram_allocated_gb = torch.cuda.max_memory_allocated(0) / 1e9

            logger.log_console(global_step, {
                "reward": mean_reward,
                "feas_rate": f"{feas_rate:.1f}%",
                "p_loss": policy_loss.item(),
                "v_loss": value_loss.item(),
                "fps": f"{fps:.1f}",
                "vram_gb": f"{vram_allocated_gb:.2f}GB" if device.type == "cuda" else "N/A",
            })

            if global_step % model_cfg["training"]["save_interval"] == 0:
                os.makedirs("checkpoints", exist_ok=True)
                ckpt_path = f"checkpoints/tgnn_ppo_step_{global_step}.pt"
                torch.save({"model_state": model.state_dict(), "step": global_step}, ckpt_path)
                print(f"Checkpoint saved to {ckpt_path}")

    finally:
        if hasattr(vec_env, "close"):
            vec_env.close()

    logger.close()
    print("Training finished successfully!")


if __name__ == "__main__":
    main()
