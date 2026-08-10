import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.env.continuum_env import ContinuumEnv
from src.models.actor_critic import ActorCritic
from src.utils.logger import TrainingLogger
from src.utils.seed import set_seed


def obs_to_tensors(obs: dict, env: ContinuumEnv, device: torch.device):
    """
    Converts observation dict arrays to PyTorch Tensors on target device.
    """
    node_feats = torch.tensor(obs["node_features"], dtype=torch.float32, device=device).unsqueeze(0)
    edge_idx = torch.tensor(env.current_state.edge_index, dtype=torch.long, device=device)
    node_hist = torch.tensor(obs["node_history"], dtype=torch.float32, device=device).unsqueeze(0)
    cnf_feats = torch.tensor(obs["cnf_features"], dtype=torch.float32, device=device).unsqueeze(0)
    action_mask = torch.tensor(obs["action_mask"], dtype=torch.bool, device=device).unsqueeze(0)

    return node_feats, edge_idx, node_hist, cnf_feats, action_mask


def compute_gae(rewards, values, next_value, dones, gamma=0.99, gae_lambda=0.95):
    """
    Computes Generalized Advantage Estimation (GAE) and Returns.
    """
    advantages = []
    gae = 0.0
    values_extended = values + [next_value]

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values_extended[step + 1] * (1.0 - float(dones[step])) - values_extended[step]
        gae = delta + gamma * gae_lambda * (1.0 - float(dones[step])) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Train TGNN-NCO PPO Placement Policy")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to model config")
    parser.add_argument("--env-config", type=str, default="configs/env_config.yaml", help="Path to env config")
    parser.add_argument("--max-steps", type=int, default=2000000, help="Max timesteps")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cuda:0, cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run test mode")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        model_cfg = yaml.safe_load(f)
    with open(args.env_config, "r") as f:
        env_cfg = yaml.safe_load(f)

    seed = model_cfg["training"]["seed"]
    set_seed(seed)

    # 1. Device Resolution
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device

    device = torch.device(device_str)

    print("=" * 60)
    print(f"  TGNN-NCO PPO Training Engine")
    print(f"  Target Device : {device_str.upper()}")
    if device.type == "cuda":
        print(f"  GPU Name      : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Total    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 60)

    # 2. Environment & Model Initialization
    env = ContinuumEnv(cfg_or_path=env_cfg, seed=seed)
    model = ActorCritic(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(model_cfg["ppo"]["learning_rate"]))
    logger = TrainingLogger(log_dir="runs/")

    ppo_cfg = model_cfg["ppo"]
    n_steps = 10 if args.dry_run else ppo_cfg["n_steps"]
    batch_size = 2 if args.dry_run else ppo_cfg["batch_size"]
    n_epochs = 1 if args.dry_run else ppo_cfg["n_epochs"]
    total_timesteps = 20 if args.dry_run else args.max_steps

    obs, _ = env.reset(seed=seed)
    global_step = 0

    while global_step < total_timesteps:
        # Rollout Buffers
        obs_node_feats_list = []
        obs_edge_idx_list = []
        obs_node_hist_list = []
        obs_cnf_feats_list = []
        obs_action_mask_list = []

        actions_list = []
        log_probs_list = []
        rewards_list = []
        values_list = []
        dones_list = []
        feasibility_list = []

        # 3. Rollout Collection Loop
        for step in range(n_steps):
            global_step += 1
            node_f, edge_i, node_h, cnf_f, mask = obs_to_tensors(obs, env, device)

            with torch.no_grad():
                action, log_prob, entropy, value = model.get_action_and_value(
                    node_f, edge_i, node_h, cnf_f, action_mask=mask
                )

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            obs_node_feats_list.append(node_f)
            obs_edge_idx_list.append(edge_i)
            obs_node_hist_list.append(node_h)
            obs_cnf_feats_list.append(cnf_f)
            obs_action_mask_list.append(mask)

            actions_list.append(action)
            log_probs_list.append(log_prob.item())
            rewards_list.append(reward)
            values_list.append(value.item())
            dones_list.append(done)
            feasibility_list.append(info["feasible"])

            obs = next_obs
            if done:
                obs, _ = env.reset()

        # 4. GAE Advantage Calculation
        with torch.no_grad():
            node_f_next, edge_i_next, node_h_next, cnf_f_next, _ = obs_to_tensors(obs, env, device)
            next_val = model.get_value(node_f_next, edge_i_next, node_h_next, cnf_f_next).item()

        advantages, returns = compute_gae(
            rewards_list, values_list, next_val, dones_list,
            gamma=ppo_cfg["gamma"], gae_lambda=ppo_cfg["gae_lambda"]
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.to(device)
        returns = returns.to(device)

        # Batched Tensor Stack
        b_node_feats = torch.cat(obs_node_feats_list, dim=0)    # (N_steps, C_max, F_node)
        b_node_hist = torch.cat(obs_node_hist_list, dim=0)      # (N_steps, W, C_max, F_node)
        b_cnf_feats = torch.cat(obs_cnf_feats_list, dim=0)      # (N_steps, M_max, F_cnf)
        b_action_masks = torch.cat(obs_action_mask_list, dim=0)  # (N_steps, M_max, C_max)
        b_actions = torch.cat(actions_list, dim=0)              # (N_steps, M_max)
        b_old_log_probs = torch.tensor(log_probs_list, dtype=torch.float32, device=device)

        # 5. PPO Batched Mini-Batch Optimization
        dataset_size = len(rewards_list)
        for epoch in range(n_epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)

            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                mb_node_f = b_node_feats[mb_idx]
                mb_node_h = b_node_hist[mb_idx]
                mb_cnf_f = b_cnf_feats[mb_idx]
                mb_mask = b_action_masks[mb_idx]
                mb_act = b_actions[mb_idx]
                mb_old_lp = b_old_log_probs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]

                # Edge index shared for same graph structure
                mb_edge_i = obs_edge_idx_list[mb_idx[0]]

                _, new_log_prob, new_entropy, new_value = model.get_action_and_value(
                    mb_node_f, mb_edge_i, mb_node_h, mb_cnf_f, action_mask=mb_mask, action=mb_act
                )

                ratio = torch.exp(new_log_prob - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - ppo_cfg["clip_epsilon"], 1.0 + ppo_cfg["clip_epsilon"]) * mb_adv

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_value.squeeze(-1), mb_ret)
                entropy_loss = -new_entropy.mean()

                loss = (policy_loss
                        + float(ppo_cfg["value_loss_coef"]) * value_loss
                        + float(ppo_cfg["entropy_coef"]) * entropy_loss)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(ppo_cfg["max_grad_norm"]))
                optimizer.step()

        # 6. Metrics & Progress Logging
        mean_reward = float(np.mean(rewards_list))
        feas_rate = float(np.mean(feasibility_list)) * 100.0
        logger.log_console(global_step, {
            "reward": mean_reward,
            "feas_rate": f"{feas_rate:.1f}%",
            "p_loss": policy_loss.item(),
            "v_loss": value_loss.item(),
        })

        if global_step % model_cfg["training"]["save_interval"] == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/tgnn_ppo_step_{global_step}.pt"
            torch.save({"model_state": model.state_dict(), "step": global_step}, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    logger.close()
    print("Training finished successfully!")


if __name__ == "__main__":
    main()
