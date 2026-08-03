import argparse
import os
import torch
import yaml

from src.env.continuum_env import ContinuumEnv
from src.models.actor_critic import ActorCritic
from src.utils.logger import TrainingLogger
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train TGNN-NCO PPO Placement Policy")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to model config")
    parser.add_argument("--env-config", type=str, default="configs/env_config.yaml", help="Path to env config")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps")
    parser.add_argument("--dry-run", action="store_true", help="Dry run test mode")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        model_cfg = yaml.safe_load(f)
    with open(args.env_config, "r") as f:
        env_cfg = yaml.safe_load(f)

    seed = model_cfg["training"]["seed"]
    set_seed(seed)

    env = ContinuumEnv(cfg_or_path=env_cfg, seed=seed)
    model = ActorCritic(model_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["ppo"]["learning_rate"])
    logger = TrainingLogger(log_dir="runs/")

    obs, _ = env.reset(seed=seed)
    max_steps = 10 if args.dry_run else args.max_steps

    print(f"Starting training run... Dry Run: {args.dry_run}, Steps: {max_steps}")

    for step in range(1, max_steps + 1):
        # Convert observation dict arrays to PyTorch Tensors
        node_feats = torch.tensor(obs["node_features"], dtype=torch.float32).unsqueeze(0)
        edge_idx = torch.tensor(env.current_state.edge_index, dtype=torch.long)
        node_hist = torch.tensor(obs["node_history"], dtype=torch.float32).unsqueeze(0)
        cnf_feats = torch.tensor(obs["cnf_features"], dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(obs["action_mask"], dtype=torch.bool).unsqueeze(0)

        action, log_prob, entropy, value = model.get_action_and_value(
            node_feats, edge_idx, node_hist, cnf_feats, action_mask=mask
        )

        action_np = action.squeeze(0).detach().cpu().numpy()
        next_obs, reward, terminated, truncated, info = env.step(action_np)

        # Dummy loss step verification
        loss = -log_prob.mean() + 0.5 * value.pow(2).mean() - 0.01 * entropy.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs = next_obs
        if step % model_cfg["training"]["log_interval"] == 0 or args.dry_run:
            logger.log_console(step, {"reward": reward, "feasible": info["feasible"], "loss": loss.item()})

    logger.close()
    print("Training finished successfully!")


if __name__ == "__main__":
    main()
