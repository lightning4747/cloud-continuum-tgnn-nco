import numpy as np
from src.env.continuum_env import ContinuumEnv
from src.utils.reward_wrapper import VecRewardScaler


class VectorContinuumEnv:
    """
    Vectorized parallel wrapper for running num_envs ContinuumEnv instances simultaneously.
    Provides batched observation dictionary tensors and batched step transitions.
    """

    def __init__(self, num_envs: int = 16, cfg_or_path: dict | str = "configs/env_config.yaml", seed: int = 42):
        self.num_envs = num_envs
        self.envs = [ContinuumEnv(cfg_or_path=cfg_or_path, seed=seed + i) for i in range(num_envs)]
        self.c_max = self.envs[0].c_max
        self.m_max = self.envs[0].m_max
        self.w = self.envs[0].w
        self.reward_scaler = VecRewardScaler(num_envs=num_envs)

    def reset(self, seed: int = 42) -> tuple[dict, list[dict]]:
        obs_list = []
        info_list = []

        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=seed + i)
            obs_list.append(obs)
            info_list.append(info)

        self.reward_scaler.reset()
        batched_obs = self._stack_obs(obs_list)
        return batched_obs, info_list

    def set_beta(self, beta: float):
        """Broadcast updated infeasibility penalty to all worker environments."""
        for env in self.envs:
            env.set_beta(beta)

    def step(self, actions: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """
        actions: (num_envs, M_max) numpy array
        """
        obs_list = []
        raw_rewards = []
        terminateds = []
        truncateds = []
        info_list = []

        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            if terminated or truncated:
                obs, _ = env.reset()

            obs_list.append(obs)
            raw_rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            info_list.append(info)

        raw_rewards_np = np.array(raw_rewards, dtype=np.float32)
        dones_np = np.array(terminateds, dtype=bool) | np.array(truncateds, dtype=bool)
        scaled_rewards = self.reward_scaler.transform(raw_rewards_np, dones_np)

        batched_obs = self._stack_obs(obs_list)
        return (
            batched_obs,
            scaled_rewards,
            np.array(terminateds, dtype=bool),
            np.array(truncateds, dtype=bool),
            info_list,
        )

    def _stack_obs(self, obs_list: list[dict]) -> dict:
        batched_obs = {}
        for key in obs_list[0].keys():
            shapes = [obs[key].shape for obs in obs_list]
            if all(s == shapes[0] for s in shapes):
                batched_obs[key] = np.stack([obs[key] for obs in obs_list], axis=0)
            else:
                max_dims = [max(s[i] for s in shapes) for i in range(len(shapes[0]))]
                padded_list = []
                for obs in obs_list:
                    arr = obs[key]
                    pad_width = [(0, max_dims[i] - arr.shape[i]) for i in range(len(arr.shape))]
                    padded_list.append(np.pad(arr, pad_width, mode="constant"))
                batched_obs[key] = np.stack(padded_list, axis=0)
        return batched_obs
