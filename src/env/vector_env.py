import numpy as np
from src.env.continuum_env import ContinuumEnv


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

    def reset(self, seed: int = 42) -> tuple[dict, list[dict]]:
        obs_list = []
        info_list = []

        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=seed + i)
            obs_list.append(obs)
            info_list.append(info)

        batched_obs = self._stack_obs(obs_list)
        return batched_obs, info_list

    def step(self, actions: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """
        actions: (num_envs, M_max) numpy array
        """
        obs_list = []
        rewards = []
        terminateds = []
        truncateds = []
        info_list = []

        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            if terminated or truncated:
                obs, _ = env.reset()

            obs_list.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            info_list.append(info)

        batched_obs = self._stack_obs(obs_list)
        return (
            batched_obs,
            np.array(rewards, dtype=np.float32),
            np.array(terminateds, dtype=bool),
            np.array(truncateds, dtype=bool),
            info_list,
        )

    def _stack_obs(self, obs_list: list[dict]) -> dict:
        batched_obs = {}
        for key in obs_list[0].keys():
            batched_obs[key] = np.stack([obs[key] for obs in obs_list], axis=0)
        return batched_obs
