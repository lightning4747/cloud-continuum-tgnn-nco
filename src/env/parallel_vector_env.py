from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
from src.env.continuum_env import ContinuumEnv


class ParallelVectorContinuumEnv:
    """
    Lightweight parallel vectorized environment wrapper using ThreadPoolExecutor.
    Executes environment steps concurrently within a single process without RAM process duplication.
    """

    def __init__(self, num_envs: int = 16, cfg_or_path: dict | str = "configs/env_config.yaml", seed: int = 42):
        self.num_envs = num_envs
        self.envs = [ContinuumEnv(cfg_or_path=cfg_or_path, seed=seed + i) for i in range(num_envs)]

        # Cap max thread workers to 2x physical CPU core count to avoid Python GIL lock thrashing
        cpu_count = os.cpu_count() or 4
        max_workers = min(num_envs, cpu_count * 2)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.c_max = self.envs[0].c_max
        self.m_max = self.envs[0].m_max
        self.w = self.envs[0].w

    def reset(self, seed: int = 42) -> tuple[dict, list[dict]]:
        def _reset_env(args):
            i, env = args
            return env.reset(seed=seed + i)

        results = list(self.executor.map(_reset_env, enumerate(self.envs)))
        obs_list, info_list = zip(*results)

        batched_obs = self._stack_obs(obs_list)
        return batched_obs, list(info_list)

    def step(self, actions: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """
        actions: (num_envs, M_max) numpy array
        """
        def _step_env(args):
            i, env = args
            obs, reward, terminated, truncated, info = env.step(actions[i])
            if terminated or truncated:
                obs, _ = env.reset()
            return obs, reward, terminated, truncated, info

        results = list(self.executor.map(_step_env, enumerate(self.envs)))
        obs_list, rewards, terminateds, truncateds, info_list = zip(*results)

        batched_obs = self._stack_obs(obs_list)
        return (
            batched_obs,
            np.array(rewards, dtype=np.float32),
            np.array(terminateds, dtype=bool),
            np.array(truncateds, dtype=bool),
            list(info_list),
        )

    def _stack_obs(self, obs_list: tuple) -> dict:
        batched_obs = {}
        for key in obs_list[0].keys():
            batched_obs[key] = np.stack([obs[key] for obs in obs_list], axis=0)
        return batched_obs

    def close(self):
        self.executor.shutdown(wait=True)
