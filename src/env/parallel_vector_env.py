import multiprocessing as mp
import numpy as np
from src.env.continuum_env import ContinuumEnv


def _worker_loop(remote, parent_remote, cfg_or_path, seed):
    parent_remote.close()
    env = ContinuumEnv(cfg_or_path=cfg_or_path, seed=seed)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                action = data
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset()
                remote.send((obs, reward, terminated, truncated, info))
            elif cmd == "reset":
                obs, info = env.reset(seed=data)
                remote.send((obs, info))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown worker command: {cmd}")
    except Exception as e:
        remote.send(e)


class ParallelVectorContinuumEnv:
    """
    Multiprocess parallel vectorized environment wrapper.
    Executes environment steps concurrently across dedicated CPU worker processes.
    """

    def __init__(self, num_envs: int = 16, cfg_or_path: dict | str = "configs/env_config.yaml", seed: int = 42):
        self.num_envs = num_envs
        self.closed = False
        ctx = mp.get_context("spawn")

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self.ps = []

        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            p = ctx.Process(
                target=_worker_loop,
                args=(work_remote, remote, cfg_or_path, seed + i),
                daemon=True,
            )
            p.start()
            self.ps.append(p)
            work_remote.close()

        # Query single env sample properties
        sample_env = ContinuumEnv(cfg_or_path=cfg_or_path, seed=seed)
        self.c_max = sample_env.c_max
        self.m_max = sample_env.m_max
        self.w = sample_env.w
        self.sample_obs, _ = sample_env.reset(seed=seed)

    def reset(self, seed: int = 42) -> tuple[dict, list[dict]]:
        for i, remote in enumerate(self.remotes):
            remote.send(("reset", seed + i))

        results = [remote.recv() for remote in self.remotes]
        obs_list, info_list = zip(*results)

        batched_obs = self._stack_obs(obs_list)
        return batched_obs, list(info_list)

    def step(self, actions: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """
        actions: (num_envs, M_max) numpy array
        """
        for i, remote in enumerate(self.remotes):
            remote.send(("step", actions[i]))

        results = [remote.recv() for remote in self.remotes]
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
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True
