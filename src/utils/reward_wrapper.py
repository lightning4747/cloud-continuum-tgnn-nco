import numpy as np


class RunningMeanStd:
    """
    Tracks running mean and variance using Welford's algorithm (Stable-Baselines3 standard).
    Supports parallel batch updates for vectorized environments.
    """

    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0.0
        self.epsilon = epsilon

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / max(tot_count, 1.0)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / max(tot_count, 1.0)

        new_var = M2 / max(tot_count, 1.0)

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class VecRewardScaler:
    """
    Vectorized environment reward scaling wrapper.
    Tracks discounted returns G_t = gamma * G_{t-1} + r_t to compute running return variance,
    while scaling step rewards r_t into [-clip_reward, clip_reward].
    Preserves all metadata in info_list (including FeasRate and feasible).

    warmup_steps: Number of environment steps before normalization is applied.
    During warmup, raw rewards are clipped to [-clip_reward, clip_reward] but not normalized.
    This allows sufficient return variance to accumulate before Welford scaling kicks in.
    """

    def __init__(self, num_envs: int, gamma: float = 0.99, clip_reward: float = 10.0,
                 epsilon: float = 1e-8, warmup_steps: int = 8192):
        self.num_envs = num_envs
        self.gamma = gamma
        self.clip_reward = clip_reward
        self.epsilon = epsilon
        self.warmup_steps = warmup_steps
        self.total_steps = 0

        self.returns = np.zeros(num_envs, dtype=np.float32)
        self.running_ms = RunningMeanStd(shape=())

    def reset(self):
        self.returns = np.zeros(self.num_envs, dtype=np.float32)

    def transform(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """
        Updates discounted returns and RunningMeanStd, returning scaled step rewards.
        rewards: (num_envs,) np.ndarray
        dones: (num_envs,) np.ndarray bool
        """
        self.total_steps += self.num_envs

        # Update discounted return tracking: G_t = gamma * G_{t-1} * (1 - done) + r_t
        self.returns = self.returns * self.gamma * (1.0 - dones.astype(np.float32)) + rewards

        # Always update running variance so it is ready when warmup ends
        self.running_ms.update(self.returns)

        # Reset return tracking for completed episode environments
        self.returns[dones] = 0.0

        # During warmup: return raw rewards clipped to safe range (no normalization)
        if self.total_steps < self.warmup_steps:
            return np.clip(rewards, -self.clip_reward, self.clip_reward).astype(np.float32)

        # Post-warmup: normalize by running std of returns
        std = np.sqrt(self.running_ms.var + self.epsilon)
        scaled_rewards = rewards / std

        # Clip scaled step rewards to safe bounds [-clip_reward, clip_reward]
        scaled_rewards = np.clip(scaled_rewards, -self.clip_reward, self.clip_reward)

        return scaled_rewards.astype(np.float32)
