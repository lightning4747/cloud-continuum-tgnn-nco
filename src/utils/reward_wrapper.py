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
    while scaling step rewards r_t into [-10.0, 10.0].
    Preserves all metadata in info_list (including FeasRate and feasible).
    """

    def __init__(self, num_envs: int, gamma: float = 0.99, clip_reward: float = 10.0, epsilon: float = 1e-8):
        self.num_envs = num_envs
        self.gamma = gamma
        self.clip_reward = clip_reward
        self.epsilon = epsilon

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
        # Update discounted return tracking: G_t = gamma * G_{t-1} * (1 - done) + r_t
        self.returns = self.returns * self.gamma * (1.0 - dones.astype(np.float32)) + rewards

        # Update running variance using Welford's algorithm
        self.running_ms.update(self.returns)

        # Scale step rewards by standard deviation of returns
        std = np.sqrt(self.running_ms.var + self.epsilon)
        scaled_rewards = rewards / std

        # Clip scaled step rewards to safe bounds [-clip_reward, clip_reward]
        scaled_rewards = np.clip(scaled_rewards, -self.clip_reward, self.clip_reward)

        # Reset return tracking for completed episode environments
        self.returns[dones] = 0.0

        return scaled_rewards.astype(np.float32)
