import numpy as np
import pytest
from src.utils.reward_wrapper import RunningMeanStd, VecRewardScaler


def test_running_mean_std_welford():
    rms = RunningMeanStd(shape=())

    data1 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    rms.update(data1)

    assert np.isclose(rms.mean, np.mean(data1), atol=1e-3)
    assert np.isclose(rms.var, np.var(data1), atol=1e-3)

    data2 = np.array([-10.0, -20.0, -30.0, 0.0])
    rms.update(data2)

    all_data = np.concatenate([data1, data2])
    assert np.isclose(rms.mean, np.mean(all_data), atol=1e-3)
    assert np.isclose(rms.var, np.var(all_data), atol=1e-3)


def test_vec_reward_scaler_bounds_and_reset():
    num_envs = 4
    scaler = VecRewardScaler(num_envs=num_envs, gamma=0.99, clip_reward=10.0)

    # Initial step rewards (heavy negative rewards)
    rewards = np.array([-1000.0, -2000.0, -1500.0, -500.0], dtype=np.float32)
    dones = np.array([False, False, False, False], dtype=bool)

    scaled = scaler.transform(rewards, dones)

    assert scaled.shape == (4,)
    assert np.all(scaled >= -10.0) and np.all(scaled <= 10.0)

    # Step 2 with one env done
    rewards2 = np.array([-1200.0, -800.0, -3000.0, -100.0], dtype=np.float32)
    dones2 = np.array([True, False, False, False], dtype=bool)

    scaled2 = scaler.transform(rewards2, dones2)
    assert scaled2.shape == (4,)
    assert np.all(scaled2 >= -10.0) and np.all(scaled2 <= 10.0)

    # env 0 return tracker should have been reset to 0 after done
    assert scaler.returns[0] == 0.0
