from collections import deque
import numpy as np
from src.env.generator import NetworkState


class TemporalStateBuffer:
    """
    Sliding window buffer of historical NetworkState snapshots.
    Produces stacked node feature array of shape (W, C_max, F_node) for GRU temporal inputs.
    """

    def __init__(self, window_size: int = 5, c_max: int = 50, f_node: int = 6):
        self.window_size = window_size
        self.c_max = c_max
        self.f_node = f_node
        self.buffer = deque(maxlen=window_size)

    def reset(self, initial_state: NetworkState) -> None:
        """Fill all W slots with copies of initial_state."""
        self.buffer.clear()
        for _ in range(self.window_size):
            self.buffer.append(initial_state.node_features.copy())

    def push(self, state: NetworkState) -> None:
        """Append state to circular buffer, evicting the oldest."""
        self.buffer.append(state.node_features.copy())

    def get_history(self) -> np.ndarray:
        """
        Return (W, C_max, F_node) array, oldest first.
        """
        return np.stack(list(self.buffer), axis=0).astype(np.float32)
