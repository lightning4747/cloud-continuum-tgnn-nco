from abc import ABC, abstractmethod
import numpy as np
from src.env.generator import NetworkState, SFCBatch


class BaseSolver(ABC):
    """Abstract interface for all placement solvers."""

    @abstractmethod
    def solve(self, state: NetworkState, sfcs: SFCBatch) -> tuple[np.ndarray, dict]:
        """
        Compute placement matrix for given state and SFC batch.
        Returns:
          placement: (M_max, C_max) binary matrix
          info: dict containing feasibility, cost, solve_time_ms
        """
        pass

    def name(self) -> str:
        return self.__class__.__name__
