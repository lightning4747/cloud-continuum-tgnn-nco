import time
from gekko import GEKKO
import numpy as np

from src.baselines.base_solver import BaseSolver
from src.env.generator import NetworkState, SFCBatch


class MINLPSolver(BaseSolver):
    """
    Exact MILP baseline solver using GEKKO.
    Only applicable for small topologies (C <= 15, M <= 30).
    """

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def solve(self, state: NetworkState, sfcs: SFCBatch) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        n_act = state.n_active_nodes
        m_act = sfcs.n_active_cnfs

        if n_act > 15 or m_act > 30:
            # Beyond exact solver limits -> fallback to zeros
            return np.zeros((len(sfcs.cnf_cpu), len(state.node_cpu)), dtype=np.float32), {
                "feasible": False,
                "out_of_scope": True,
                "solve_time_ms": 0.0,
            }

        m = GEKKO(remote=False)
        m.options.SOLVER = 1  # APOPT solver for MINLP/MILP
        m.options.TIME_LOC = self.timeout

        # Decision variables x[m, i] binary
        x = m.Array(m.Var, (m_act, n_act), lb=0, ub=1, integer=True)

        # C1: Each active CNF placed on exactly one active node
        for i_cnf in range(m_act):
            m.Equation(m.sum([x[i_cnf, i_node] for i_node in range(n_act)]) == 1)

        # C2: Node CPU Capacity
        for i_node in range(n_act):
            m.Equation(
                m.sum([x[i_cnf, i_node] * float(sfcs.cnf_cpu[i_cnf]) for i_cnf in range(m_act)])
                <= float(state.node_cpu[i_node])
            )

        # C3: Node RAM Capacity
        for i_node in range(n_act):
            m.Equation(
                m.sum([x[i_cnf, i_node] * float(sfcs.cnf_ram[i_cnf]) for i_cnf in range(m_act)])
                <= float(state.node_ram[i_node])
            )

        # C4: Node Storage Capacity
        for i_node in range(n_act):
            m.Equation(
                m.sum([x[i_cnf, i_node] * float(sfcs.cnf_storage[i_cnf]) for i_cnf in range(m_act)])
                <= float(state.node_storage[i_node])
            )

        # Objective Function: Minimize placement cost
        cost_map = {0: 0.05, 1: 0.10, 2: 0.20}
        node_costs = [cost_map[int(state.node_layer[i])] for i in range(n_act)]

        obj = m.sum(
            [
                x[i_cnf, i_node] * float(sfcs.cnf_cpu[i_cnf]) * node_costs[i_node]
                for i_cnf in range(m_act)
                for i_node in range(n_act)
            ]
        )
        m.Minimize(obj)

        feasible = True
        try:
            m.solve(disp=False)
            placement = np.zeros((len(sfcs.cnf_cpu), len(state.node_cpu)), dtype=np.float32)
            for i_cnf in range(m_act):
                for i_node in range(n_act):
                    if x[i_cnf, i_node].value[0] > 0.5:
                        placement[i_cnf, i_node] = 1.0
        except Exception:
            feasible = False
            placement = np.zeros((len(sfcs.cnf_cpu), len(state.node_cpu)), dtype=np.float32)

        solve_time_ms = (time.perf_counter() - t0) * 1000.0
        return placement, {"feasible": feasible, "solve_time_ms": solve_time_ms}
