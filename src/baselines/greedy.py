import time
import networkx as nx
import numpy as np

from src.baselines.base_solver import BaseSolver
from src.env.generator import NetworkState, SFCBatch


class GreedyFFD(BaseSolver):
    """
    First-Fit Decreasing Greedy Solver.
    Sorts CNFs by CPU demand descending and places on first node with sufficient capacity.
    """

    def solve(self, state: NetworkState, sfcs: SFCBatch) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        c_max = len(state.node_cpu)
        m_max = len(sfcs.cnf_cpu)
        placement = np.zeros((m_max, c_max), dtype=np.float32)

        rem_cpu = state.node_cpu.copy()
        rem_ram = state.node_ram.copy()
        rem_stor = state.node_storage.copy()

        active_cnfs = np.where(sfcs.cnf_active)[0]
        # Sort CNFs by CPU demand descending
        sorted_cnfs = sorted(active_cnfs, key=lambda m: sfcs.cnf_cpu[m], reverse=True)

        feasible = True
        for m in sorted_cnfs:
            cpu_req = sfcs.cnf_cpu[m]
            ram_req = sfcs.cnf_ram[m]
            stor_req = sfcs.cnf_storage[m]

            placed = False
            for i in range(c_max):
                if not state.node_active[i]:
                    continue
                if rem_cpu[i] >= cpu_req and rem_ram[i] >= ram_req and rem_stor[i] >= stor_req:
                    placement[m, i] = 1.0
                    rem_cpu[i] -= cpu_req
                    rem_ram[i] -= ram_req
                    rem_stor[i] -= stor_req
                    placed = True
                    break

            if not placed:
                feasible = False
                # Fallback to node 0
                placement[m, 0] = 1.0

        solve_time_ms = (time.perf_counter() - t0) * 1000.0
        return placement, {"feasible": feasible, "solve_time_ms": solve_time_ms}


class GreedyLatencyAware(BaseSolver):
    """
    Latency-Aware Greedy Solver.
    Prioritizes placing consecutive chain CNFs on shortest propagation delay paths.
    """

    def solve(self, state: NetworkState, sfcs: SFCBatch) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        c_max = len(state.node_cpu)
        m_max = len(sfcs.cnf_cpu)
        placement = np.zeros((m_max, c_max), dtype=np.float32)

        rem_cpu = state.node_cpu.copy()
        rem_ram = state.node_ram.copy()
        rem_stor = state.node_storage.copy()

        # Build NetworkX graph
        n_act = state.n_active_nodes
        g = nx.Graph()
        for i in range(n_act):
            g.add_node(i)

        for u in range(n_act):
            for v in range(u + 1, n_act):
                if state.edge_active[u, v]:
                    g.add_edge(u, v, weight=state.edge_latency[u, v])

        active_sfcs = np.where(sfcs.sfc_active)[0]
        # Sort SFCs by delay budget ascending (tightest budget first)
        sorted_sfcs = sorted(active_sfcs, key=lambda sid: sfcs.sfc_delay_budget[sid])

        feasible = True
        for sid in sorted_sfcs:
            cnf_indices = np.where((sfcs.sfc_id == sid) & sfcs.cnf_active)[0]
            prev_node = None

            for m in cnf_indices:
                cpu_req = sfcs.cnf_cpu[m]
                ram_req = sfcs.cnf_ram[m]
                stor_req = sfcs.cnf_storage[m]

                best_node = None
                best_score = float("inf")

                for i in range(c_max):
                    if not state.node_active[i]:
                        continue
                    if rem_cpu[i] >= cpu_req and rem_ram[i] >= ram_req and rem_stor[i] >= stor_req:
                        if prev_node is None or prev_node == i:
                            dist = 0.0
                        else:
                            try:
                                dist = nx.shortest_path_length(g, source=prev_node, target=i, weight="weight")
                            except (nx.NetworkXNoPath, nx.NodeNotFound):
                                dist = 1000.0

                        # Score combines propagation latency and cost
                        score = dist
                        if score < best_score:
                            best_score = score
                            best_node = i

                if best_node is not None:
                    placement[m, best_node] = 1.0
                    rem_cpu[best_node] -= cpu_req
                    rem_ram[best_node] -= ram_req
                    rem_stor[best_node] -= stor_req
                    prev_node = best_node
                else:
                    feasible = False
                    placement[m, 0] = 1.0

        solve_time_ms = (time.perf_counter() - t0) * 1000.0
        return placement, {"feasible": feasible, "solve_time_ms": solve_time_ms}
