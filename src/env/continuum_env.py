import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import floyd_warshall
import yaml

from src.env.generator import NetworkState, SFCBatch, TopologyGenerator
from src.env.state_buffer import TemporalStateBuffer


class ContinuumEnv(gym.Env):
    """
    Gymnasium-compliant environment for dynamic CNF placement on the Cloud-Continuum.
    Optimized with precomputed Floyd-Warshall routing matrices for fast environment steps.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg_or_path: dict | str = "configs/env_config.yaml", seed: int = 42):
        super().__init__()
        if isinstance(cfg_or_path, str):
            with open(cfg_or_path, "r") as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = cfg_or_path

        self.c_max = self.cfg["c_max"]
        self.m_max = self.cfg["m_max"]
        self.h_max = self.cfg["h_max"]
        self.w = self.cfg["temporal_window"]
        self.max_steps = self.cfg["max_episode_steps"]

        self.generator = TopologyGenerator(self.cfg, seed=seed)
        self.state_buffer = TemporalStateBuffer(window_size=self.w, c_max=self.c_max, f_node=6)

        # Spaces definition
        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(0.0, 1.0, shape=(self.c_max, 6), dtype=np.float32),
                "edge_attr": spaces.Box(0.0, 1.0, shape=(self.c_max * self.c_max, 3), dtype=np.float32),
                "node_history": spaces.Box(0.0, 1.0, shape=(self.w, self.c_max, 6), dtype=np.float32),
                "cnf_features": spaces.Box(0.0, 1.0, shape=(self.m_max, 5), dtype=np.float32),
                "action_mask": spaces.MultiBinary((self.m_max, self.c_max)),
            }
        )

        self.action_space = spaces.MultiDiscrete([self.c_max] * self.m_max)

        self.current_state: NetworkState = None
        self.current_sfcs: SFCBatch = None
        self.current_step = 0
        self._dist_matrix = None
        self._pred_matrix = None

    def _update_shortest_paths(self):
        """
        Precomputes all-pairs shortest path distance and predecessor matrices using Floyd-Warshall.
        Executed once per topology state update for fast O(1) routing lookups.
        """
        n_act = self.current_state.n_active_nodes
        adj = np.full((n_act, n_act), np.inf, dtype=np.float64)
        np.fill_diagonal(adj, 0.0)

        for u in range(n_act):
            for v in range(u + 1, n_act):
                if self.current_state.edge_active[u, v]:
                    lat = self.current_state.edge_latency[u, v]
                    adj[u, v] = lat
                    adj[v, u] = lat

        dist, pred = floyd_warshall(adj, return_predecessors=True)
        self._dist_matrix = dist
        self._pred_matrix = pred

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.current_state, self.current_sfcs = self.generator.reset(seed=seed)
        self.state_buffer.reset(self.current_state)
        self._update_shortest_paths()

        obs = self._build_obs()
        info = {
            "n_active_nodes": self.current_state.n_active_nodes,
            "n_active_cnfs": self.current_sfcs.n_active_cnfs,
            "n_active_sfcs": self.current_sfcs.n_active_sfcs,
        }
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self.current_step += 1
        placement_matrix = self._decode_action(action)

        # Check hard capacity & resource constraints
        cap_feasible, cap_details = self._check_capacity_constraints(placement_matrix)
        bw_feasible, bw_details = self._check_bw_constraints(placement_matrix)
        feasible = cap_feasible and bw_feasible

        # Compute costs and penalties
        cost = self._compute_deployment_cost(placement_matrix)
        latency_penalty, e2e_latencies = self._compute_latency_penalty(placement_matrix)
        reward = self._compute_reward(cost, latency_penalty, feasible)

        # Advance environment state
        self.current_state, self.current_sfcs = self.generator.step(self.current_state, self.current_sfcs)
        self.state_buffer.push(self.current_state)
        self._update_shortest_paths()

        obs = self._build_obs()
        truncated = self.current_step >= self.max_steps
        terminated = False  # Continuous scheduling environment

        info = {
            "feasible": feasible,
            "cap_feasible": cap_feasible,
            "bw_feasible": bw_feasible,
            "deployment_cost": cost,
            "latency_penalty": latency_penalty,
            "mean_e2e_latency": float(np.mean(list(e2e_latencies.values()))) if e2e_latencies else 0.0,
            "cap_details": cap_details,
            "bw_details": bw_details,
        }

        return obs, reward, terminated, truncated, info

    def _decode_action(self, action: np.ndarray) -> np.ndarray:
        placement = np.zeros((self.m_max, self.c_max), dtype=np.float32)
        for m in range(self.m_max):
            if self.current_sfcs.cnf_active[m]:
                node_idx = int(action[m])
                placement[m, node_idx] = 1.0
        return placement

    def build_action_mask(self) -> np.ndarray:
        mask = np.zeros((self.m_max, self.c_max), dtype=int)
        for m in range(self.m_max):
            if not self.current_sfcs.cnf_active[m]:
                # Inactive CNF slots assigned to node 0 mask
                mask[m, 0] = 1
                continue

            cnf_c = self.current_sfcs.cnf_cpu[m]
            cnf_r = self.current_sfcs.cnf_ram[m]
            cnf_s = self.current_sfcs.cnf_storage[m]

            for i in range(self.c_max):
                if not self.current_state.node_active[i]:
                    continue

                if (
                    self.current_state.node_cpu[i] >= cnf_c
                    and self.current_state.node_ram[i] >= cnf_r
                    and self.current_state.node_storage[i] >= cnf_s
                ):
                    mask[m, i] = 1
            if mask[m].sum() == 0:
                mask[m, 0] = 1  # Fallback to avoid all-zero mask
        return mask

    def _check_capacity_constraints(self, placement: np.ndarray) -> tuple[bool, dict]:
        node_cpu_demand = np.sum(placement * self.current_sfcs.cnf_cpu[:, None], axis=0)
        node_ram_demand = np.sum(placement * self.current_sfcs.cnf_ram[:, None], axis=0)
        node_stor_demand = np.sum(placement * self.current_sfcs.cnf_storage[:, None], axis=0)

        cpu_ok = np.all(node_cpu_demand <= self.current_state.node_cpu + 1e-5)
        ram_ok = np.all(node_ram_demand <= self.current_state.node_ram + 1e-5)
        stor_ok = np.all(node_stor_demand <= self.current_state.node_storage + 1e-5)

        feasible = bool(cpu_ok and ram_ok and stor_ok)
        details = {
            "cpu_violations": int(np.sum(node_cpu_demand > self.current_state.node_cpu + 1e-5)),
            "ram_violations": int(np.sum(node_ram_demand > self.current_state.node_ram + 1e-5)),
            "stor_violations": int(np.sum(node_stor_demand > self.current_state.node_storage + 1e-5)),
        }
        return feasible, details

    def _check_bw_constraints(self, placement: np.ndarray) -> tuple[bool, dict]:
        link_flow = np.zeros((self.c_max, self.c_max), dtype=np.float32)
        active_sfcs = np.where(self.current_sfcs.sfc_active)[0]

        for sid in active_sfcs:
            cnf_indices = np.where((self.current_sfcs.sfc_id == sid) & self.current_sfcs.cnf_active)[0]
            if len(cnf_indices) < 2:
                continue

            for k in range(len(cnf_indices) - 1):
                u_cnf = cnf_indices[k]
                v_cnf = cnf_indices[k + 1]

                node_u = int(np.argmax(placement[u_cnf]))
                node_v = int(np.argmax(placement[v_cnf]))
                rate = self.current_sfcs.cnf_rate[u_cnf]

                if node_u != node_v:
                    if (
                        node_u >= self._dist_matrix.shape[0]
                        or node_v >= self._dist_matrix.shape[0]
                        or np.isinf(self._dist_matrix[node_u, node_v])
                    ):
                        return False, {"bw_violations": 1, "no_path": True}

                    # Reconstruct path using predecessor matrix
                    curr = node_v
                    path = [curr]
                    while curr != node_u:
                        curr = self._pred_matrix[node_u, curr]
                        if curr == -9999 or curr < 0:  # Invalid predecessor
                            return False, {"bw_violations": 1, "no_path": True}
                        path.append(curr)
                    path.reverse()

                    for p in range(len(path) - 1):
                        link_flow[path[p], path[p + 1]] += rate
                        link_flow[path[p + 1], path[p]] += rate

        bw_ok = np.all(link_flow <= self.current_state.edge_bw + 1e-5)
        return bool(bw_ok), {"bw_violations": int(np.sum(link_flow > self.current_state.edge_bw + 1e-5))}

    def _compute_deployment_cost(self, placement: np.ndarray) -> float:
        cost_map = {0: self.cfg["cost_per_cpu"]["edge"], 1: self.cfg["cost_per_cpu"]["fog"], 2: self.cfg["cost_per_cpu"]["cloud"]}
        node_costs = np.array([cost_map[int(self.current_state.node_layer[i])] for i in range(self.c_max)], dtype=np.float32)

        cnf_cpus = self.current_sfcs.cnf_cpu * self.current_sfcs.cnf_active
        total_cost = float(np.sum(placement * cnf_cpus[:, None] * node_costs[None, :]))
        return total_cost

    def _compute_latency_penalty(self, placement: np.ndarray) -> tuple[float, dict[int, float]]:
        latencies = {}
        penalty = 0.0
        active_sfcs = np.where(self.current_sfcs.sfc_active)[0]

        for sid in active_sfcs:
            cnf_indices = np.where((self.current_sfcs.sfc_id == sid) & self.current_sfcs.cnf_active)[0]
            if len(cnf_indices) == 0:
                continue

            proc_delay = float(np.sum(self.current_sfcs.cnf_proc_delay[cnf_indices]))
            trans_delay = 0.0

            for k in range(len(cnf_indices) - 1):
                u_cnf = cnf_indices[k]
                v_cnf = cnf_indices[k + 1]
                node_u = int(np.argmax(placement[u_cnf]))
                node_v = int(np.argmax(placement[v_cnf]))

                if node_u != node_v:
                    if (
                        node_u >= self._dist_matrix.shape[0]
                        or node_v >= self._dist_matrix.shape[0]
                        or np.isinf(self._dist_matrix[node_u, node_v])
                    ):
                        trans_delay += 500.0  # Path failure penalty
                    else:
                        trans_delay += float(self._dist_matrix[node_u, node_v])

            total_delay = proc_delay + trans_delay
            latencies[int(sid)] = total_delay

            budget = self.current_sfcs.sfc_delay_budget[sid]
            if total_delay > budget:
                penalty += float(total_delay - budget)

        return penalty, latencies

    def _compute_reward(self, cost: float, latency_penalty: float, feasible: bool) -> float:
        r = -cost - self.cfg["alpha"] * latency_penalty
        if not feasible:
            r -= self.cfg["beta"]
        return r

    def _build_obs(self) -> dict:
        edge_attr_mat = np.zeros((self.c_max * self.c_max, 3), dtype=np.float32)
        if len(self.current_state.edge_attr) > 0:
            src = self.current_state.edge_index[0]
            dst = self.current_state.edge_index[1]
            flat_indices = src * self.c_max + dst
            edge_attr_mat[flat_indices] = self.current_state.edge_attr

        return {
            "node_features": self.current_state.node_features,
            "edge_attr": edge_attr_mat,
            "node_history": self.state_buffer.get_history(),
            "cnf_features": self.current_sfcs.cnf_features,
            "action_mask": self.build_action_mask(),
        }
