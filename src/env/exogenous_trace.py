import dataclasses
from dataclasses import dataclass
import numpy as np
import networkx as nx


@dataclass
class ExogenousTrace:
    """
    Deterministic exogenous event trace for a single environment episode of length T.
    Ensures every solver algorithm experiences the exact same external workload arrivals,
    resource demands, TTLs, OU noise processes, and node/link failures.
    """
    seed: int
    max_steps: int
    n_nodes: int
    node_layers: np.ndarray
    cpu_max: np.ndarray
    ram_max: np.ndarray
    storage_max: np.ndarray
    bw_max: np.ndarray
    latency_matrix: np.ndarray
    edge_active_matrix: np.ndarray

    ou_noise_cpu: np.ndarray         # (T + 1, n_nodes)
    ou_noise_bw: np.ndarray          # (T + 1, n_nodes, n_nodes)
    node_failures: dict[int, np.ndarray]    # step -> (n_nodes,) bool array
    link_degradations: dict[int, np.ndarray]# step -> (n_nodes, n_nodes) float multipliers
    sfc_arrivals: dict[int, list[dict]]     # step -> list of SFC spec dicts


class ExogenousTraceGenerator:
    """
    Pre-computes or generates deterministic ExogenousTrace objects for a given seed and env config.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def generate(self, seed: int, max_steps: int = 100, scenario_override: dict | None = None) -> ExogenousTrace:
        rng = np.random.default_rng(seed)

        c_range = self.cfg.get("c_range", [20, 50])
        n_nodes = int(rng.integers(c_range[0], c_range[1] + 1))

        # Waxman topology
        g = nx.waxman_graph(
            n_nodes,
            alpha=self.cfg.get("waxman_alpha", 0.5),
            beta=self.cfg.get("waxman_beta", 0.5),
            seed=seed,
        )
        if not nx.is_connected(g):
            components = list(nx.connected_components(g))
            for i in range(len(components) - 1):
                u = list(components[i])[0]
                v = list(components[i + 1])[0]
                g.add_edge(u, v)

        probs = self.cfg.get("layer_probs", [0.4, 0.3, 0.3])
        node_layers = rng.choice([0, 1, 2], size=n_nodes, p=probs)

        cpu_range = self.cfg.get("cpu_range", [20, 60])
        ram_range = self.cfg.get("ram_range", [32, 128])
        storage_range = self.cfg.get("storage_range", [100, 1000])

        cpu_max = rng.uniform(cpu_range[0], cpu_range[1], size=n_nodes)
        ram_max = rng.uniform(ram_range[0], ram_range[1], size=n_nodes)
        storage_max = rng.uniform(storage_range[0], storage_range[1], size=n_nodes)

        bw_range = self.cfg.get("bw_range", [2500, 10000])
        lat_range = self.cfg.get("latency_range", [1, 50])

        edge_bw_mat = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        edge_lat_mat = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        edge_act_mat = np.zeros((n_nodes, n_nodes), dtype=bool)

        for u, v in g.edges():
            bw = rng.uniform(bw_range[0], bw_range[1])
            lat = rng.uniform(lat_range[0], lat_range[1])
            edge_bw_mat[u, v] = edge_bw_mat[v, u] = bw
            edge_lat_mat[u, v] = edge_lat_mat[v, u] = lat
            edge_act_mat[u, v] = edge_act_mat[v, u] = True

        # Pre-compute OU noise sequence
        theta = self.cfg.get("ou_theta", 0.15)
        sigma = self.cfg.get("ou_sigma", 0.05)
        dt = self.cfg.get("ou_dt", 1.0)

        ou_cpu = np.zeros((max_steps + 1, n_nodes), dtype=np.float32)
        ou_bw = np.zeros((max_steps + 1, n_nodes, n_nodes), dtype=np.float32)

        ou_state_cpu = cpu_max.copy()
        ou_state_bw = edge_bw_mat.copy()

        ou_cpu[0] = ou_state_cpu.copy()
        ou_bw[0] = ou_state_bw.copy()

        for t in range(1, max_steps + 1):
            dx_cpu = theta * (cpu_max - ou_state_cpu) * dt + sigma * rng.normal(size=n_nodes)
            ou_state_cpu = np.clip(ou_state_cpu + dx_cpu, 0.1 * cpu_max, cpu_max)
            ou_cpu[t] = ou_state_cpu.copy()

            dx_bw = theta * (edge_bw_mat - ou_state_bw) * dt + sigma * rng.normal(size=(n_nodes, n_nodes))
            ou_state_bw = np.clip(ou_state_bw + dx_bw, 0.1 * edge_bw_mat, edge_bw_mat)
            ou_state_bw = np.tril(ou_state_bw) + np.tril(ou_state_bw, -1).T
            ou_bw[t] = ou_state_bw.copy()

        # Node failure events
        p_fail = self.cfg.get("node_failure_prob", 0.01)
        if scenario_override and "node_failure_prob" in scenario_override:
            p_fail = scenario_override["node_failure_prob"]

        node_failures = {}
        curr_active = np.ones(n_nodes, dtype=bool)
        for t in range(max_steps + 1):
            if scenario_override and scenario_override.get("scenario") == "F_recovery":
                # Scenario F: node 0 drops at t=20, recovers at t=50
                if t >= 20 and t < 50:
                    curr_active[0] = False
                else:
                    curr_active[0] = True
            else:
                for i in range(n_nodes):
                    if rng.random() < p_fail:
                        curr_active[i] = not curr_active[i]
            node_failures[t] = curr_active.copy()

        # Link degradation events
        link_degradations = {}
        for t in range(max_steps + 1):
            deg = np.ones((n_nodes, n_nodes), dtype=np.float32)
            if scenario_override and scenario_override.get("scenario") == "D_link_degradation":
                if t >= 20 and t <= 70:
                    deg *= 0.5  # 50% degradation
            link_degradations[t] = deg

        # SFC Arrival generation across timesteps
        sfc_ttl_range = self.cfg.get("sfc_ttl_range", [10, 40])
        h_range = self.cfg.get("h_range", [5, 10])

        sfc_arrivals = {}
        next_sfc_id = 0

        # Initial batch at t=0
        n_initial = rng.integers(h_range[0], h_range[1] + 1)
        initial_sfcs = []
        for _ in range(n_initial):
            sfc_spec = self._sample_single_sfc(rng, next_sfc_id, sfc_ttl_range, scenario_override, t=0)
            initial_sfcs.append(sfc_spec)
            next_sfc_id += 1
        sfc_arrivals[0] = initial_sfcs

        # Subsequent step arrivals
        arrival_prob = self.cfg.get("sfc_arrival_prob", 0.30)
        if scenario_override and "sfc_arrival_prob" in scenario_override:
            arrival_prob = scenario_override["sfc_arrival_prob"]

        for t in range(1, max_steps + 1):
            step_arrivals = []
            cur_prob = arrival_prob
            if scenario_override and scenario_override.get("scenario") == "B_load_burst" and 30 <= t <= 60:
                cur_prob = min(1.0, arrival_prob * 2.5)

            if rng.random() < cur_prob:
                num_new = rng.integers(1, 4)
                for _ in range(num_new):
                    sfc_spec = self._sample_single_sfc(rng, next_sfc_id, sfc_ttl_range, scenario_override, t=t)
                    step_arrivals.append(sfc_spec)
                    next_sfc_id += 1
            sfc_arrivals[t] = step_arrivals

        return ExogenousTrace(
            seed=seed,
            max_steps=max_steps,
            n_nodes=n_nodes,
            node_layers=node_layers,
            cpu_max=cpu_max,
            ram_max=ram_max,
            storage_max=storage_max,
            bw_max=edge_bw_mat,
            latency_matrix=edge_lat_mat,
            edge_active_matrix=edge_act_mat,
            ou_noise_cpu=ou_cpu,
            ou_noise_bw=ou_bw,
            node_failures=node_failures,
            link_degradations=link_degradations,
            sfc_arrivals=sfc_arrivals,
        )

    def _sample_single_sfc(self, rng: np.random.Generator, sfc_id: int, sfc_ttl_range: list, scenario_override: dict | None, t: int) -> dict:
        chain_len = int(rng.integers(2, 9))
        budget = float(rng.uniform(
            self.cfg.get("sfc_delay_budget_range", [120, 350])[0],
            self.cfg.get("sfc_delay_budget_range", [120, 350])[1]
        ))
        ttl = int(rng.integers(sfc_ttl_range[0], sfc_ttl_range[1] + 1))

        cnf_cpu_range = self.cfg.get("cnf_cpu_range", [0.5, 3.0])
        cnf_ram_range = self.cfg.get("cnf_ram_range", [0.5, 6.0])
        cnf_storage_range = self.cfg.get("cnf_storage_range", [1, 20])
        cnf_rate_range = self.cfg.get("cnf_rate_range", [10, 200])
        cnf_proc_delay_range = self.cfg.get("cnf_proc_delay_range", [0.1, 2.0])

        mult = 1.0
        if scenario_override and scenario_override.get("scenario") == "B_load_burst" and 30 <= t <= 60:
            mult = 1.8

        cnfs = []
        for pos in range(chain_len):
            cnfs.append({
                "cpu": float(rng.uniform(cnf_cpu_range[0], cnf_cpu_range[1]) * mult),
                "ram": float(rng.uniform(cnf_ram_range[0], cnf_ram_range[1]) * mult),
                "storage": float(rng.uniform(cnf_storage_range[0], cnf_storage_range[1]) * mult),
                "rate": float(rng.uniform(cnf_rate_range[0], cnf_rate_range[1]) * mult),
                "proc_delay": float(rng.uniform(cnf_proc_delay_range[0], cnf_proc_delay_range[1])),
            })

        return {
            "sfc_id": sfc_id,
            "ttl": ttl,
            "delay_budget": budget,
            "cnfs": cnfs,
        }
