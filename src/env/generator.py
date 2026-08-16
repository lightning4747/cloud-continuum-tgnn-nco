import dataclasses
from dataclasses import dataclass
import numpy as np
import networkx as nx

from src.env.exogenous_trace import ExogenousTrace


@dataclass
class NetworkState:
    node_features: np.ndarray          # (C_max, F_node)
    node_cpu: np.ndarray               # (C_max,) - Available CPU
    node_ram: np.ndarray               # (C_max,) - Available RAM
    node_storage: np.ndarray           # (C_max,) - Available Storage
    node_layer: np.ndarray             # (C_max,)
    node_active: np.ndarray            # (C_max,)

    edge_bw: np.ndarray                # (C_max, C_max) - Available Bandwidth
    edge_latency: np.ndarray           # (C_max, C_max)
    edge_active: np.ndarray            # (C_max, C_max)

    edge_index: np.ndarray             # (2, E)
    edge_attr: np.ndarray              # (E, F_edge)

    n_active_nodes: int
    timestep: int

    # Capacity totals (unallocated max capacity) for metrics tracking
    node_cpu_total: np.ndarray = None
    node_ram_total: np.ndarray = None
    node_storage_total: np.ndarray = None
    edge_bw_total: np.ndarray = None


@dataclass
class SFCBatch:
    cnf_cpu: np.ndarray                # (M_max,)
    cnf_ram: np.ndarray                # (M_max,)
    cnf_storage: np.ndarray            # (M_max,)
    cnf_rate: np.ndarray               # (M_max,)
    cnf_proc_delay: np.ndarray         # (M_max,)
    cnf_features: np.ndarray           # (M_max, F_cnf)

    sfc_id: np.ndarray                 # (M_max,)
    sfc_position: np.ndarray           # (M_max,)
    sfc_delay_budget: np.ndarray       # (H_max,)
    sfc_active: np.ndarray             # (H_max,)

    cnf_active: np.ndarray             # (M_max,)
    n_active_cnfs: int
    n_active_sfcs: int


class TopologyGenerator:
    """
    Generates dynamic cloud-continuum networks and persistent SFC workloads.
    Supports deterministic ExogenousTrace objects for exact benchmark reproducibility across solvers.
    """

    def __init__(self, cfg: dict, seed: int = 42):
        self.cfg = cfg
        self.c_max = cfg["c_max"]
        self.m_max = cfg["m_max"]
        self.h_max = cfg["h_max"]
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.timestep = 0
        self.exogenous_trace: ExogenousTrace | None = None
        self.active_sfcs: dict[int, dict] = {}
        self.next_sfc_id = 0

    def reset(self, seed: int | None = None, exogenous_trace: ExogenousTrace | None = None) -> tuple[NetworkState, SFCBatch]:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        self.timestep = 0
        self.exogenous_trace = exogenous_trace
        self.active_sfcs = {}
        self.next_sfc_id = 0

        if self.exogenous_trace is not None:
            self.n_nodes = self.exogenous_trace.n_nodes
            self.node_layers = self.exogenous_trace.node_layers
            self.cpu_max = self.exogenous_trace.cpu_max.copy()
            self.ram_max = self.exogenous_trace.ram_max.copy()
            self.storage_max = self.exogenous_trace.storage_max.copy()
            self.bw_max = self.exogenous_trace.bw_max.copy()
            self.lat_matrix = self.exogenous_trace.latency_matrix.copy()
            self.edge_act_mat = self.exogenous_trace.edge_active_matrix.copy()

            self.ou_state_cpu = self.exogenous_trace.ou_noise_cpu[0].copy()
            self.ou_state_bw = self.exogenous_trace.ou_noise_bw[0].copy()
            node_active_override = self.exogenous_trace.node_failures[0]

            initial_sfc_specs = self.exogenous_trace.sfc_arrivals.get(0, [])
        else:
            c_range = self.cfg.get("c_range", [20, 50])
            self.n_nodes = int(self.rng.integers(c_range[0], c_range[1] + 1))

            g = nx.waxman_graph(
                self.n_nodes,
                alpha=self.cfg.get("waxman_alpha", 0.5),
                beta=self.cfg.get("waxman_beta", 0.5),
                seed=self.seed,
            )
            if not nx.is_connected(g):
                components = list(nx.connected_components(g))
                for i in range(len(components) - 1):
                    u = list(components[i])[0]
                    v = list(components[i + 1])[0]
                    g.add_edge(u, v)

            probs = self.cfg.get("layer_probs", [0.4, 0.3, 0.3])
            self.node_layers = self.rng.choice([0, 1, 2], size=self.n_nodes, p=probs)

            cpu_range = self.cfg.get("cpu_range", [20, 60])
            ram_range = self.cfg.get("ram_range", [32, 128])
            storage_range = self.cfg.get("storage_range", [100, 1000])

            self.cpu_max = self.rng.uniform(cpu_range[0], cpu_range[1], size=self.n_nodes)
            self.ram_max = self.rng.uniform(ram_range[0], ram_range[1], size=self.n_nodes)
            self.storage_max = self.rng.uniform(storage_range[0], storage_range[1], size=self.n_nodes)

            self.ou_state_cpu = self.cpu_max.copy()

            bw_range = self.cfg.get("bw_range", [2500, 10000])
            lat_range = self.cfg.get("latency_range", [1, 50])

            edge_bw_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
            edge_lat_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
            edge_act_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=bool)

            for u, v in g.edges():
                bw = self.rng.uniform(bw_range[0], bw_range[1])
                lat = self.rng.uniform(lat_range[0], lat_range[1])
                edge_bw_mat[u, v] = edge_bw_mat[v, u] = bw
                edge_lat_mat[u, v] = edge_lat_mat[v, u] = lat
                edge_act_mat[u, v] = edge_act_mat[v, u] = True

            self.bw_max = edge_bw_mat.copy()
            self.ou_state_bw = edge_bw_mat.copy()
            self.lat_matrix = edge_lat_mat
            self.edge_act_mat = edge_act_mat
            node_active_override = np.ones(self.n_nodes, dtype=bool)

            n_initial = self.rng.integers(self.cfg["h_range"][0], self.cfg["h_range"][1] + 1)
            initial_sfc_specs = [self._sample_sfc_spec(self.next_sfc_id + i) for i in range(n_initial)]

        # Register initial SFCs
        for sfc_spec in initial_sfc_specs:
            sid = sfc_spec["sfc_id"]
            self.active_sfcs[sid] = {
                "sfc_id": sid,
                "ttl": sfc_spec["ttl"],
                "delay_budget": sfc_spec["delay_budget"],
                "cnfs": sfc_spec["cnfs"],
                "placed": False,
                "placements": None,
                "path_allocations": None,
            }
            self.next_sfc_id = max(self.next_sfc_id, sid + 1)

        state = self._build_network_state(
            cpu_avail=self.ou_state_cpu,
            ram_avail=self.ram_max.copy(),
            storage_avail=self.storage_max.copy(),
            layers=self.node_layers,
            edge_bw_avail=self.ou_state_bw,
            edge_lat=self.lat_matrix,
            edge_act=self.edge_act_mat,
            node_active_override=node_active_override,
            cpu_total=self.ou_state_cpu.copy(),
            ram_total=self.ram_max.copy(),
            storage_total=self.storage_max.copy(),
            bw_total=self.ou_state_bw.copy(),
        )

        sfc_batch = self._pack_active_sfcs()
        return state, sfc_batch

    def step(
        self,
        current_state: NetworkState,
        node_cpu_allocated: np.ndarray,
        node_ram_allocated: np.ndarray,
        node_storage_allocated: np.ndarray,
        edge_bw_allocated: np.ndarray,
    ) -> tuple[NetworkState, SFCBatch, list[dict]]:
        self.timestep += 1

        # 1. Environment Exogenous Process Updates
        if self.exogenous_trace is not None:
            t_idx = min(self.timestep, self.exogenous_trace.max_steps)
            self.ou_state_cpu = self.exogenous_trace.ou_noise_cpu[t_idx].copy()
            self.ou_state_bw = self.exogenous_trace.ou_noise_bw[t_idx].copy()

            if self.exogenous_trace.link_degradations and t_idx in self.exogenous_trace.link_degradations:
                deg = self.exogenous_trace.link_degradations[t_idx]
                self.ou_state_bw[:self.n_nodes, :self.n_nodes] *= deg

            node_active = self.exogenous_trace.node_failures.get(t_idx, np.ones(self.n_nodes, dtype=bool))
            new_arrivals = self.exogenous_trace.sfc_arrivals.get(t_idx, [])
        else:
            theta = self.cfg.get("ou_theta", 0.15)
            sigma = self.cfg.get("ou_sigma", 0.05)
            dt = self.cfg.get("ou_dt", 1.0)

            dx_cpu = theta * (self.cpu_max - self.ou_state_cpu) * dt + sigma * self.rng.normal(size=self.n_nodes)
            self.ou_state_cpu = np.clip(self.ou_state_cpu + dx_cpu, 0.1 * self.cpu_max, self.cpu_max)

            dx_bw = theta * (self.bw_max - self.ou_state_bw) * dt + sigma * self.rng.normal(size=(self.n_nodes, self.n_nodes))
            self.ou_state_bw = np.clip(self.ou_state_bw + dx_bw, 0.1 * self.bw_max, self.bw_max)
            self.ou_state_bw = np.tril(self.ou_state_bw) + np.tril(self.ou_state_bw, -1).T

            p_fail = self.cfg.get("node_failure_prob", 0.01)
            node_active = current_state.node_active[:self.n_nodes].copy()
            for i in range(self.n_nodes):
                if self.rng.random() < p_fail:
                    node_active[i] = not node_active[i]

            new_arrivals = []
            arrival_prob = self.cfg.get("sfc_arrival_prob", 0.30)
            if self.rng.random() < arrival_prob:
                num_new = self.rng.integers(1, 3)
                for _ in range(num_new):
                    new_arrivals.append(self._sample_sfc_spec(self.next_sfc_id))
                    self.next_sfc_id += 1

        # 2. SFC Lifecycle & Retirement (TTL Decrement)
        retired_sfcs = []
        active_ids = list(self.active_sfcs.keys())
        for sid in active_ids:
            sfc = self.active_sfcs[sid]
            sfc["ttl"] -= 1
            if sfc["ttl"] <= 0:
                retired_sfcs.append(sfc)
                del self.active_sfcs[sid]

        # Register new arrivals
        for sfc_spec in new_arrivals:
            sid = sfc_spec["sfc_id"]
            if sid not in self.active_sfcs:
                self.active_sfcs[sid] = {
                    "sfc_id": sid,
                    "ttl": sfc_spec["ttl"],
                    "delay_budget": sfc_spec["delay_budget"],
                    "cnfs": sfc_spec["cnfs"],
                    "placed": False,
                    "placements": None,
                    "path_allocations": None,
                }

        # Calculate Available Resources
        cpu_avail = np.maximum(0.0, self.ou_state_cpu - node_cpu_allocated[:self.n_nodes])
        ram_avail = np.maximum(0.0, self.ram_max - node_ram_allocated[:self.n_nodes])
        storage_avail = np.maximum(0.0, self.storage_max - node_storage_allocated[:self.n_nodes])
        bw_avail = np.maximum(0.0, self.ou_state_bw - edge_bw_allocated[:self.n_nodes, :self.n_nodes])

        state = self._build_network_state(
            cpu_avail=cpu_avail,
            ram_avail=ram_avail,
            storage_avail=storage_avail,
            layers=self.node_layers,
            edge_bw_avail=bw_avail,
            edge_lat=self.lat_matrix,
            edge_act=self.edge_act_mat,
            node_active_override=node_active,
            cpu_total=self.ou_state_cpu.copy(),
            ram_total=self.ram_max.copy(),
            storage_total=self.storage_max.copy(),
            bw_total=self.ou_state_bw.copy(),
        )

        sfc_batch = self._pack_active_sfcs()
        return state, sfc_batch, retired_sfcs

    def _sample_sfc_spec(self, sfc_id: int) -> dict:
        chain_len = int(self.rng.integers(2, 9))
        budget = float(self.rng.uniform(
            self.cfg.get("sfc_delay_budget_range", [120, 350])[0],
            self.cfg.get("sfc_delay_budget_range", [120, 350])[1]
        ))
        ttl_range = self.cfg.get("sfc_ttl_range", [10, 40])
        ttl = int(self.rng.integers(ttl_range[0], ttl_range[1] + 1))

        cnfs = []
        for pos in range(chain_len):
            cnfs.append({
                "cpu": float(self.rng.uniform(self.cfg["cnf_cpu_range"][0], self.cfg["cnf_cpu_range"][1])),
                "ram": float(self.rng.uniform(self.cfg["cnf_ram_range"][0], self.cfg["cnf_ram_range"][1])),
                "storage": float(self.rng.uniform(self.cfg["cnf_storage_range"][0], self.cfg["cnf_storage_range"][1])),
                "rate": float(self.rng.uniform(self.cfg["cnf_rate_range"][0], self.cfg["cnf_rate_range"][1])),
                "proc_delay": float(self.rng.uniform(self.cfg["cnf_proc_delay_range"][0], self.cfg["cnf_proc_delay_range"][1])),
            })

        return {
            "sfc_id": sfc_id,
            "ttl": ttl,
            "delay_budget": budget,
            "cnfs": cnfs,
        }

    def _build_network_state(
        self,
        cpu_avail: np.ndarray,
        ram_avail: np.ndarray,
        storage_avail: np.ndarray,
        layers: np.ndarray,
        edge_bw_avail: np.ndarray,
        edge_lat: np.ndarray,
        edge_act: np.ndarray,
        node_active_override: np.ndarray | None = None,
        cpu_total: np.ndarray | None = None,
        ram_total: np.ndarray | None = None,
        storage_total: np.ndarray | None = None,
        bw_total: np.ndarray | None = None,
    ) -> NetworkState:
        c_eff = max(self.c_max, self.n_nodes)

        node_active = np.zeros(c_eff, dtype=bool)
        if node_active_override is None:
            node_active[:self.n_nodes] = True
        else:
            node_active[:self.n_nodes] = node_active_override

        # Padded Node Tensors
        pad_cpu = np.zeros(c_eff, dtype=np.float32)
        pad_ram = np.zeros(c_eff, dtype=np.float32)
        pad_storage = np.zeros(c_eff, dtype=np.float32)
        pad_layer = np.zeros(c_eff, dtype=np.int64)

        pad_cpu[:self.n_nodes] = cpu_avail
        pad_ram[:self.n_nodes] = ram_avail
        pad_storage[:self.n_nodes] = storage_avail
        pad_layer[:self.n_nodes] = layers

        # Normalized features (6 dims): [cpu_avail_norm, ram_avail_norm, storage_avail_norm, layer_oh0, layer_oh1, layer_oh2]
        node_feats = np.zeros((c_eff, 6), dtype=np.float32)
        node_feats[:self.n_nodes, 0] = cpu_avail / self.cfg["cpu_range"][1]
        node_feats[:self.n_nodes, 1] = ram_avail / self.cfg["ram_range"][1]
        node_feats[:self.n_nodes, 2] = storage_avail / self.cfg["storage_range"][1]

        for i in range(self.n_nodes):
            l = layers[i]
            node_feats[i, 3 + l] = 1.0

        # Padded Edge Tensors
        pad_bw = np.zeros((c_eff, c_eff), dtype=np.float32)
        pad_lat = np.zeros((c_eff, c_eff), dtype=np.float32)
        pad_edge_act = np.zeros((c_eff, c_eff), dtype=bool)

        pad_bw[:self.n_nodes, :self.n_nodes] = edge_bw_avail
        pad_lat[:self.n_nodes, :self.n_nodes] = edge_lat
        pad_edge_act[:self.n_nodes, :self.n_nodes] = edge_act

        # PyG Edge Index & Attributes
        src, dst = np.where(edge_act[:self.n_nodes, :self.n_nodes])
        edge_index = np.stack([src, dst], axis=0).astype(np.int64)

        edge_attr_list = []
        for k in range(len(src)):
            u, v = src[k], dst[k]
            bw_norm = pad_bw[u, v] / self.cfg["bw_range"][1]
            lat_norm = pad_lat[u, v] / self.cfg["latency_range"][1]
            act = 1.0 if pad_edge_act[u, v] else 0.0
            edge_attr_list.append([bw_norm, lat_norm, act])

        edge_attr = np.array(edge_attr_list, dtype=np.float32) if edge_attr_list else np.zeros((0, 3), dtype=np.float32)

        # Totals padding
        pad_cpu_tot = np.zeros(c_eff, dtype=np.float32)
        pad_ram_tot = np.zeros(c_eff, dtype=np.float32)
        pad_stor_tot = np.zeros(c_eff, dtype=np.float32)
        pad_bw_tot = np.zeros((c_eff, c_eff), dtype=np.float32)

        if cpu_total is not None:
            pad_cpu_tot[:self.n_nodes] = cpu_total
        if ram_total is not None:
            pad_ram_tot[:self.n_nodes] = ram_total
        if storage_total is not None:
            pad_stor_tot[:self.n_nodes] = storage_total
        if bw_total is not None:
            pad_bw_tot[:self.n_nodes, :self.n_nodes] = bw_total

        return NetworkState(
            node_features=node_feats,
            node_cpu=pad_cpu,
            node_ram=pad_ram,
            node_storage=pad_storage,
            node_layer=pad_layer,
            node_active=node_active,
            edge_bw=pad_bw,
            edge_latency=pad_lat,
            edge_active=pad_edge_act,
            edge_index=edge_index,
            edge_attr=edge_attr,
            n_active_nodes=self.n_nodes,
            timestep=self.timestep,
            node_cpu_total=pad_cpu_tot,
            node_ram_total=pad_ram_tot,
            node_storage_total=pad_stor_tot,
            edge_bw_total=pad_bw_tot,
        )

    def _pack_active_sfcs(self) -> SFCBatch:
        # Pack unplaced/active SFCs into SFCBatch
        active_list = list(self.active_sfcs.values())
        total_cnfs = sum(len(s["cnfs"]) for s in active_list)
        m_eff = max(self.m_max, total_cnfs)
        h_eff = max(self.h_max, len(active_list))

        cnf_cpu = np.zeros(m_eff, dtype=np.float32)
        cnf_ram = np.zeros(m_eff, dtype=np.float32)
        cnf_storage = np.zeros(m_eff, dtype=np.float32)
        cnf_rate = np.zeros(m_eff, dtype=np.float32)
        cnf_proc_delay = np.zeros(m_eff, dtype=np.float32)
        cnf_features = np.zeros((m_eff, 5), dtype=np.float32)
        sfc_id = np.zeros(m_eff, dtype=np.int64)
        sfc_position = np.zeros(m_eff, dtype=np.int64)
        cnf_active = np.zeros(m_eff, dtype=bool)

        sfc_delay_budget = np.zeros(h_eff, dtype=np.float32)
        sfc_active = np.zeros(h_eff, dtype=bool)

        curr_m = 0
        for s_idx, s in enumerate(active_list):
            sid = s["sfc_id"]
            if s_idx < h_eff:
                sfc_delay_budget[s_idx] = s["delay_budget"]
                sfc_active[s_idx] = True

            for pos, cnf in enumerate(s["cnfs"]):
                if curr_m >= m_eff:
                    break
                cnf_cpu[curr_m] = cnf["cpu"]
                cnf_ram[curr_m] = cnf["ram"]
                cnf_storage[curr_m] = cnf["storage"]
                cnf_rate[curr_m] = cnf["rate"]
                cnf_proc_delay[curr_m] = cnf["proc_delay"]
                sfc_id[curr_m] = sid
                sfc_position[curr_m] = pos
                cnf_active[curr_m] = True

                cnf_features[curr_m, 0] = cnf["cpu"] / self.cfg["cnf_cpu_range"][1]
                cnf_features[curr_m, 1] = cnf["ram"] / self.cfg["cnf_ram_range"][1]
                cnf_features[curr_m, 2] = cnf["storage"] / self.cfg["cnf_storage_range"][1]
                cnf_features[curr_m, 3] = cnf["rate"] / self.cfg["cnf_rate_range"][1]
                cnf_features[curr_m, 4] = cnf["proc_delay"] / self.cfg["cnf_proc_delay_range"][1]

                curr_m += 1

        return SFCBatch(
            cnf_cpu=cnf_cpu,
            cnf_ram=cnf_ram,
            cnf_storage=cnf_storage,
            cnf_rate=cnf_rate,
            cnf_proc_delay=cnf_proc_delay,
            cnf_features=cnf_features,
            sfc_id=sfc_id,
            sfc_position=sfc_position,
            sfc_delay_budget=sfc_delay_budget,
            sfc_active=sfc_active,
            cnf_active=cnf_active,
            n_active_cnfs=curr_m,
            n_active_sfcs=len(active_list),
        )
