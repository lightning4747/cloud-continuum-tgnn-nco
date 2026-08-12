import dataclasses
from dataclasses import dataclass
import numpy as np
import networkx as nx


@dataclass
class NetworkState:
    node_features: np.ndarray    # (C_max, F_node)
    node_cpu: np.ndarray         # (C_max,)
    node_ram: np.ndarray         # (C_max,)
    node_storage: np.ndarray     # (C_max,)
    node_layer: np.ndarray       # (C_max,)
    node_active: np.ndarray      # (C_max,)

    edge_bw: np.ndarray          # (C_max, C_max)
    edge_latency: np.ndarray     # (C_max, C_max)
    edge_active: np.ndarray      # (C_max, C_max)

    edge_index: np.ndarray       # (2, E)
    edge_attr: np.ndarray        # (E, F_edge)

    n_active_nodes: int
    timestep: int


@dataclass
class SFCBatch:
    cnf_cpu: np.ndarray          # (M_max,)
    cnf_ram: np.ndarray          # (M_max,)
    cnf_storage: np.ndarray      # (M_max,)
    cnf_rate: np.ndarray         # (M_max,)
    cnf_proc_delay: np.ndarray   # (M_max,)
    cnf_features: np.ndarray     # (M_max, F_cnf)

    sfc_id: np.ndarray           # (M_max,)
    sfc_position: np.ndarray     # (M_max,)
    sfc_delay_budget: np.ndarray # (H_max,)
    sfc_active: np.ndarray       # (H_max,)

    cnf_active: np.ndarray       # (M_max,)
    n_active_cnfs: int
    n_active_sfcs: int


class TopologyGenerator:
    """
    Generates dynamic cloud-continuum networks (Waxman topology) and SFC service batches.
    Handles dynamic node resource fluctuations (OU process), failures, and SFC arrivals.
    """

    def __init__(self, cfg: dict, seed: int = 42):
        self.cfg = cfg
        self.c_max = cfg["c_max"]
        self.m_max = cfg["m_max"]
        self.h_max = cfg["h_max"]
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.timestep = 0

    def reset(self, seed: int | None = None) -> tuple[NetworkState, SFCBatch]:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        self.timestep = 0

        # 1. Sample Active Infrastructure Scale
        c_range = self.cfg["c_range"]
        self.n_nodes = int(self.rng.integers(c_range[0], c_range[1] + 1))

        # 2. Generate Waxman Topology Graph
        g = nx.waxman_graph(
            self.n_nodes,
            alpha=self.cfg["waxman_alpha"],
            beta=self.cfg["waxman_beta"],
            seed=self.seed,
        )
        if not nx.is_connected(g):
            # Connect components to ensure valid routing paths
            components = list(nx.connected_components(g))
            for i in range(len(components) - 1):
                u = list(components[i])[0]
                v = list(components[i + 1])[0]
                g.add_edge(u, v)

        # 3. Sample Node Layer & Resource Capacities
        probs = self.cfg["layer_probs"]
        layers = self.rng.choice([0, 1, 2], size=self.n_nodes, p=probs)
        self.node_layers = layers

        cpu_max = self.rng.uniform(self.cfg["cpu_range"][0], self.cfg["cpu_range"][1], size=self.n_nodes)
        ram_max = self.rng.uniform(self.cfg["ram_range"][0], self.cfg["ram_range"][1], size=self.n_nodes)
        storage_max = self.rng.uniform(self.cfg["storage_range"][0], self.cfg["storage_range"][1], size=self.n_nodes)

        self.cpu_max = cpu_max.copy()
        self.ou_state_cpu = cpu_max.copy()

        # Link Resources
        edge_bw_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        edge_lat_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        edge_act_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=bool)

        for u, v in g.edges():
            bw = self.rng.uniform(self.cfg["bw_range"][0], self.cfg["bw_range"][1])
            lat = self.rng.uniform(self.cfg["latency_range"][0], self.cfg["latency_range"][1])
            edge_bw_mat[u, v] = edge_bw_mat[v, u] = bw
            edge_lat_mat[u, v] = edge_lat_mat[v, u] = lat
            edge_act_mat[u, v] = edge_act_mat[v, u] = True

        self.bw_max = edge_bw_mat.copy()
        self.ou_state_bw = edge_bw_mat.copy()

        # Build initial NetworkState
        state = self._build_network_state(
            cpu=self.ou_state_cpu,
            ram=ram_max.copy(),
            storage=storage_max.copy(),
            layers=layers,
            edge_bw=self.ou_state_bw,
            edge_lat=edge_lat_mat,
            edge_act=edge_act_mat,
        )

        # 4. Generate Initial SFCs
        n_sfcs = self.rng.integers(self.cfg["h_range"][0], self.cfg["h_range"][1] + 1)
        sfcs = self._sample_sfcs(n_sfcs)
        sfc_batch = self._pack_sfcs(sfcs)

        return state, sfc_batch

    def step(self, current_state: NetworkState, current_sfcs: SFCBatch) -> tuple[NetworkState, SFCBatch]:
        self.timestep += 1

        # 1. Update OU process for node CPU and link BW
        theta = self.cfg["ou_theta"]
        sigma = self.cfg["ou_sigma"]
        dt = self.cfg["ou_dt"]

        # CPU noise
        dx_cpu = theta * (self.cpu_max - self.ou_state_cpu) * dt + sigma * self.rng.normal(size=self.n_nodes)
        self.ou_state_cpu = np.clip(self.ou_state_cpu + dx_cpu, 0.1 * self.cpu_max, self.cpu_max)

        # Link BW noise
        dx_bw = theta * (self.bw_max - self.ou_state_bw) * dt + sigma * self.rng.normal(size=(self.n_nodes, self.n_nodes))
        self.ou_state_bw = np.clip(self.ou_state_bw + dx_bw, 0.1 * self.bw_max, self.bw_max)
        self.ou_state_bw = np.tril(self.ou_state_bw) + np.tril(self.ou_state_bw, -1).T

        # Build active node mask (random failures with prob 0.01)
        node_active = current_state.node_active[:self.n_nodes].copy()
        for i in range(self.n_nodes):
            if self.rng.random() < 0.01:
                node_active[i] = not node_active[i]

        state = self._build_network_state(
            cpu=self.ou_state_cpu,
            ram=current_state.node_ram[:self.n_nodes],
            storage=current_state.node_storage[:self.n_nodes],
            layers=self.node_layers,
            edge_bw=self.ou_state_bw,
            edge_lat=current_state.edge_latency[:self.n_nodes, :self.n_nodes],
            edge_act=current_state.edge_active[:self.n_nodes, :self.n_nodes],
            node_active_override=node_active,
        )

        # 2. SFC Lifecycle
        active_sfc_ids = np.where(current_sfcs.sfc_active)[0]
        n_retire = 0
        for sid in active_sfc_ids:
            if self.rng.random() < 0.10 and len(active_sfc_ids) - n_retire > self.cfg["h_range"][0]:
                n_retire += 1

        target_n_sfcs = self.rng.integers(self.cfg["h_range"][0], self.cfg["h_range"][1] + 1)
        new_sfcs = self._sample_sfcs(target_n_sfcs)
        sfc_batch = self._pack_sfcs(new_sfcs)

        return state, sfc_batch

    def _build_network_state(
        self,
        cpu: np.ndarray,
        ram: np.ndarray,
        storage: np.ndarray,
        layers: np.ndarray,
        edge_bw: np.ndarray,
        edge_lat: np.ndarray,
        edge_act: np.ndarray,
        node_active_override: np.ndarray | None = None,
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

        pad_cpu[:self.n_nodes] = cpu
        pad_ram[:self.n_nodes] = ram
        pad_storage[:self.n_nodes] = storage
        pad_layer[:self.n_nodes] = layers

        # Normalized features (6 dims): [cpu_norm, ram_norm, storage_norm, layer_oh0, layer_oh1, layer_oh2]
        node_feats = np.zeros((c_eff, 6), dtype=np.float32)
        node_feats[:self.n_nodes, 0] = cpu / self.cfg["cpu_range"][1]
        node_feats[:self.n_nodes, 1] = ram / self.cfg["ram_range"][1]
        node_feats[:self.n_nodes, 2] = storage / self.cfg["storage_range"][1]

        for i in range(self.n_nodes):
            l = layers[i]
            node_feats[i, 3 + l] = 1.0

        # Padded Edge Tensors
        pad_bw = np.zeros((c_eff, c_eff), dtype=np.float32)
        pad_lat = np.zeros((c_eff, c_eff), dtype=np.float32)
        pad_edge_act = np.zeros((c_eff, c_eff), dtype=bool)

        pad_bw[:self.n_nodes, :self.n_nodes] = edge_bw
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
        )

    def _sample_sfcs(self, n_sfcs: int) -> list[dict]:
        sfcs = []
        for sid in range(n_sfcs):
            chain_len = int(self.rng.integers(2, 9))
            budget = float(self.rng.uniform(self.cfg["sfc_delay_budget_range"][0], self.cfg["sfc_delay_budget_range"][1]))

            cnfs = []
            for pos in range(chain_len):
                cnfs.append({
                    "cpu": float(self.rng.uniform(self.cfg["cnf_cpu_range"][0], self.cfg["cnf_cpu_range"][1])),
                    "ram": float(self.rng.uniform(self.cfg["cnf_ram_range"][0], self.cfg["cnf_ram_range"][1])),
                    "storage": float(self.rng.uniform(self.cfg["cnf_storage_range"][0], self.cfg["cnf_storage_range"][1])),
                    "rate": float(self.rng.uniform(self.cfg["cnf_rate_range"][0], self.cfg["cnf_rate_range"][1])),
                    "proc_delay": float(self.rng.uniform(self.cfg["cnf_proc_delay_range"][0], self.cfg["cnf_proc_delay_range"][1])),
                })

            sfcs.append({
                "sfc_id": sid,
                "delay_budget": budget,
                "cnfs": cnfs,
            })
        return sfcs

    def _pack_sfcs(self, sfcs: list[dict]) -> SFCBatch:
        total_cnfs = sum(len(s["cnfs"]) for s in sfcs)
        m_eff = max(self.m_max, total_cnfs)
        h_eff = max(self.h_max, len(sfcs))

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
        for s in sfcs:
            sid = s["sfc_id"]
            if sid < h_eff:
                sfc_delay_budget[sid] = s["delay_budget"]
                sfc_active[sid] = True

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
            n_active_sfcs=len(sfcs),
        )
