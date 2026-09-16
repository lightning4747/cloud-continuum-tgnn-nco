"""
State Migration Cost Model — Dirty-Memory Pre-Copy Transfer Overhead.

Grounded in context.md:
    "Moving a containerized CNF to a lower-latency compute node creates transient
     latency spikes and bandwidth saturation if the associated state payload is large."

Mathematical model (NFV pre-copy migration literature):
    T_mig(m) = (dirty_ratio × RAM_m_GB × 1000 MB/GB) / BW_link_Mbps
    penalty_m = alpha_mig × T_mig(m)  [result in ms]

Scope:
    Applies ONLY to CNFs that change nodes between consecutive steps
    (prev_node >= 0 AND prev_node != new_node).
    New SFC arrivals (prev_node == -1) incur zero migration cost.
"""
import numpy as np

DEFAULT_DIRTY_RATIO = 0.20   # 20% dirty-page fraction (moderate NFV workload)
DEFAULT_ALPHA_MIG   = 0.50   # migration penalty scaling coefficient


def compute_migration_cost(
    placement_matrix: np.ndarray,       # (M_max, C_max) one-hot float32
    cnf_prev_node: np.ndarray,          # (M_max,) int32, -1 = new/unplaced
    cnf_active: np.ndarray,             # (M_max,) bool
    cnf_ram: np.ndarray,                # (M_max,) RAM demand in raw GB units
    edge_bw: np.ndarray,                # (C_max, C_max) available BW [0,1] normalized
    ram_range_max: float = 128.0,       # denorm: not used (cnf_ram is raw GB)
    bw_range_max: float = 10000.0,      # denorm: 1.0 → bw_range_max Mbps
    alpha_mig: float = DEFAULT_ALPHA_MIG,
    dirty_ratio: float = DEFAULT_DIRTY_RATIO,
) -> float:
    """
    Compute total migration penalty in milliseconds (equivalent latency units).

    Returns 0.0 if no CNF changes its placement node.

    Args:
        placement_matrix: (M_max, C_max) one-hot — new placement decisions
        cnf_prev_node:    (M_max,) — node index where each CNF was placed last step;
                          -1 means the CNF is newly arrived (no migration cost)
        cnf_active:       (M_max,) bool — which CNF slots are active
        cnf_ram:          (M_max,) raw GB values (e.g. drawn from cnf_ram_range)
        edge_bw:          (C_max, C_max) normalized [0,1]; 1.0 == bw_range_max Mbps
        ram_range_max:    kept for API consistency; cnf_ram is already in GB
        bw_range_max:     bandwidth denormalization factor in Mbps
        alpha_mig:        penalty scaling coefficient (tunable; default 0.5)
        dirty_ratio:      fraction of RAM actively dirtied during migration (default 0.20)

    Returns:
        total_penalty (float): summed migration cost in ms across all migrating CNFs
    """
    total_penalty = 0.0
    M_max = placement_matrix.shape[0]

    for m in range(M_max):
        if not cnf_active[m]:
            continue

        prev_node = int(cnf_prev_node[m])
        new_node  = int(np.argmax(placement_matrix[m]))

        # No migration cost for new SFCs or CNFs that stay on the same node
        if prev_node < 0 or prev_node == new_node:
            continue

        # RAM payload in GB (raw value from SFCBatch, not normalized)
        ram_gb = float(cnf_ram[m])

        # Available link BW on direct edge prev_node → new_node in Mbps
        bw_norm = float(edge_bw[prev_node, new_node])
        bw_mbps = bw_norm * bw_range_max
        bw_mbps = max(bw_mbps, 10.0)   # floor: 10 Mbps (avoids div-by-zero on isolated nodes)

        # Migration time (seconds): dirty_pages_MB / bandwidth_Mbps
        # dirty_pages_MB = dirty_ratio × ram_GB × 1000 MB/GB
        t_mig_sec = (dirty_ratio * ram_gb * 1000.0) / bw_mbps

        # Convert to ms and apply penalty coefficient
        total_penalty += alpha_mig * t_mig_sec * 1000.0

    return float(total_penalty)
