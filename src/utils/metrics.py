import numpy as np


def compute_deployment_cost(placement: np.ndarray, node_cpu_demands: np.ndarray, node_layers: np.ndarray) -> float:
    cost_map = {0: 0.05, 1: 0.10, 2: 0.20}
    node_costs = np.array([cost_map[int(l)] for l in node_layers], dtype=np.float32)
    return float(np.sum(placement * node_cpu_demands[:, None] * node_costs[None, :]))


def compute_feasibility_rate(results: list[dict]) -> float:
    if not results:
        return 0.0
    feasible_count = sum(1 for r in results if r.get("feasible", False))
    return (feasible_count / len(results)) * 100.0


def compute_optimality_gap(rl_cost: float, minlp_cost: float) -> float:
    if minlp_cost <= 0:
        return 0.0
    return ((rl_cost - minlp_cost) / minlp_cost) * 100.0
