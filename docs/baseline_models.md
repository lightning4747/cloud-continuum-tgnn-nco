# Baseline Models Specification & Benchmarking Protocol

## 1. Overview
This document specifies the target baseline algorithms used to evaluate the efficiency, feasibility rate, optimality gap, and inference latency of the proposed **TGNN-NCO** placement policy.

---

## 2. Benchmark Algorithm Suite

### 2.1 Exact Solver Baseline: `MINLPSolver` (GEKKO MILP)
- **Formulation:** Mixed-Integer Linear Program formulated in GEKKO.
- **Objective:**
  $$\min \sum_{m, i} x_{imh} \cdot \text{Cost}_i(m) + \alpha \cdot \text{LatencyPenalty}$$
- **Scope Limitation:** Evaluated strictly on small topology instances ($C_{\text{active}} \le 15, M_{\text{active}} \le 30$) with a 60-second timeout limit.
- **Role:** Computes the exact ground-truth optimal placement for calculating Optimality Gap (%).

---

### 2.2 Heuristic Baselines

#### A. `GreedyFFD` (First-Fit Decreasing)
- **Algorithm:** Sorts active CNF requests in descending order of CPU demand $x_m$. Iterates through nodes sorted by available capacity, placing CNF $m$ on the first node with sufficient CPU, RAM, and Storage.
- **Role:** Represents standard heuristic scheduler performance and establishes the runtime speed benchmark.

#### B. `GreedyLatencyAware`
- **Algorithm:** Sorts SFCs by delay budget tightness ($T_h$). Places consecutive CNFs along shortest propagation latency paths using `NetworkX` shortest path algorithms.
- **Role:** Evaluates heuristic SLA delay budget compliance vs. deployment cost.

---

### 2.3 Deep Learning & RL Ablation Baselines

#### A. `StaticGNN` (Ablation: No Temporal Module)
- **Architecture:** Employs spatial graph convolutions (`GCNConv`) on the current graph state $G(t)$ without temporal GRU history ($W=1$).
- **Role:** Isolates and quantifies the performance contribution of the temporal GRU module in dynamic topologies.

#### B. `FlatRL` (Ablation: No Graph Neural Network)
- **Architecture:** Standard PPO actor-critic network operating on flattened node and CNF feature vectors without GNN message-passing layers.
- **Role:** Demonstrates the impact of spatial topology graph embeddings.

#### C. `NoMaskRL` (Ablation: No Action Masking)
- **Architecture:** TGNN-NCO architecture trained without applying $-\infty$ logit action masks.
- **Role:** Proves the necessity of explicit action masking for achieving zero resource/capacity constraint violations.

---

## 3. Evaluation Metrics

Every solver is evaluated across 500 test episodes using the following standardized metrics:
1. **Feasibility Rate (%):** Percentage of placement actions satisfying all hard CPU, RAM, Storage, Bandwidth, and Latency constraints.
2. **Mean Deployment Cost:** Total monetary cost of assigned computing resources.
3. **End-to-End Latency (ms):** Average SFC traversal delay across all chains.
4. **Optimality Gap (%):** $\frac{\text{Cost}_{\text{RL}} - \text{Cost}_{\text{MINLP}}}{\text{Cost}_{\text{MINLP}}} \times 100$
5. **Inference Time (ms):** Runtime required to compute placement decisions for an entire interval.
