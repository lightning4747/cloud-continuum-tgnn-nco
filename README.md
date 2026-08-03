# Dynamic Cloud-Native Network Function Placement on Cloud-Continuum using Spatio-Temporal Graph Neural Networks and Neural Combinatorial Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3%2B-39a0ed.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end framework for near real-time, zero-violation placement of Service Function Chains (SFCs) composed of Cloud-Native Network Functions (CNFs) across heterogeneous Cloud-Continuum infrastructure (Edge, Fog, Cloud). Employs Spatio-Temporal Graph Neural Networks (TGNN) to capture continuous network temporal dynamics and Actor-Critic Reinforcement Learning (PPO) with differentiable tensor action-masking for hard constraint enforcement.

---

## Table of Contents

1. [Executive Summary & System Framework](#executive-summary--system-framework)
2. [Initial Setup & Replication Guide](#initial-setup--replication-guide)
3. [Simulation Environment Details](#simulation-environment-details)
4. [Baseline Models to Compare](#baseline-models-to-compare)
5. [Testing & Verification Plan](#testing--verification-plan)
6. [Repository Structure](#repository-structure)
7. [IEEE Paper Artifact Generation](#ieee-paper-artifact-generation)

---

## Executive Summary & System Framework

### Technical Architecture

```text
[ Infrastructure Telemetry & Dynamic SFC Requests ]
                        │
                        ▼
            [ Temporal Graph G(t) ]
                        │
                        ▼
     [ TGNN Encoder (Spatio-Temporal Embeddings) ]
                        │
                        ▼
      [ NCO Actor-Critic RL Policy (PPO) ]
                        │
      ┌─────────────────┴─────────────────┐
      │  Action Masking (-inf Logits)     │ ◄── Enforces hard CPU/RAM/Storage limits
      └─────────────────┬─────────────────┘
                        ▼
          [ Discrete Placement Decision ]
                        │
                        ▼
           [ Kubernetes Cluster API ]
                        │
                        ▼ (Telemetry Feedback Loop)
            [ Environment State Update ]
```

- **Temporal Encoding (TGNN):** Combines spatial graph convolutions (`GCNConv` / `SAGEConv`) over Waxman topologies with a `GRU` temporal aggregation layer across a sliding window $W=5$.
- **Combinatorial Optimization (NCO):** Employs an Actor-Critic PPO policy with cross-attention mapping CNF demands to continuum node embeddings.
- **Differentiable Action Masking:** Sets logits of invalid placement actions to $-\infty$ ($-1\times 10^9$) prior to Softmax sampling to guarantee zero capacity/RAM/Storage constraint violations.
- **Execution Boundary:** External dry-run Kubernetes API interface submitting pod specs without cluster modifications.

---

## Initial Setup & Replication Guide

### Prerequisites

- Linux / macOS (Python $\ge 3.10$)
- Virtual Environment (`venv` or `conda`)
- PyTorch $\ge 2.0$ (CPU local development, CUDA automatically selected on Colab)

### Step-by-step Installation

1. **Clone Repository & Navigate to Workspace:**
   ```bash
   cd /path/to/cloud-continuum-tgnn-nco
   ```

2. **Activate Virtual Environment:**
   ```bash
   source venv/bin/activate  # or activate.fish for Fish shell
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Package in Editable Mode:**
   ```bash
   pip install -e .
   ```

5. **Verify Installation:**
   ```bash
   python -c "import torch; import torch_geometric; import src; print('Setup successful!')"
   ```

---

## Simulation Environment Details

The simulation framework represents dynamic Cloud-Continuum network fluctuations via discrete scheduling intervals $t$.

### 1. Continuum Topology & Resources
- **Graph Representation:** $G(t) = (V(t), E(t))$ generated via Waxman random graph parameters ($\alpha=0.5, \beta=0.5$).
- **Padded Dimensions:** $C_{\max} = 50$ nodes, $M_{\max} = 150$ CNFs, $H_{\max} = 30$ active SFCs per interval.
- **Layer Allocation:** Heterogeneous node distribution across Edge (40%), Fog (30%), and Cloud (30%).
- **Resource Ranges:**
  - Node CPU: $U[4, 32]$ cores
  - Node RAM: $U[8, 128]$ GB
  - Node Storage: $U[50, 1000]$ GB
  - Link Bandwidth: $U[100, 10000]$ Mbps
  - Link Propagation Latency: $U[1, 100]$ ms

### 2. Temporal Network Dynamics
- **Ornstein-Uhlenbeck (OU) Noise:** Applied to available node resources and link capacities at each timestep to model non-stationary network load and background dynamic traffic:
  $$\mathrm{d}X_t = \theta (\mu - X_t)\mathrm{d}t + \sigma \mathrm{d}W_t \quad (\theta=0.15, \sigma=0.05)$$
- **Dynamic Events:** Random node failures/recoveries (probability $0.01$/step) and dynamic SFC arrivals/retirements (probability $0.10$/step).

### 3. Objective & Constraints
- **Objective:**
  $$\min_{X(t)} \sum_{h=1}^{H} \sum_{m=1}^{l_h} \sum_{i=1}^{C} x_{imh}(t) \cdot \text{Cost}_i(m) + \alpha \cdot \text{LatencyPenalty}(X(t))$$
- **Hard Constraints:** CPU capacity, RAM capacity, Storage capacity, Link bandwidth, End-to-end SFC delay budget ($D_h(t) \le T_h$), and layer compatibility.

---

## Baseline Models to Compare

To validate the efficiency, scalability, and optimality of **TGNN-NCO**, the framework includes comparative benchmarks across four distinct algorithmic families:

| Model Family | Specific Baseline | Description & Purpose |
|--------------|-------------------|-----------------------|
| **Exact Solver** | `MINLPSolver` (GEKKO) | Formulates MILP over small topologies ($C \le 15, M \le 30$) to measure exact Optimality Gap (%). |
| **Heuristics** | `GreedyFFD` | First-Fit Decreasing heuristic sorting CNFs by CPU demand; measures runtime speedup vs. optimization quality. |
| **Heuristics** | `GreedyLatencyAware` | Prioritizes placing consecutive chain CNFs on shortest propagation path nodes. |
| **DL / RL (Ablation)** | `StaticGNN` | Uses spatial GCN without temporal GRU history ($W=1$) to quantify the benefit of spatio-temporal modeling. |
| **DL / RL (Ablation)** | `FlatRL` | Standard PPO without GNN graph convolutions (flattened feature input). |
| **DL / RL (Ablation)** | `NoMaskRL` | TGNN-NCO trained without action masking to prove the necessity of hard constraint layer. |

---

## Testing & Verification Plan

The codebase includes an extensive Pytest suite in `tests/` with strict coverage targets:

### 1. Unit Tests
- **Environment Tests (`tests/test_env.py`):**
  - Observation shape verification against `C_max=50`, `M_max=150`, `W=5`.
  - Step function determinism given identical random seeds.
  - 1000-step random rollout stability checks.
- **Action Masking Tests (`tests/test_masking.py`):**
  - Verification that invalid node logits are masked to $-1\times 10^9$.
  - Verification that capacity mask correctly blocks overloaded nodes.
  - Gradient flow verification through unmasked action paths.
- **Model Tests (`tests/test_models.py`):**
  - Tensor dimensionality checks for `TGNNEncoder` and `ActorCritic`.
  - PPO update step loss backward pass sanity checks.

### 2. Verification Commands
```bash
# Run pytest suite
pytest tests/ -v --tb=short

# Run smoke test on environment
python -c "from src.env.continuum_env import ContinuumEnv; env = ContinuumEnv(); obs, _ = env.reset(seed=42); print(obs['node_features'].shape)"

# Execute dry-run training loop (10 steps)
python scripts/train.py --config configs/model_config.yaml --max-steps 10 --dry-run
```

---

## Repository Structure

```text
cloud-continuum-tgnn-nco/
├── configs/                      # YAML configurations
│   ├── env_config.yaml           # Dimensions, noise parameters, node bounds
│   ├── model_config.yaml         # TGNN hidden dims, PPO learning rates
│   └── evaluation_config.yaml    # Benchmark scenarios and protocols
│
├── src/                          # Main library package
│   ├── env/                      # Gymnasium continuum environment & generators
│   ├── models/                   # PyTorch TGNN encoder, action masker, PPO model
│   ├── baselines/                # GEKKO MINLP, Greedy, Static GNN baselines
│   └── utils/                    # Metrics, logger, seed helpers
│
├── scripts/                      # CLI entrypoints
│   ├── train.py                  # Training pipeline script
│   ├── evaluate.py               # Comparative benchmark script
│   └── generate_plots.py         # IEEE paper plot generation script
│
├── tests/                        # Pytest suite
│   ├── test_env.py
│   ├── test_masking.py
│   └── test_models.py
│
├── pyproject.toml                # Package metadata
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation & replication guide
```

---

## IEEE Paper Artifact Generation

After running model training (`scripts/train.py`) and comparative evaluation (`scripts/evaluate.py`), run:

```bash
python scripts/generate_plots.py
```

This generates IEEE-formatted PDF vector graphics in `results/figures/`:
- **Fig. 1:** Feasibility Rate (%) across tight/loose constraint regimes.
- **Fig. 2:** Inference Time (ms) vs. Number of Nodes (Log-Scale).
- **Fig. 3:** Optimality Gap (%) CDF curve vs. Exact GEKKO MINLP.
- **Fig. 4:** Training Reward & Constraint Violation convergence curves.
- **Fig. 5:** Ablation Study comparison (TGNN vs. Static GNN vs. Flat RL vs. NoMask).
