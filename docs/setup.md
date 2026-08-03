# Project Setup & Environment Guide

## 1. Overview
This document provides instructions for setting up the local development environment, installing project dependencies, configuring environment and model parameters, and verifying environment sanity for the **Dynamic Cloud-Native Network Function (CNF) Placement** framework.

---

## 2. Prerequisites & Environment

- **Operating System:** Linux (Arch / Ubuntu) or macOS
- **Python Version:** Python $\ge 3.10$
- **Virtual Environment Tool:** `venv` or `conda`

---

## 3. Step-by-Step Installation

### 3.1 Clone & Navigate
```bash
cd /home/bow/projects/cloud-continuum-tgnn-nco
```

### 3.2 Activate Virtual Environment
```bash
source venv/bin/activate  # for bash/zsh
# or: source venv/bin/activate.fish  # for fish shell
```

### 3.3 Install Dependencies
Install production dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3.4 Install Local Package in Editable Mode
Install the root package via `pyproject.toml` so `src/` modules can be imported across tests and scripts:
```bash
pip install -e .
```

---

## 4. Configuration System Overview

All hyperparameters and parameters are managed centrally via YAML configuration files located in `configs/`:

1. **`configs/env_config.yaml`**:
   - Padded scale: $C_{\max}=50$, $M_{\max}=150$, $H_{\max}=30$, $W=5$.
   - Resource ranges (CPU, RAM, Storage, Bandwidth, Latency).
   - Dynamic noise parameters (Ornstein-Uhlenbeck process speeds and volatilities).
   - Cost weights per layer (Edge, Fog, Cloud).

2. **`configs/model_config.yaml`**:
   - TGNN encoder dimensions ($f_{\text{node}}=6$, $f_{\text{edge}}=3$, $f_{\text{cnf}}=5$, $d_{\text{model}}=128$).
   - Actor-Critic cross-attention architecture parameters.
   - PPO hyperparameters ($\text{lr}=3\times 10^{-4}$, $\gamma=0.99$, $\lambda=0.95$, $\epsilon=0.2$).

3. **`configs/evaluation_config.yaml`**:
   - Evaluation scenarios (In-Distribution, Out-of-Distribution, Constraint-Tight).
   - GEKKO MINLP baseline timeout bounds and node/CNF limits.

---

## 5. Verification Commands

Run the following sanity check to confirm PyTorch, PyTorch Geometric, and package imports function properly:

```bash
python -c "import torch; import torch_geometric; import src; print('Setup successfully verified!')"
```
