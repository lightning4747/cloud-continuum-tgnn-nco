# Dynamic Cloud-Continuum Simulation Specification

## 1. Overview
The continuum simulation engine models non-stationary, heterogeneous network environments where Edge, Fog, and Cloud computing nodes experience dynamic resource fluctuations and fluctuating Service Function Chain (SFC) demands over discrete scheduling intervals $t$.

---

## 2. Infrastructure Representation

The physical infrastructure graph at time $t$ is represented as:
$$G(t) = (V(t), E(t))$$

### 2.1 Node Metrics & Tensors
Nodes are allocated across three continuum tiers (Edge: 40%, Fog: 30%, Cloud: 30%):
- **CPU Capacity ($X_i(t)$):** Available processing cores $\in U[4, 32]$.
- **RAM Capacity ($D_i(t)$):** Available memory $\in U[8, 128]$ GB.
- **Storage Capacity ($S_i(t)$):** Available disk $\in U[50, 1000]$ GB.
- **Node Cost Rate ($\text{Cost}_i$):** Per-core deployment cost factors (Edge: \$0.05, Fog: \$0.10, Cloud: \$0.20).

### 2.2 Link Metrics & Tensors
Communication links $(i, j) \in E(t)$ contain time-varying properties:
- **Bandwidth ($c_{ij}(t)$):** Available link capacity $\in U[100, 10000]$ Mbps.
- **Latency ($l_{ij}(t)$):** Propagation delay $\in U[1, 100]$ ms.

---

## 3. Workload Model (SFCs and CNFs)

Each scheduling step introduces active Service Function Chains $h \in \mathcal{S}$:
- **Atomic CNFs:** Each SFC $h$ consists of $l_h \in [2, 8]$ ordered CNFs.
- **CNF Resource Demands:** CPU $x_m \in [0.5, 8]$ cores, RAM $d_m \in [0.5, 16]$ GB, Storage $s_m \in [1, 50]$ GB.
- **Inter-CNF Traffic Rate:** $r_{uv}(t) \in [10, 1000]$ Mbps.
- **End-to-End Latency Budget ($T_h$):** Maximum allowed delay budget $\in [20, 200]$ ms per chain.

---

## 4. Non-Stationary Dynamic Processes

### 4.1 Ornstein-Uhlenbeck Noise
Resource availability fluctuates smoothly across consecutive timesteps via mean-reverting noise:
$$\mathrm{d}X_t = \theta (\mu - X_t)\mathrm{d}t + \sigma \mathrm{d}W_t$$
with mean reversion rate $\theta=0.15$ and volatility $\sigma=0.05$.

### 4.2 Dynamic Network Events
- **Node Availability:** Stochastic node failure/recovery events occur at probability $p_{\text{fail}} = 0.01$ per timestep.
- **Workload Lifecycle:** SFCs complete and retire with probability $p_{\text{retire}} = 0.10$ per timestep, triggering arrival of new SFC requests.

---

## 5. Tensor Dimension Padding
To maintain static PyTorch tensor input sizes during dynamic graph updates:
- Node feature tensors are padded to $C_{\max} = 50$.
- CNF feature tensors are padded to $M_{\max} = 150$.
- Inactive node/CNF tensor slots are masked using dynamic action masks ($-\infty$ logits).
