# Testing & Verification Plan

## 1. Overview
The testing framework ensures code correctness, tensor shape alignment, constraint enforcement, and policy stability across all components of the repository.

---

## 2. Test Architecture & Coverage Targets

| Test Suite File | Focus Area | Target Code Coverage |
|-----------------|------------|----------------------|
| `tests/test_env.py` | Environment reset, step transitions, observation shape validation, state buffer updates | $\ge 80\%$ |
| `tests/test_masking.py` | Action masking functions, capacity checks, layer compatibility, gradient flow | $100\%$ |
| `tests/test_models.py` | TGNN spatial/temporal forward passes, Actor-Critic sampling, PPO update backward pass | $\ge 70\%$ |

---

## 3. Test Suite Details

### 3.1 Environment Tests (`tests/test_env.py`)
- **`test_reset_obs_shapes`**: Verifies that environment reset returns observation tensors matching padded bounds ($C_{\max}=50, M_{\max}=150, W=5$).
- **`test_reset_reproducibility`**: Confirms that setting identical seeds yields deterministic environment states.
- **`test_1000_step_rollout_no_crash`**: Executes a 1000-step random action rollout to verify environment stability under dynamic node failures and SFC arrivals.

### 3.2 Action Masking Tests (`tests/test_masking.py`)
- **`test_masked_logits_are_neg_inf`**: Validates that setting action mask values to `False` sets corresponding logits to $-1\times 10^9$.
- **`test_capacity_mask_zero_cpu`**: Confirms that nodes with insufficient remaining CPU receive all-False mask columns.
- **`test_gradient_flows_through_unmasked`**: Ensures gradient computation flows through valid action logit paths during backpropagation.

### 3.3 Model Tests (`tests/test_models.py`)
- **`test_tgnn_encoder_output_shapes`**: Verifies `TGNNEncoder` returns node embeddings of shape `(B, C_max, 128)` and CNF embeddings of shape `(B, M_max, 128)`.
- **`test_actor_critic_forward`**: Checks categorical distribution sampling and value output shape `(B, 1)`.
- **`test_ppo_single_update_step`**: Validates that PPO loss computation and optimizer step execute without throwing NaN/Inf values.

---

## 4. Test Execution Instructions

### Run Pytest Suite
```bash
# Run all unit tests with short traceback
pytest tests/ -v --tb=short

# Run tests with coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

### Dry-Run Integration Checks
```bash
# Sanity test environment reset
python -c "from src.env.continuum_env import ContinuumEnv; e = ContinuumEnv(); obs, _ = e.reset(seed=42); print(obs['node_features'].shape)"

# Smoke test training pipeline (10 steps)
python scripts/train.py --config configs/model_config.yaml --max-steps 10 --dry-run
```
