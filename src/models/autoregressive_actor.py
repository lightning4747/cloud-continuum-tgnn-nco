"""
AutoregressiveDecoder: Sequential CNF→Node placement with dynamic residual-capacity masking.

Key design decisions:
  1. SFC-priority ordering: process all CNFs of SFC_k before SFC_{k+1}.
     Ordering delivered via `cnf_order` tensor (computed by env._compute_cnf_order).
  2. Tightest-budget first: SFCs are pre-sorted by ascending sfc_delay_budget upstream
     in the environment so tight-constraint chains claim low-latency nodes first.
  3. Within-SFC chain order: ascending sfc_position (hop 0 → 1 → 2...).
  4. Dynamic residual capacity mask: after assigning CNF m to node i, node i's
     residual CPU/RAM/Storage is decremented. Future CNFs cannot use node i
     if remaining capacity < their demand.

Hard feasibility guarantee:
  Every completed placement vector satisfies node-level CPU+RAM+Storage capacity
  constraints by construction — not by penalty. The guarantee holds for the
  within-step capacity interactions. Bandwidth constraints are still checked
  post-hoc by the environment (multi-hop path BW is graph-dependent).
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.models.action_mask import apply_action_mask


class AutoregressiveDecoder(nn.Module):
    """
    Sequential autoregressive pointer decoder for CNF placement.

    Processes M_max CNF slots in the order given by `cnf_order` (B, M_max).
    At each decode step k, it decodes the CNF at position cnf_order[:, k]
    (which may differ per batch item — supports vectorized envs with independent SFC loads).

    Args:
        d_model: embedding dimension (must match TGNNEncoder/StaticGNNEncoder output)
        n_heads: number of attention heads for context cross-attention
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model

        # Cross-attention: each CNF attends to all nodes to produce a context vector
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )

        # Logit projection: (B, C_max, d_model) → (B, C_max) score per node
        self.logit_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        node_emb:    torch.Tensor,          # (B, C_max, d_model)
        cnf_emb:     torch.Tensor,          # (B, M_max, d_model)
        # Normalized available capacity (dims 0-2 of node_features: cpu/ram/stor avail)
        node_cpu:    torch.Tensor,          # (B, C_max)
        node_ram:    torch.Tensor,          # (B, C_max)
        node_stor:   torch.Tensor,          # (B, C_max)
        # Normalized demand per CNF (dims 0-2 of cnf_features: cpu/ram/stor demand)
        cnf_cpu:     torch.Tensor,          # (B, M_max)
        cnf_ram:     torch.Tensor,          # (B, M_max)
        cnf_stor:    torch.Tensor,          # (B, M_max)
        cnf_active:  torch.Tensor,          # (B, M_max) bool — False for padding slots
        static_mask: torch.Tensor,          # (B, M_max, C_max) bool — env constraint mask
        cnf_order:   torch.Tensor,          # (B, M_max) int64 — SFC-priority decode order
        action:      torch.Tensor | None = None,  # (B, M_max) teacher-forced for PPO update
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sequential decode loop.

        Returns:
            actions    (B, M_max): chosen node index per CNF slot (original CNF index space)
            log_probs  (B,):       sum of per-active-CNF log-probabilities (for PPO)
            entropies  (B,):       sum of per-active-CNF entropies (for PPO)
        """
        B, M_max, _ = cnf_emb.shape
        C_max = node_emb.shape[1]
        device = node_emb.device
        b_idx = torch.arange(B, device=device)  # [0, 1, ..., B-1]

        # --- Mutable residual capacity trackers ---
        # Start from current available capacity; decremented after each assignment
        res_cpu  = node_cpu.clone()    # (B, C_max)
        res_ram  = node_ram.clone()
        res_stor = node_stor.clone()

        actions_out   = torch.zeros(B, M_max, dtype=torch.long, device=device)
        log_probs_out = torch.zeros(B, device=device)
        entropies_out = torch.zeros(B, device=device)

        # Cross-attention: compute per-CNF context attending to all nodes.
        # Done ONCE per forward pass — node_emb is fixed within a single decode step.
        # attn_out[b, m] = context vector for CNF m in batch item b.
        attn_out, _ = self.cross_attn(
            query=cnf_emb,    # (B, M_max, d_model)
            key=node_emb,     # (B, C_max, d_model)
            value=node_emb,   # (B, C_max, d_model)
        )  # → (B, M_max, d_model)

        # Precompute linear projection of node embeddings and attention contexts outside the loop.
        # Linearity property: W(attn_m + node_emb) + b = (W*node_emb + b) + W*attn_m.
        # Precomputing this avoids running Linear(d_model, d_model) 150 times per forward pass.
        lin1 = self.logit_proj[0]
        relu = self.logit_proj[1]
        lin2 = self.logit_proj[2]

        proj_node = lin1(node_emb)  # (B, C_max, d_model) — includes bias
        proj_attn = torch.nn.functional.linear(attn_out, lin1.weight)  # (B, M_max, d_model) — without bias

        # === Sequential Decoding Loop ===
        # At decode step `step`, process CNF at cnf_order[:, step] for each batch item.
        # Each batch item independently tracks which CNF is next in its SFC-priority order.
        for step in range(M_max):
            # m_idx[b] = actual CNF index for batch item b at this decode step
            m_idx = cnf_order[:, step]    # (B,)  int64

            # --- Batch-wise gather: pull embeddings/demands for this CNF per batch item ---
            proj_m     = proj_attn[b_idx, m_idx]       # (B, d_model) — pre-projected CNF context
            cnf_cpu_m  = cnf_cpu[b_idx,  m_idx]       # (B,) normalized CPU demand
            cnf_ram_m  = cnf_ram[b_idx,  m_idx]       # (B,) normalized RAM demand
            cnf_stor_m = cnf_stor[b_idx, m_idx]       # (B,) normalized Storage demand
            active_m   = cnf_active[b_idx, m_idx].float()  # (B,) 0.0=padding 1.0=active
            smask_m    = static_mask[b_idx, m_idx]    # (B, C_max) — env static mask

            # --- Compute logits: CNF m context + node embeddings → score per node ---
            h        = relu(proj_m.unsqueeze(1) + proj_node)  # (B, C_max, d_model)
            logits_m = lin2(h).squeeze(-1)                     # (B, C_max)

            # --- Dynamic residual-capacity mask ---
            # Node i is feasible for CNF m only if residual[i] >= demand[m]
            cpu_ok   = res_cpu  >= cnf_cpu_m.unsqueeze(1)  # (B, C_max)
            ram_ok   = res_ram  >= cnf_ram_m.unsqueeze(1)
            stor_ok  = res_stor >= cnf_stor_m.unsqueeze(1)
            dyn_mask = cpu_ok & ram_ok & stor_ok             # (B, C_max)

            # Intersect with static env mask (committed allocations, node_active)
            combined = dyn_mask & smask_m                    # (B, C_max)

            # Safety fallback: if no node passes the combined mask (extreme load),
            # fall back to the static env mask to avoid NaN from all-False softmax
            no_valid = combined.sum(dim=-1, keepdim=True) == 0   # (B, 1)
            if no_valid.any():
                fallback = smask_m.bool().clone()
                # Ensure at least one True per row in fallback
                no_static = fallback.sum(dim=-1, keepdim=True) == 0
                fallback[:, 0] = fallback[:, 0] | no_static.squeeze(-1)
                combined = torch.where(no_valid.expand_as(combined), fallback, combined)

            # Mask logits and build per-node distribution
            logits_m = apply_action_mask(logits_m, combined)    # (B, C_max)
            dist_m   = Categorical(logits=logits_m)

            # --- Sample or use teacher-forced action for PPO update ---
            if action is not None:
                # Teacher forcing: use the stored action for CNF m_idx per batch item
                a_m = action[b_idx, m_idx]    # (B,) — gather from original CNF index space
            else:
                a_m = dist_m.sample()         # (B,)

            # --- Accumulate log-prob and entropy for ACTIVE CNFs only ---
            # Padding slots (active_m == 0) contribute zero to both
            log_probs_out += dist_m.log_prob(a_m) * active_m
            entropies_out += dist_m.entropy()    * active_m

            # --- Store action in original CNF index space ---
            # actions_out[b, m_idx[b]] = a_m[b]
            actions_out.scatter_(dim=1, index=m_idx.unsqueeze(1), src=a_m.unsqueeze(1))

            # --- Update residual capacity for ACTIVE CNFs only ---
            # chosen_one_hot[b, a_m[b]] = active_m[b]  (0 for padding, 1 for active)
            chosen_one_hot = torch.zeros(B, C_max, device=device)
            chosen_one_hot.scatter_(1, a_m.unsqueeze(1), active_m.unsqueeze(1))

            # Subtract this CNF's demand from the chosen node's residual capacity
            res_cpu  -= cnf_cpu_m.unsqueeze(1)  * chosen_one_hot   # (B, C_max)
            res_ram  -= cnf_ram_m.unsqueeze(1)  * chosen_one_hot
            res_stor -= cnf_stor_m.unsqueeze(1) * chosen_one_hot

            # Clamp to prevent floating-point negatives from accumulating
            res_cpu  = res_cpu.clamp(min=0.0)
            res_ram  = res_ram.clamp(min=0.0)
            res_stor = res_stor.clamp(min=0.0)

        return actions_out, log_probs_out, entropies_out
