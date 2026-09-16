import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.models.action_mask import apply_action_mask
from src.models.autoregressive_actor import AutoregressiveDecoder
from src.models.tgnn_encoder import TGNNEncoder


class ActorCritic(nn.Module):
    """
    PPO Actor-Critic Architecture for Discrete CNF Placement.
    Actor: Multi-Head Cross-Attention (CNF queries -> Node keys/values)
    Critic: Mean-pooled global graph embedding -> MLP state value V(s)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.encoder = TGNNEncoder(cfg)

        ac_cfg = cfg.get("actor_critic", cfg)
        self.d_model = ac_cfg.get("d_model", 128)
        self.n_heads = ac_cfg.get("n_attention_heads", 4)

        # Cross-Attention Actor
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            batch_first=True,
        )
        self.logit_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
        )

        # Critic MLP
        self.critic = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_history: torch.Tensor,
        cnf_features: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[Categorical, torch.Tensor]:
        # Encodings
        node_emb, cnf_emb = self.encoder(node_features, edge_index, node_history, cnf_features)
        B, M_max, _ = cnf_emb.shape
        _, C_max, _ = node_emb.shape

        # Cross Attention: Q=cnf_emb, K=node_emb, V=node_emb
        attn_out, _ = self.cross_attn(query=cnf_emb, key=node_emb, value=node_emb)  # (B, M_max, d_model)

        # Logit calculation per (CNF, Node) pair with fine-grained chunking (chunk_size=32) for memory efficiency
        # Uses attn_out (cross-attended CNF embeddings) rather than raw cnf_emb
        if B > 32:
            chunk_size = 32
            logits_list = []
            for i in range(0, B, chunk_size):
                end = min(i + chunk_size, B)
                comb_chunk = attn_out[i:end].unsqueeze(2) + node_emb[i:end].unsqueeze(1)
                log_chunk = self.logit_proj(comb_chunk).squeeze(-1)
                logits_list.append(log_chunk)
            logits = torch.cat(logits_list, dim=0)
        else:
            combined = attn_out.unsqueeze(2) + node_emb.unsqueeze(1)  # Broadcast add: (B, M_max, C_max, d_model)
            logits = self.logit_proj(combined).squeeze(-1)             # (B, M_max, C_max)

        if action_mask is not None:
            if action_mask.shape[1] > M_max:
                action_mask = action_mask[:, :M_max, :]
            elif action_mask.shape[1] < M_max:
                pad_m = M_max - action_mask.shape[1]
                action_mask = torch.cat([action_mask, torch.zeros_like(action_mask[:, :pad_m, :])], dim=1)

            if action_mask.shape[2] > C_max:
                action_mask = action_mask[:, :, :C_max]
            elif action_mask.shape[2] < C_max:
                pad_c = C_max - action_mask.shape[2]
                action_mask = torch.cat([action_mask, torch.zeros_like(action_mask[:, :, :pad_c])], dim=2)

            logits = apply_action_mask(logits, action_mask)

        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        dist = Categorical(logits=logits)

        # Critic Value
        global_graph_emb = node_emb.mean(dim=1)  # (B, d_model)
        value = self.critic(global_graph_emb)    # (B, 1)

        return dist, value

    def get_action_and_value(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_history: torch.Tensor,
        cnf_features: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(node_features, edge_index, node_history, cnf_features, action_mask)

        if action is None:
            action = dist.sample()  # (B, M_max)

        log_prob = dist.log_prob(action).sum(dim=-1)  # Sum per-CNF log-probs -> (B,)
        entropy = dist.entropy().sum(dim=-1)           # Sum per-CNF entropies -> (B,)

        return action, log_prob, entropy, value

    def get_value(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_history: torch.Tensor,
        cnf_features: torch.Tensor,
    ) -> torch.Tensor:
        node_emb, _ = self.encoder(node_features, edge_index, node_history, cnf_features)
        global_graph_emb = node_emb.mean(dim=1)
        return self.critic(global_graph_emb)


class AutoregressiveActorCritic(nn.Module):
    """
    PPO Actor-Critic with Autoregressive Pointer Actor (SFC-Priority Ordering).

    Actor:  Sequential autoregressive decoding via AutoregressiveDecoder.
            Processes CNF slots in SFC-priority order (tightest delay_budget first,
            chain-position order within each SFC). Dynamic residual-capacity mask
            updated after each assignment guarantees 100% node capacity feasibility.

    Critic: Mean-pooled graph embedding → MLP value V(s). Identical to ActorCritic.

    Compatible with both TGNNEncoder (temporal) and StaticGNNEncoder (static ablation).
    Encoder is configurable via `cfg` or by replacing model.encoder after construction.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        tgnn_cfg = cfg.get("tgnn", cfg)
        ac_cfg   = cfg.get("actor_critic", cfg)
        d_model  = ac_cfg.get("d_model", tgnn_cfg.get("d_model", 256))
        n_heads  = ac_cfg.get("n_attention_heads", 8)

        self.encoder = TGNNEncoder(cfg)
        self.decoder = AutoregressiveDecoder(d_model=d_model, n_heads=n_heads)

        self.critic = nn.Sequential(
            nn.Linear(d_model, 512), nn.ReLU(),
            nn.Linear(512, 256),     nn.ReLU(),
            nn.Linear(256, 1),
        )

    def get_action_and_value(
        self,
        node_features: torch.Tensor,             # (B, C_max, 9)
        edge_index:    torch.Tensor,             # (2, E)
        node_history:  torch.Tensor,             # (B, W, C_max, 9)
        cnf_features:  torch.Tensor,             # (B, M_max, 5)
        action_mask:   torch.Tensor | None = None,  # (B, M_max, C_max) bool
        cnf_order:     torch.Tensor | None = None,  # (B, M_max) int64 SFC-priority order
        cnf_active:    torch.Tensor | None = None,  # (B, M_max) bool
        action:        torch.Tensor | None = None,  # (B, M_max) teacher-forced for PPO
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, M_max = cnf_features.shape[:2]
        C_max    = node_features.shape[1]
        device   = node_features.device

        # 1. Encode graph state and CNF demands
        node_emb, cnf_emb = self.encoder(
            node_features, edge_index, node_history, cnf_features
        )  # (B, C_max, d_model), (B, M_max, d_model)

        # 2. Defaults for optional tensors
        if action_mask is None:
            action_mask = torch.ones(B, M_max, C_max, dtype=torch.bool, device=device)
        if cnf_order is None:
            # Fallback: natural order 0..M_max-1 if env doesn't supply ordering
            cnf_order = torch.arange(M_max, device=device).unsqueeze(0).expand(B, -1)
        if cnf_active is None:
            cnf_active = torch.ones(B, M_max, dtype=torch.bool, device=device)

        # 3. Extract normalized capacity tensors from observation arrays
        # node_features dims 0-2: [cpu_avail_norm, ram_avail_norm, stor_avail_norm]
        node_cpu  = node_features[..., 0]    # (B, C_max)
        node_ram  = node_features[..., 1]
        node_stor = node_features[..., 2]
        # cnf_features dims 0-2: [cpu_demand_norm, ram_demand_norm, stor_demand_norm]
        cnf_cpu   = cnf_features[..., 0]     # (B, M_max)
        cnf_ram   = cnf_features[..., 1]
        cnf_stor  = cnf_features[..., 2]

        # 4. Autoregressive sequential decode
        actions, log_probs, entropies = self.decoder(
            node_emb=node_emb,
            cnf_emb=cnf_emb,
            node_cpu=node_cpu, node_ram=node_ram, node_stor=node_stor,
            cnf_cpu=cnf_cpu,   cnf_ram=cnf_ram,   cnf_stor=cnf_stor,
            cnf_active=cnf_active,
            static_mask=action_mask,
            cnf_order=cnf_order,
            action=action,
        )

        # 5. Critic: global mean-pooled graph embedding → state value V(s)
        global_emb = node_emb.mean(dim=1)    # (B, d_model)
        value = self.critic(global_emb)       # (B, 1)

        return actions, log_probs, entropies, value

    def get_value(
        self,
        node_features: torch.Tensor,
        edge_index:    torch.Tensor,
        node_history:  torch.Tensor,
        cnf_features:  torch.Tensor,
    ) -> torch.Tensor:
        node_emb, _ = self.encoder(node_features, edge_index, node_history, cnf_features)
        return self.critic(node_emb.mean(dim=1))
