import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.models.action_mask import apply_action_mask
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

        # Logit calculation per (CNF, Node) pair with memory-efficient chunking for large batch sizes B
        if B > 256:
            chunk_size = 256
            logits_list = []
            for i in range(0, B, chunk_size):
                end = min(i + chunk_size, B)
                comb_chunk = cnf_emb[i:end].unsqueeze(2) + node_emb[i:end].unsqueeze(1)
                log_chunk = self.logit_proj(comb_chunk).squeeze(-1)
                logits_list.append(log_chunk)
            logits = torch.cat(logits_list, dim=0)
        else:
            combined = cnf_emb.unsqueeze(2) + node_emb.unsqueeze(1)  # Broadcast add: (B, M_max, C_max, d_model)
            logits = self.logit_proj(combined).squeeze(-1)            # (B, M_max, C_max)

        if action_mask is not None:
            logits = apply_action_mask(logits, action_mask)

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
