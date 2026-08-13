import torch
import torch.nn as nn
from src.models.actor_critic import ActorCritic


class FlatRLEncoder(nn.Module):
    """
    Flat MLP encoder without spatial GCN convolution or temporal GRU aggregation.
    Used for the Flat RL ablation baseline study.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        tgnn_cfg = cfg.get("tgnn", cfg)
        self.f_node = tgnn_cfg.get("f_node", 6)
        self.f_cnf = tgnn_cfg.get("f_cnf", 5)
        self.d_hidden = tgnn_cfg.get("d_hidden", 128)
        self.d_model = tgnn_cfg.get("d_model", 128)

        self.node_mlp = nn.Sequential(
            nn.Linear(self.f_node, self.d_hidden),
            nn.LayerNorm(self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, self.d_model),
        )

        self.cnf_mlp = nn.Sequential(
            nn.Linear(self.f_cnf, self.d_hidden),
            nn.LayerNorm(self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, self.d_model),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_history: torch.Tensor,
        cnf_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Ignores edge_index and node_history, processes raw current-step node features
        node_embeddings = self.node_mlp(node_features)
        cnf_embeddings = self.cnf_mlp(cnf_features)
        return node_embeddings, cnf_embeddings


class FlatRLActorCritic(ActorCritic):
    """
    ActorCritic model utilizing FlatRLEncoder for ablation comparison.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.encoder = FlatRLEncoder(cfg)
