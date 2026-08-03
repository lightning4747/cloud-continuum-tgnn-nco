import torch
import torch.nn as nn
from src.models.actor_critic import ActorCritic
from src.models.tgnn_encoder import SpatialGNN


class StaticGNNEncoder(nn.Module):
    """
    Spatial-only GNN encoder without GRU temporal aggregation.
    Used for the Static GNN baseline ablation study.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        tgnn_cfg = cfg.get("tgnn", cfg)
        self.f_node = tgnn_cfg.get("f_node", 6)
        self.f_cnf = tgnn_cfg.get("f_cnf", 5)
        self.d_hidden = tgnn_cfg.get("d_hidden", 128)
        self.d_model = tgnn_cfg.get("d_model", 128)
        self.n_spatial = tgnn_cfg.get("n_spatial_layers", 2)

        self.spatial_gnn = SpatialGNN(in_dim=self.f_node, hidden_dim=self.d_hidden, n_layers=self.n_spatial)
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
        # Ignores node_history, uses only current step node_features
        node_embeddings = self.spatial_gnn(node_features, edge_index)
        cnf_embeddings = self.cnf_mlp(cnf_features)
        return node_embeddings, cnf_embeddings


class StaticGNNActorCritic(ActorCritic):
    """
    ActorCritic model utilizing StaticGNNEncoder for ablation comparison.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.encoder = StaticGNNEncoder(cfg)
