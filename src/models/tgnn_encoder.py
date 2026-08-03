import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SpatialGNN(nn.Module):
    """
    Spatial GNN feature encoder applying stacked GCNConv + LayerNorm + ReLU.
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        curr_dim = in_dim
        for _ in range(n_layers):
            self.layers.append(GCNConv(curr_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            curr_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim) or (B, N, in_dim)
        is_batched = x.dim() == 3
        if is_batched:
            B, N, D = x.shape
            outputs = []
            for b in range(B):
                h = x[b]
                for layer, norm in zip(self.layers, self.norms):
                    h = layer(h, edge_index)
                    h = norm(h)
                    h = F.relu(h)
                outputs.append(h)
            return torch.stack(outputs, dim=0)
        else:
            h = x
            for layer, norm in zip(self.layers, self.norms):
                h = layer(h, edge_index)
                h = norm(h)
                h = F.relu(h)
            return h


class TGNNEncoder(nn.Module):
    """
    Spatio-Temporal Graph Neural Network Encoder.
    Combines SpatialGNN over W history steps with GRU temporal aggregation.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        tgnn_cfg = cfg.get("tgnn", cfg)
        self.f_node = tgnn_cfg.get("f_node", 6)
        self.f_cnf = tgnn_cfg.get("f_cnf", 5)
        self.d_hidden = tgnn_cfg.get("d_hidden", 128)
        self.d_model = tgnn_cfg.get("d_model", 128)
        self.w = tgnn_cfg.get("temporal_window", 5)
        self.n_spatial = tgnn_cfg.get("n_spatial_layers", 2)

        self.spatial_gnn = SpatialGNN(in_dim=self.f_node, hidden_dim=self.d_hidden, n_layers=self.n_spatial)

        self.gru = nn.GRU(
            input_size=self.d_hidden,
            hidden_size=self.d_model,
            num_layers=1,
            batch_first=True,
        )

        self.cnf_mlp = nn.Sequential(
            nn.Linear(self.f_cnf, self.d_hidden),
            nn.LayerNorm(self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, self.d_model),
        )

    def forward(
        self,
        node_features: torch.Tensor,   # (B, C_max, F_node)
        edge_index: torch.Tensor,      # (2, E)
        node_history: torch.Tensor,    # (B, W, C_max, F_node)
        cnf_features: torch.Tensor,    # (B, M_max, F_cnf)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, C_max, _ = node_features.shape
        _, W, _, _ = node_history.shape

        # 1. Spatial Encoding across (W + 1) timesteps
        spatial_embeddings = []
        for t in range(W):
            h_t = self.spatial_gnn(node_history[:, t, :, :], edge_index)  # (B, C_max, d_hidden)
            spatial_embeddings.append(h_t)

        h_curr = self.spatial_gnn(node_features, edge_index)  # (B, C_max, d_hidden)
        spatial_embeddings.append(h_curr)

        # Stack over time: (B, W+1, C_max, d_hidden)
        seq = torch.stack(spatial_embeddings, dim=1)

        # 2. GRU Temporal Aggregation
        # Reshape to (B * C_max, W+1, d_hidden)
        seq_flat = seq.permute(0, 2, 1, 3).reshape(B * C_max, W + 1, self.d_hidden)
        gru_out, _ = self.gru(seq_flat)  # (B * C_max, W+1, d_model)

        node_embeddings = gru_out[:, -1, :].reshape(B, C_max, self.d_model)

        # 3. CNF MLP Encoding
        cnf_embeddings = self.cnf_mlp(cnf_features)  # (B, M_max, d_model)

        return node_embeddings, cnf_embeddings
