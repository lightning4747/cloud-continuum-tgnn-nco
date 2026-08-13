import torch
import yaml
from src.baselines.flat_rl import FlatRLActorCritic
from src.baselines.static_gnn import StaticGNNActorCritic
from src.models.actor_critic import ActorCritic
from src.models.tgnn_encoder import TGNNEncoder


def test_tgnn_encoder_shapes():
    with open("configs/model_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    d_model = cfg["tgnn"]["d_model"]
    encoder = TGNNEncoder(cfg)
    node_features = torch.randn(2, 50, 6)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    node_history = torch.randn(2, 5, 50, 6)
    cnf_features = torch.randn(2, 150, 5)

    node_emb, cnf_emb = encoder(node_features, edge_index, node_history, cnf_features)
    assert node_emb.shape == (2, 50, d_model)
    assert cnf_emb.shape == (2, 150, d_model)


def test_actor_critic_forward_and_sample():
    with open("configs/model_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    ac = ActorCritic(cfg)
    node_features = torch.randn(2, 50, 6)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    node_history = torch.randn(2, 5, 50, 6)
    cnf_features = torch.randn(2, 150, 5)
    action_mask = torch.ones((2, 150, 50), dtype=torch.bool)

    action, log_prob, entropy, value = ac.get_action_and_value(
        node_features, edge_index, node_history, cnf_features, action_mask=action_mask
    )

    assert action.shape == (2, 150)
    assert log_prob.shape == (2,)
    assert entropy.shape == (2,)
    assert value.shape == (2, 1)


def test_ablation_models_forward_and_interface_conformance():
    with open("configs/model_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    B, C_max, M_max, W = 4, 50, 150, 5
    node_f = torch.randn(B, C_max, 6)
    edge_i = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    node_h = torch.randn(B, W, C_max, 6)
    cnf_f = torch.randn(B, M_max, 5)
    mask = torch.ones(B, M_max, C_max, dtype=torch.bool)

    models = [
        ("TGNN-NCO", ActorCritic(cfg)),
        ("Static-GNN", StaticGNNActorCritic(cfg)),
        ("Flat-RL", FlatRLActorCritic(cfg)),
    ]

    for name, model in models:
        model.eval()
        with torch.no_grad():
            actions, log_prob, entropy, value = model.get_action_and_value(
                node_f, edge_i, node_h, cnf_f, action_mask=mask
            )
            val_only = model.get_value(node_f, edge_i, node_h, cnf_f)

        assert actions.shape == (B, M_max), f"{name} actions shape mismatch"
        assert log_prob.shape == (B,), f"{name} log_prob shape mismatch"
        assert entropy.shape == (B,), f"{name} entropy shape mismatch"
        assert value.shape == (B, 1), f"{name} value shape mismatch"
        assert val_only.shape == (B, 1), f"{name} val_only shape mismatch"
