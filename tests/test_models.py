import torch
import yaml
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
