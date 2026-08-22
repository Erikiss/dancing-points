import pytest
import torch

from models.CVAEModule import CVAEModel
from models.MLPPoseMapping import OneFrameMLP
from unified_utils import model_dispatch


def test_mapping_mlp_forward_and_loss_shapes():
    model = OneFrameMLP([6, 8, 3], bn=False, dropout=0.0)
    inputs = torch.randn(4, 6)
    targets = torch.randn(4, 3)

    outputs, info = model(inputs)
    losses, _ = model.learn(inputs, targets)

    assert outputs.shape == (4, 3)
    assert info is None
    assert losses['rec'].ndim == 0
    assert torch.isfinite(losses['rec'])


def test_cvae_eval_is_deterministic_and_shape_stable():
    torch.manual_seed(7)
    model = CVAEModel(
        input_dim=6,
        output_dim=4,
        encoder_dim=12,
        estimator_dim=10,
        decoder_dim=12,
        codebook_channels=2,
        codebook_dim=3,
        dropout=0.0,
        activation_codebook='softmax',
    )
    inputs = torch.randn(5, 6)

    model.eval()
    first, first_info = model(inputs)
    second, second_info = model(inputs)

    assert first.shape == (5, 4)
    assert torch.equal(first, second)
    assert torch.equal(first_info['mu'], second_info['mu'])


def test_cvae_training_smoke():
    model = CVAEModel(6, 4, 12, 10, 12, 2, 3, 0.0, 'softmax')
    inputs = torch.randn(5, 6)
    targets = torch.randn(5, 4)

    losses, outputs = model.learn(inputs, targets)

    assert outputs.shape == targets.shape
    assert set(losses) == {'rec', 'matching'}
    assert all(torch.isfinite(value) for value in losses.values())


def test_cvae_rejects_unknown_codebook_activation():
    with pytest.raises(ValueError, match='activation_codebook'):
        CVAEModel(6, 4, 12, 10, 12, 2, 3, 0.0, 'mystery')


def test_model_dispatch_rejects_unknown_model():
    with pytest.raises(ValueError, match='Unknown model_type'):
        model_dispatch(['--model_type=unknown'])


def test_model_dispatch_preserves_help_for_selected_parser():
    model, remaining, model_arg = model_dispatch(['--model_type=mlp_pose', '--help'])

    assert model.__name__ == 'models.MLPPoseMapping'
    assert remaining == ['--help']
    assert model_arg == '--model_type=mlp_pose'
