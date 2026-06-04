import torch

from curiosity_ppo.networks.encoders import NatureDQNEncoder, CrafterEncoder


def test_nature_dqn_encoder_output_shape():
    """NatureDQNEncoder: 输入 (8, 4, 84, 84) -> 输出 (8, 512)."""
    encoder = NatureDQNEncoder(in_channels=4, out_dim=512)
    obs = torch.randn(8, 4, 84, 84)
    out = encoder(obs)
    assert out.shape == (8, 512), f"expected (8, 512), got {tuple(out.shape)}"


def test_crafter_encoder_output_shape():
    """CrafterEncoder: 输入 (8, 3, 64, 64) -> 输出 (8, 288)."""
    encoder = CrafterEncoder(in_channels=3, out_dim=288)
    obs = torch.randn(8, 3, 64, 64)
    out = encoder(obs)
    assert out.shape == (8, 288), f"expected (8, 288), got {tuple(out.shape)}"


def test_nature_dqn_encoder_no_nan():
    encoder = NatureDQNEncoder(in_channels=4, out_dim=512)
    obs = torch.randn(8, 4, 84, 84)
    out = encoder(obs)
    assert not torch.isnan(out).any(), "NatureDQNEncoder output contains NaN"


def test_crafter_encoder_no_nan():
    encoder = CrafterEncoder(in_channels=3, out_dim=288)
    obs = torch.randn(8, 3, 64, 64)
    out = encoder(obs)
    assert not torch.isnan(out).any(), "CrafterEncoder output contains NaN"


def test_encoders_run_in_eval_mode():
    """编码器在 eval 模式下前向应稳定且确定性 (dropout/batchnorm 无关)."""
    encoder = NatureDQNEncoder(in_channels=4, out_dim=512)
    encoder.eval()
    obs = torch.randn(4, 4, 84, 84)
    with torch.no_grad():
        out1 = encoder(obs)
        out2 = encoder(obs)
    assert torch.allclose(out1, out2), "encoder forward not deterministic in eval mode"
