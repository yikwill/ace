import warnings

import pytest
import torch
import torch.nn as nn

import fme
from fme.core.models.spherical_resnet import SphericalResNet
from fme.core.models.spherical_unet import s2unet as s2unet_mod

SMALL_IMG = (32, 64)


def _small_resnet(
    *,
    img_size: tuple[int, int] = SMALL_IMG,
    in_chans: int = 5,
    out_chans: int = 3,
    embed_dim: int = 8,
    depth: int = 2,
    theta_cutoff: float | None = None,
) -> SphericalResNet:
    return SphericalResNet(
        img_size=img_size,
        in_chans=in_chans,
        out_chans=out_chans,
        embed_dim=embed_dim,
        depth=depth,
        theta_cutoff=theta_cutoff,
    )


def test_forward_shape_small():
    model = _small_resnet().to(fme.get_device())
    x = torch.randn(2, 5, *SMALL_IMG, device=fme.get_device())
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        out = model(x)
    assert out.shape == (2, 3, *SMALL_IMG)


def test_gradient_flows():
    model = _small_resnet().to(fme.get_device())
    x = torch.randn(1, 5, *SMALL_IMG, device=fme.get_device(), requires_grad=True)
    out = model(x)
    out.mean().backward()
    assert x.grad is not None
    assert any(p.grad is not None for p in model.parameters())


def test_theta_cutoff_override_reaches_all_disco_convs(monkeypatch):
    """Explicit theta_cutoff must be shared by stem and every ConvNeXt DISCO layer."""
    captured: list[float] = []

    class _CaptureDisco(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured.append(kwargs["theta_cutoff"])
            self.out_channels = args[1]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.new_zeros(x.shape[0], self.out_channels, *x.shape[-2:])

    monkeypatch.setattr(s2unet_mod, "DiscreteContinuousConvS2", _CaptureDisco)
    override = 0.25
    _ = _small_resnet(depth=2, theta_cutoff=override)
    # stem + 2 blocks
    assert captured == [override, override, override]


def test_resolve_theta_cutoff_prefers_override():
    heuristic = s2unet_mod._compute_cutoff_radius(32, (3, 3), "harmonic")
    assert s2unet_mod._resolve_theta_cutoff(32, (3, 3), "harmonic", None) == heuristic
    assert s2unet_mod._resolve_theta_cutoff(32, (3, 3), "harmonic", 0.4) == 0.4


@pytest.mark.gpu
def test_forward_shape_era5():
    model = SphericalResNet(
        img_size=(180, 360),
        in_chans=44,
        out_chans=16,
        embed_dim=32,
        depth=2,
    ).to(fme.get_device())
    x = torch.randn(1, 44, 180, 360, device=fme.get_device())
    out = model(x)
    assert out.shape == (1, 16, 180, 360)
