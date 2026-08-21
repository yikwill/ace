import dataclasses
import unittest.mock
from typing import Any

import torch

import fme
from fme.ace.registry.spherical_unet import (
    DiscoBasisSpecConfig,
    NoiseConditionedSphericalUNetBuilder,
    SphericalUNetBuilder,
)
from fme.ace.registry.stochastic_sfno import NoiseConditionedModel
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.models.conditional_sfno.layers import Context
from fme.core.registry import ModuleSelector

IMG_SHAPE = (32, 64)
SMALL_EMBED = [8, 16, 32]


def _get_dataset_info(all_labels: set[str] | None = None) -> DatasetInfo:
    device = fme.get_device()
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.zeros(IMG_SHAPE[0], device=device),
            lon=torch.zeros(IMG_SHAPE[1], device=device),
        ),
        vertical_coordinate=HybridSigmaPressureCoordinate(
            ak=torch.arange(7, device=device),
            bk=torch.arange(7, device=device),
        ),
        all_labels=all_labels,
    )


def _builder(**kwargs: Any) -> SphericalUNetBuilder:
    defaults: dict[str, Any] = dict(
        embed_dims=SMALL_EMBED,
        depths=[1, 1, 1],
        path_drop_rate=0.0,
        mlp_drop_rate=0.0,
    )
    defaults.update(kwargs)
    return SphericalUNetBuilder(**defaults)


def _nc_builder(**kwargs: Any) -> NoiseConditionedSphericalUNetBuilder:
    defaults: dict[str, Any] = dict(
        embed_dims=SMALL_EMBED,
        depths=[1, 1, 1],
        path_drop_rate=0.0,
        mlp_drop_rate=0.0,
        noise_embed_dim=8,
    )
    defaults.update(kwargs)
    return NoiseConditionedSphericalUNetBuilder(**defaults)


def test_spherical_unet_is_registered():
    assert "SphericalUNet" in ModuleSelector.get_available_types()


def test_spherical_unet_build_and_forward():
    n_in, n_out = 5, 3
    dataset_info = _get_dataset_info()
    module = _builder().build(n_in, n_out, dataset_info).to(fme.get_device())
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, n_out, *IMG_SHAPE)


def test_spherical_unet_via_selector():
    selector = ModuleSelector(
        type="SphericalUNet",
        config=dataclasses.asdict(_builder()),
    )
    dataset_info = _get_dataset_info()
    module = selector.build(
        n_in_channels=5, n_out_channels=3, dataset_info=dataset_info
    ).to(fme.get_device())
    x = torch.randn(2, 5, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, 3, *IMG_SHAPE)


def test_nc_spherical_unet_is_registered():
    assert "NoiseConditionedSphericalUNet" in ModuleSelector.get_available_types()


def test_nc_spherical_unet_returns_noise_conditioned_model():
    n_in, n_out = 5, 3
    dataset_info = _get_dataset_info()
    module = _nc_builder().build(n_in, n_out, dataset_info)
    assert isinstance(module, NoiseConditionedModel)


def test_nc_spherical_unet_via_selector():
    n_in, n_out = 5, 3
    dataset_info = _get_dataset_info()
    selector = ModuleSelector(
        type="NoiseConditionedSphericalUNet",
        config=dataclasses.asdict(_nc_builder()),
    )
    module = selector.build(
        n_in_channels=n_in, n_out_channels=n_out, dataset_info=dataset_info
    ).to(fme.get_device())
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, n_out, *IMG_SHAPE)


def test_nc_spherical_unet_noise_embed_dim_zero():
    n_in, n_out = 5, 3
    dataset_info = _get_dataset_info()
    module = (
        _nc_builder(noise_embed_dim=0)
        .build(n_in, n_out, dataset_info)
        .to(fme.get_device())
    )
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, n_out, *IMG_SHAPE)


def test_nc_spherical_unet_noise_divergence():
    """After one optimizer step, resampled noise yields distinct outputs."""
    n_in, n_out = 4, 2
    dataset_info = _get_dataset_info()
    module = _nc_builder().build(n_in, n_out, dataset_info).to(fme.get_device())
    module.train()
    optimizer = torch.optim.SGD(module.parameters(), lr=1.0)

    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())

    out = module(x)
    out.sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    with torch.no_grad():
        out1 = module(x)
        out2 = module(x)
    assert (
        out1 - out2
    ).abs().max().item() > 1e-4, "Expected noise-dependence after step"


def test_noise_conditioned_spherical_unet_conditioning():
    mock_net = unittest.mock.MagicMock()
    img_shape = (32, 64)
    n_noise = 8
    model = NoiseConditionedModel(
        conditional_model=mock_net,
        img_shape=img_shape,
        embed_dim_noise=n_noise,
        n_labels=0,
        label_embed_dim=0,
    )
    batch_size = 2
    x = torch.randn(batch_size, 3, *img_shape)
    _ = model(x)
    mock_net.assert_called()
    args, _ = mock_net.call_args
    context = args[1]
    assert isinstance(context, Context)
    assert context.noise is not None
    assert context.noise.shape == (batch_size, n_noise, *img_shape)


def test_builder_filter_bases():
    n_in, n_out = 5, 3
    dataset_info = _get_dataset_info()
    module = (
        _builder(
            filter_bases=[
                DiscoBasisSpecConfig(basis_type="harmonic", kernel_shape=[3, 3]),
                DiscoBasisSpecConfig(basis_type="zernike", kernel_shape=[4]),
                DiscoBasisSpecConfig(basis_type="isotropic morlet", kernel_shape=[8]),
            ]
        )
        .build(n_in, n_out, dataset_info)
        .to(fme.get_device())
    )
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, n_out, *IMG_SHAPE)


def test_selector_filter_bases():
    n_in, n_out = 5, 3
    dataset_info = _get_dataset_info()
    config = dataclasses.asdict(_builder())
    config["filter_bases"] = [
        {"basis_type": "harmonic", "kernel_shape": [3, 3]},
        {"basis_type": "zernike", "kernel_shape": [4]},
        {"basis_type": "isotropic morlet", "kernel_shape": [8]},
    ]
    selector = ModuleSelector(type="SphericalUNet", config=config)
    module = selector.build(
        n_in_channels=n_in, n_out_channels=n_out, dataset_info=dataset_info
    ).to(fme.get_device())
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, n_out, *IMG_SHAPE)
