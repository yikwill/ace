import dataclasses

import torch

import fme
from fme.ace.registry.spherical_resnet import (
    NoiseConditionedSphericalResNetBuilder,
    SphericalResNetBuilder,
)
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.registry import ModuleSelector

IMG_SHAPE = (32, 64)


def _get_dataset_info() -> DatasetInfo:
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
        all_labels=set(),
    )


def test_spherical_resnet_is_registered():
    assert "SphericalResNet" in ModuleSelector.get_available_types()
    assert "NoiseConditionedSphericalResNet" in ModuleSelector.get_available_types()


def test_spherical_resnet_build_and_forward():
    n_in, n_out = 5, 3
    dataset_info = _get_dataset_info()
    module = (
        SphericalResNetBuilder(embed_dim=8, depth=2)
        .build(n_in, n_out, dataset_info)
        .to(fme.get_device())
    )
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, n_out, *IMG_SHAPE)


def test_noise_conditioned_spherical_resnet_via_selector():
    selector = ModuleSelector(
        type="NoiseConditionedSphericalResNet",
        config=dataclasses.asdict(
            NoiseConditionedSphericalResNetBuilder(
                embed_dim=8, depth=2, noise_embed_dim=0
            )
        ),
    )
    dataset_info = _get_dataset_info()
    module = selector.build(
        n_in_channels=5, n_out_channels=3, dataset_info=dataset_info
    ).to(fme.get_device())
    x = torch.randn(2, 5, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, 3, *IMG_SHAPE)


def test_builder_passes_theta_cutoff():
    dataset_info = _get_dataset_info()
    module = SphericalResNetBuilder(embed_dim=8, depth=2, theta_cutoff=0.3).build(
        5, 3, dataset_info
    )
    assert module.theta_cutoff == 0.3
