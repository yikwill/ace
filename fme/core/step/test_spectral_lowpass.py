import pytest
import torch
from torch import nn
from torch_harmonics import RealSHT

import fme
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.registry import ModuleSelector
from fme.core.step.single_module import (
    ResidualPredictionConfig,
    SingleModuleStepConfig,
    step_with_adjustments,
)
from fme.core.step.spectral_lowpass import (
    ResidualLowpass,
    ResidualLowpassConfig,
    SphericalHarmonicLowpass,
)
from fme.core.testing import (
    trivial_network_and_loss_normalization,
    trivial_normalization,
)

NLAT = 32
NLON = 64
LMAX = 8


def _lowpass() -> SphericalHarmonicLowpass:
    return SphericalHarmonicLowpass(NLAT, NLON, LMAX, grid="equiangular").to(
        fme.get_device()
    )


def _config(**kwargs) -> SingleModuleStepConfig:
    return SingleModuleStepConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": nn.Identity()}),
        in_names=["a", "c"],
        out_names=["a", "b", "c"],
        normalization=trivial_network_and_loss_normalization(["a", "b", "c"]),
        **kwargs,
    )


def test_lowpass_drops_degrees_at_and_above_lmax():
    """Degrees l < lmax are unchanged; higher degrees are removed."""
    device = fme.get_device()
    field = torch.randn(2, NLAT, NLON, device=device)
    filtered = _lowpass()(field)
    sht = RealSHT(NLAT, NLON, grid="equiangular").to(device)
    original = sht(field.float())
    truncated = sht(filtered.float())
    torch.testing.assert_close(
        truncated[:, :LMAX], original[:, :LMAX], atol=1e-4, rtol=1e-4
    )
    # Transform round-trip leaves a small residual above the cutoff.
    assert truncated[:, LMAX:].abs().max().item() < 1e-3
    # A broadband field must actually change, or the truncation was a no-op.
    assert not torch.allclose(filtered, field)


def test_skip_filter_does_not_change_network_input():
    """Skip mode filters only the copy added to the residual."""
    device = fme.get_device()
    names = ["a", "b"]
    normalizer = trivial_normalization(names).build(names)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    inputs = {
        name: torch.randn(1, NLAT, NLON, generator=gen).to(device) for name in names
    }
    deltas = {
        name: torch.randn(1, NLAT, NLON, generator=gen).to(device) for name in names
    }
    lowpass = ResidualLowpass(
        [ResidualLowpassConfig(names=["a"], lmax=LMAX, target="skip")],
        nlat=NLAT,
        nlon=NLON,
        grid="equiangular",
    ).to(device)
    seen: dict[str, torch.Tensor] = {}

    def network_calls(input_norm):
        seen["a"] = input_norm["a"].detach().clone()
        return dict(deltas)

    output = step_with_adjustments(
        input=inputs,
        next_step_input_data={},
        network_calls=network_calls,
        normalizer=normalizer,
        corrector=None,
        ocean=None,
        residual_names=names,
        residual_lowpass=lowpass,
    ).output
    torch.testing.assert_close(seen["a"], inputs["a"])
    expected_a = lowpass.filter_skip(inputs)["a"] + deltas["a"]
    torch.testing.assert_close(output["a"], expected_a)
    # Unlisted prognostic keeps the plain residual add.
    torch.testing.assert_close(output["b"], inputs["b"] + deltas["b"])


def test_residual_filter_truncates_increment_only():
    device = fme.get_device()
    names = ["a"]
    normalizer = trivial_normalization(names).build(names)
    inputs = {"a": torch.randn(1, NLAT, NLON, device=device)}
    deltas = {"a": torch.randn(1, NLAT, NLON, device=device)}
    lowpass = ResidualLowpass(
        [ResidualLowpassConfig(names=["a"], lmax=LMAX, target="residual")],
        nlat=NLAT,
        nlon=NLON,
        grid="equiangular",
    ).to(device)

    def network_calls(input_norm):
        torch.testing.assert_close(input_norm["a"], inputs["a"])
        return dict(deltas)

    output = step_with_adjustments(
        input=inputs,
        next_step_input_data={},
        network_calls=network_calls,
        normalizer=normalizer,
        corrector=None,
        ocean=None,
        residual_names=["a"],
        residual_lowpass=lowpass,
    ).output
    expected = inputs["a"] + lowpass.filter_residual(deltas)["a"]
    torch.testing.assert_close(output["a"], expected)


def test_lowpass_requires_residual_prediction():
    with pytest.raises(
        ValueError, match="residual_lowpass requires residual_prediction"
    ):
        _config(residual_lowpass=[ResidualLowpassConfig(["a"], 4, "residual")])


def test_lowpass_name_must_be_in_the_residual_add():
    with pytest.raises(ValueError, match="not stepped as a residual"):
        _config(
            residual_prediction=ResidualPredictionConfig(names=["a"]),
            residual_lowpass=[ResidualLowpassConfig(["c"], 4, "skip")],
        )


def test_lowpass_from_config_uses_lat_lon_grid():
    nlat, nlon = 16, 32
    lat = torch.linspace(-89.0, 89.0, nlat)
    lon = torch.linspace(0.0, 360.0 - 360.0 / nlon, nlon)
    info = DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(lon=lon, lat=lat),
    )
    lowpass = ResidualLowpass.from_config(
        [ResidualLowpassConfig(names=["a"], lmax=4, target="residual")],
        info,
    )
    filtered = lowpass.filter_residual(
        {"a": torch.randn(1, nlat, nlon, device=fme.get_device())}
    )
    assert filtered["a"].shape == (1, nlat, nlon)
