"""Spherical-harmonic low-pass for the residual add.

The network still sees the unfiltered normalized state. The filter is applied
only to the tensors that enter the residual sum, so

- ``target: residual`` steps ``x + LP(dx)``
- ``target: skip`` steps ``LP(x) + dx``

``lmax`` is exclusive: degrees ``l < lmax`` (and ``m < lmax``) are kept.
With scalar centering and stds this matches the same truncation in physical
space, because a mean-preserving spectral truncation commutes with a scalar
affine normalization.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from torch import nn

from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed import Distributed
from fme.core.typing_ import TensorMapping

LowpassTarget = Literal["residual", "skip"]


@dataclasses.dataclass
class ResidualLowpassConfig:
    """One spherical-harmonic truncation applied to a set of residual names.

    Parameters:
        names: Prognostic names that are stepped as residuals.
        lmax: Keep spherical-harmonic degrees ``l < lmax`` (triangular, so
            ``m < lmax`` as well).
        target: ``residual`` filters the network increment before it is added.
            ``skip`` filters the normalized input copy that is added to the
            unfiltered increment. The network input is never filtered.
    """

    names: list[str]
    lmax: int
    target: LowpassTarget

    def __post_init__(self) -> None:
        if len(self.names) == 0:
            raise ValueError("residual_lowpass names must not be empty")
        if self.lmax < 1:
            raise ValueError(
                "residual_lowpass lmax must be >= 1 (degrees l < lmax), "
                f"got {self.lmax}"
            )
        if self.target not in ("residual", "skip"):
            raise ValueError(
                "residual_lowpass target must be 'residual' or 'skip', "
                f"got {self.target!r}"
            )


class SphericalHarmonicLowpass(nn.Module):
    """Truncate a lat-lon field to degrees ``l < lmax``."""

    def __init__(self, nlat: int, nlon: int, lmax: int, grid: str):
        super().__init__()
        if lmax > nlat:
            raise ValueError(
                f"residual_lowpass lmax ({lmax}) exceeds nlat ({nlat}); "
                "the grid cannot represent degrees l >= nlat"
            )
        dist = Distributed.get_instance()
        # Register the transforms so they follow .to(device) with the step.
        self.sht = dist.get_sht(nlat, nlon, lmax=lmax, mmax=lmax, grid=grid)
        self.isht = dist.get_isht(nlat, nlon, lmax=lmax, mmax=lmax, grid=grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        # torch_harmonics transforms are float32. Cast back so the residual
        # add stays in the caller's dtype (including autocast outputs).
        spectrum = self.sht(x.float())
        return self.isht(spectrum).to(dtype=dtype)


class ResidualLowpass(nn.Module):
    """Apply configured truncations to selected channels of a field dict."""

    def __init__(
        self,
        entries: Sequence[ResidualLowpassConfig],
        nlat: int,
        nlon: int,
        grid: Literal["equiangular", "legendre-gauss"],
    ):
        super().__init__()
        self._residual_by_lmax: dict[int, list[str]] = {}
        self._skip_by_lmax: dict[int, list[str]] = {}
        lmaxes: set[int] = set()
        for entry in entries:
            bucket = (
                self._residual_by_lmax
                if entry.target == "residual"
                else self._skip_by_lmax
            )
            bucket.setdefault(entry.lmax, []).extend(entry.names)
            lmaxes.add(entry.lmax)
        self.filters = nn.ModuleDict(
            {
                str(lmax): SphericalHarmonicLowpass(nlat, nlon, lmax, grid)
                for lmax in sorted(lmaxes)
            }
        )

    @classmethod
    def from_config(
        cls,
        entries: Sequence[ResidualLowpassConfig],
        dataset_info: DatasetInfo,
    ) -> "ResidualLowpass":
        coords = dataset_info.horizontal_coordinates
        if not isinstance(coords, LatLonCoordinates):
            raise ValueError(
                "residual_lowpass requires lat-lon coordinates, "
                f"got {type(coords).__name__}"
            )
        nlat, nlon = dataset_info.img_shape
        return cls(entries, nlat=nlat, nlon=nlon, grid=coords.grid)

    def filter_residual(self, fields: TensorMapping) -> dict[str, torch.Tensor]:
        """Filter names configured with ``target: residual``. Others pass through."""
        return self._filter_channels(fields, self._residual_by_lmax)

    def filter_skip(self, fields: TensorMapping) -> dict[str, torch.Tensor]:
        """Filter names configured with ``target: skip``. Others pass through."""
        return self._filter_channels(fields, self._skip_by_lmax)

    def _filter_channels(
        self,
        fields: Mapping[str, torch.Tensor],
        by_lmax: dict[int, list[str]],
    ) -> dict[str, torch.Tensor]:
        out = dict(fields)
        for lmax, names in by_lmax.items():
            present = [name for name in names if name in out]
            if not present:
                continue
            stacked = torch.stack([out[name] for name in present], dim=0)
            filtered = self.filters[str(lmax)](stacked)
            for index, name in enumerate(present):
                out[name] = filtered[index]
        return out
