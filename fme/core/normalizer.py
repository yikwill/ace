import dataclasses
import pathlib
from collections.abc import Iterable, Mapping
from copy import copy
from typing import Protocol

import fsspec
import numpy as np
import torch
import xarray as xr

from fme.core.device import move_tensordict_to_device
from fme.core.typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class NormalizationConfig:
    """
    Configuration for normalizing data.

    Either global_means_path and global_stds_path or explicit means and stds
    must be provided.

    Parameters:
        global_means_path: Path to a netCDF file containing global means.
        global_stds_path: Path to a netCDF file containing global stds.
        means: Mapping from variable names to means.
        stds: Mapping from variable names to stds.
        scalar_means_path: Optional netCDF of scalar means. When set with
            ``scalar_means_names``, those variables use scalar means from this
            file instead of ``global_means_path`` (e.g. keep orography / land
            masks absolute while using a spatial time-mean for other fields).
        scalar_means_names: Variable names that take means from
            ``scalar_means_path``.
        fill_nans_on_normalize: Whether to fill NaNs during normalization. If
            true, on normalization NaNs in the denormalized input become zeros in
            the normalized output.
        fill_nans_on_denormalize: Whether to fill NaNs during denormalization. If
            true, on denormalization NaNs in the normalized input become global means in
            the denormalized output.
    """

    global_means_path: str | pathlib.Path | None = None
    global_stds_path: str | pathlib.Path | None = None
    means: Mapping[str, float | list | np.ndarray] = dataclasses.field(
        default_factory=dict
    )
    stds: Mapping[str, float | list | np.ndarray] = dataclasses.field(
        default_factory=dict
    )
    scalar_means_path: str | pathlib.Path | None = None
    scalar_means_names: list[str] = dataclasses.field(default_factory=list)
    fill_nans_on_normalize: bool = False
    fill_nans_on_denormalize: bool = False

    def __post_init__(self):
        using_path = (
            self.global_means_path is not None and self.global_stds_path is not None
        )
        using_explicit = len(self.means) > 0 and len(self.stds) > 0
        if using_path and using_explicit:
            raise ValueError(
                "Cannot use both global_means_path and global_stds_path "
                "and explicit means and stds."
            )
        if not (using_path or using_explicit):
            raise ValueError(
                "Must use either global_means_path and global_stds_path "
                "or explicit means and stds."
            )
        if self.scalar_means_names and self.scalar_means_path is None:
            raise ValueError("scalar_means_names requires scalar_means_path")
        if self.scalar_means_path is not None and not self.scalar_means_names:
            raise ValueError(
                "scalar_means_path requires a non-empty scalar_means_names"
            )
        if self.scalar_means_path is not None and not using_path:
            raise ValueError(
                "scalar_means_path requires global_means_path and global_stds_path "
                "(cannot combine with explicit means/stds)"
            )

    def load(self):
        """
        Load the normalization configuration from the netCDF files.

        Updates the configuration so it no longer requires external files.
        """
        if self.global_means_path is not None and self.global_stds_path is not None:
            # convert to explicit means and stds so if the object is stored
            # and reloaded, we no longer need the netCDF files
            means = load_dict_from_netcdf(
                self.global_means_path,
                names=None,
                defaults={"x": 0.0, "y": 0.0, "z": 0.0},
            )
            if self.scalar_means_path is not None:
                means.update(
                    load_dict_from_netcdf(
                        self.scalar_means_path,
                        names=self.scalar_means_names,
                        defaults={},
                    )
                )
            stds = load_dict_from_netcdf(
                self.global_stds_path,
                names=None,
                defaults={"x": 1.0, "y": 1.0, "z": 1.0},
            )
            self.means = means
            self.stds = stds
            self.global_means_path = None
            self.global_stds_path = None
            self.scalar_means_path = None
            self.scalar_means_names = []

    def build(self, names: list[str]):
        using_path = (
            self.global_means_path is not None and self.global_stds_path is not None
        )
        if using_path:
            return get_normalizer(
                global_means_path=self.global_means_path,
                global_stds_path=self.global_stds_path,
                names=names,
                scalar_means_path=self.scalar_means_path,
                scalar_means_names=self.scalar_means_names,
                fill_nans_on_normalize=self.fill_nans_on_normalize,
                fill_nans_on_denormalize=self.fill_nans_on_denormalize,
            )
        else:
            means = {k: torch.tensor(self.means[k]) for k in names}
            stds = {k: torch.tensor(self.stds[k]) for k in names}
            return StandardNormalizer(
                means=means,
                stds=stds,
                fill_nans_on_normalize=self.fill_nans_on_normalize,
                fill_nans_on_denormalize=self.fill_nans_on_denormalize,
            )


class NormalizeFn(Protocol):
    """
    A callable that normalizes a mapping of tensors, with an option to skip
    the mean subtraction (see :meth:`StandardNormalizer.normalize`).
    """

    def __call__(
        self, tensors: TensorMapping, /, apply_mean: bool = True
    ) -> TensorDict:
        # NOTE: ``tensors`` is positional-only so implementations may name their
        # first parameter freely (e.g. test lambdas); a positional-or-keyword
        # parameter would require every implementation to use the same name.
        ...


class StandardNormalizer:
    """
    Responsible for normalizing tensors.
    """

    def __init__(
        self,
        means: TensorDict,
        stds: TensorDict,
        fill_nans_on_normalize: bool = False,
        fill_nans_on_denormalize: bool = False,
    ):
        self.means = move_tensordict_to_device(means)
        self.stds = move_tensordict_to_device(stds)
        self._names = set(means).intersection(stds)
        self._fill_nans_on_normalize = fill_nans_on_normalize
        self._fill_nans_on_denormalize = fill_nans_on_denormalize

    @property
    def fill_nans_on_normalize(self):
        return self._fill_nans_on_normalize

    @property
    def fill_nans_on_denormalize(self):
        return self._fill_nans_on_denormalize

    def normalize(self, tensors: TensorMapping, apply_mean: bool = True) -> TensorDict:
        """
        Normalize the tensors.

        Args:
            tensors: Mapping from variable names to tensors; names without
                normalization constants are dropped from the output.
            apply_mean: If False, skip the mean subtraction and divide by the
                standard deviation only, e.g. to normalize a difference of
                fields without centering it.
        """
        filtered_tensors = {k: v for k, v in tensors.items() if k in self._names}
        return _normalize(
            filtered_tensors,
            means=self.means,
            stds=self.stds,
            fill_nans=self._fill_nans_on_normalize,
            apply_mean=apply_mean,
        )

    def denormalize(self, tensors: TensorMapping) -> TensorDict:
        filtered_tensors = {k: v for k, v in tensors.items() if k in self._names}
        return _denormalize(
            filtered_tensors,
            means=self.means,
            stds=self.stds,
            fill_nans=self._fill_nans_on_denormalize,
        )

    def get_state(self):
        """
        Returns state as a serializable data structure.
        """
        return {
            "means": {k: _stat_to_serializable(v) for k, v in self.means.items()},
            "stds": {k: _stat_to_serializable(v) for k, v in self.stds.items()},
            "fill_nans_on_normalize": self._fill_nans_on_normalize,
            "fill_nans_on_denormalize": self._fill_nans_on_denormalize,
        }

    @classmethod
    def from_state(cls, state) -> "StandardNormalizer":
        """
        Loads state from a serializable data structure.
        """
        means = {
            k: torch.tensor(v, dtype=torch.float) for k, v in state["means"].items()
        }
        stds = {k: torch.tensor(v, dtype=torch.float) for k, v in state["stds"].items()}
        return cls(
            means=means,
            stds=stds,
            fill_nans_on_normalize=state.get("fill_nans_on_normalize", False),
            fill_nans_on_denormalize=state.get("fill_nans_on_denormalize", False),
        )

    def get_normalization_config(self) -> NormalizationConfig:
        return NormalizationConfig(
            means={k: _stat_to_serializable(v) for k, v in self.means.items()},
            stds={k: _stat_to_serializable(v) for k, v in self.stds.items()},
            fill_nans_on_normalize=self.fill_nans_on_normalize,
            fill_nans_on_denormalize=self.fill_nans_on_denormalize,
        )


def _normalize(
    tensors: TensorDict,
    means: TensorDict,
    stds: TensorDict,
    fill_nans: bool,
    apply_mean: bool = True,
) -> TensorDict:
    if apply_mean:
        normalized = {k: (t - means[k]) / stds[k] for k, t in tensors.items()}
    else:
        normalized = {k: t / stds[k] for k, t in tensors.items()}
    if fill_nans:
        for k, v in normalized.items():
            normalized[k] = torch.where(torch.isnan(v), torch.zeros_like(v), v)
    return normalized


def _denormalize(
    tensors: TensorDict,
    means: TensorDict,
    stds: TensorDict,
    fill_nans: bool,
) -> TensorDict:
    denormalized = {k: t * stds[k] + means[k] for k, t in tensors.items()}
    if fill_nans:
        for k, v in denormalized.items():
            # means[k] may be spatial; zeros_like broadcasts the fill to v's shape
            denormalized[k] = torch.where(
                torch.isnan(v), means[k] + torch.zeros_like(v), v
            )
    return denormalized


def get_normalizer(
    global_means_path,
    global_stds_path,
    names: list[str],
    scalar_means_path: str | pathlib.Path | None = None,
    scalar_means_names: list[str] | None = None,
    **normalizer_kwargs,
) -> StandardNormalizer:
    means = load_dict_from_netcdf(
        global_means_path, names, defaults={"x": 0.0, "y": 0.0, "z": 0.0}
    )
    if scalar_means_path is not None:
        if not scalar_means_names:
            raise ValueError("scalar_means_path requires scalar_means_names")
        override_names = [n for n in scalar_means_names if n in names]
        if override_names:
            means.update(
                load_dict_from_netcdf(scalar_means_path, override_names, defaults={})
            )
    means = {k: torch.as_tensor(v, dtype=torch.float) for k, v in means.items()}
    stds = load_dict_from_netcdf(
        global_stds_path, names, defaults={"x": 1.0, "y": 1.0, "z": 1.0}
    )
    stds = {k: torch.as_tensor(v, dtype=torch.float) for k, v in stds.items()}
    return StandardNormalizer(means=means, stds=stds, **normalizer_kwargs)


def _stat_to_serializable(tensor: torch.Tensor) -> float | list:
    """Convert a mean/std tensor to a JSON/torch-save friendly Python value."""
    array = tensor.detach().cpu().numpy()
    if array.ndim == 0:
        return float(array.item())
    return array.tolist()


def load_dict_from_netcdf(
    path: str | pathlib.Path,
    names: Iterable[str] | None,
    defaults: Mapping[str, float | np.ndarray],
) -> dict[str, float | np.ndarray]:
    """
    Load a dictionary of normalization statistics from a netCDF file.

    Values may be scalars or arrays. Spatially varying means (e.g. a time-mean
    map) are returned as float32 numpy arrays and broadcast against data tensors
    during normalize/denormalize.

    Args:
        path: Path to the netCDF file.
        names: List of variable names to load. If None, all data variables in the
            netCDF file are loaded (coordinates are excluded).
        defaults: Dictionary of default values for each variable, if not found
            in the netCDF file.
    """
    with fsspec.open(path, "rb") as f:
        ds = xr.load_dataset(f, mask_and_scale=False)

    result: dict[str, float | np.ndarray] = {}
    if names is None:
        # data_vars omits lat/lon coordinates; keep spatial data vars (time means)
        names = set(ds.data_vars).union(defaults.keys())
    for c in names:
        if c in ds.variables:
            values = np.asarray(ds.variables[c].values)
            if values.ndim == 0:
                result[c] = float(values.item())
            else:
                result[c] = values.astype(np.float32, copy=False)
        elif c in defaults:
            default = defaults[c]
            result[c] = float(default) if np.ndim(default) == 0 else np.asarray(default)
        else:
            raise ValueError(f"Variable {c} not found in {path}")
    ds.close()
    return result


def _combine_normalizers(
    base_normalizer: StandardNormalizer,
    override_normalizer: StandardNormalizer,
) -> StandardNormalizer:
    """
    Combine two normalizers by overwriting the base normalizer values that are
    present in the override normalizer.

    NaN-filling behavior is inherited from the base normalizer.
    """
    means, stds = copy(base_normalizer.means), copy(base_normalizer.stds)
    means.update(override_normalizer.means)
    stds.update(override_normalizer.stds)
    return StandardNormalizer(
        means=means,
        stds=stds,
        fill_nans_on_normalize=base_normalizer.fill_nans_on_normalize,
        fill_nans_on_denormalize=base_normalizer.fill_nans_on_denormalize,
    )


@dataclasses.dataclass
class NetworkAndLossNormalizationConfig:
    """
    Combined configuration for network and loss normalization.

    Allows loss normalization to be defined as equal to the network
    normalization, apart from a set of residual-scaled variables.

    Parameters:
        network: The normalization configuration for the network.
        loss: The normalization configuration for the loss. Default is to
            use the network configuration, except for residual-scaled variables
            which instead use the residual configuration if given.
        residual: The normalization configuration for residuals. Cannot be
            provided if loss normalization is also provided.
    """

    network: NormalizationConfig
    loss: NormalizationConfig | None = None
    residual: NormalizationConfig | None = None

    def __post_init__(self):
        if self.loss is not None and self.residual is not None:
            raise ValueError("Cannot provide both loss and residual normalization.")

    def get_network_normalizer(self, names: list[str]) -> StandardNormalizer:
        return self.network.build(names=names)

    def get_loss_normalizer(
        self,
        names: list[str],
        residual_scaled_names: list[str],
    ) -> StandardNormalizer:
        if self.loss is not None:
            return self.loss.build(names=names)
        elif self.residual is not None:
            return _combine_normalizers(
                base_normalizer=self.network.build(names=names),
                override_normalizer=self.residual.build(names=residual_scaled_names),
            )
        else:
            return self.network.build(names=names)

    def load(self):
        self.network.load()
        if self.loss is not None:
            self.loss.load()
        if self.residual is not None:
            self.residual.load()
