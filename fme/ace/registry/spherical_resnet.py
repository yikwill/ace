import dataclasses
from typing import Literal

from torch import nn
from torch_harmonics import InverseRealSHT

from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.ace.registry.spherical_unet import (
    DiscoBasisSpecConfig,
    _filter_bases_from_config,
    _kernel_shape_pair,
)
from fme.ace.registry.stochastic_sfno import NoiseConditionedModel
from fme.core.dataset_info import DatasetInfo
from fme.core.models.spherical_resnet import (
    SphericalResNet,
    SphericalResNetContextConfig,
)
from fme.core.models.spherical_unet.s2unet import BasisNormMode, FilterBasisType


def _build_spherical_resnet(
    *,
    n_in_channels: int,
    n_out_channels: int,
    img_shape: tuple[int, int],
    context_config: SphericalResNetContextConfig | None,
    embed_dim: int,
    depth: int,
    grid: str,
    activation_function: Literal["relu", "gelu", "identity"],
    kernel_shape: tuple[int, int],
    filter_basis_type: FilterBasisType,
    filter_bases: list | None,
    filter_basis_norm_mode: BasisNormMode,
    path_drop_rate: float,
    mlp_drop_rate: float,
    mlp_ratio: float,
    layer_scale: bool,
    layer_scale_init: float,
    theta_cutoff: float | None,
) -> SphericalResNet:
    return SphericalResNet(
        img_size=img_shape,
        grid=grid,
        in_chans=n_in_channels,
        out_chans=n_out_channels,
        embed_dim=embed_dim,
        depth=depth,
        activation_function=activation_function,
        kernel_shape=kernel_shape,
        filter_basis_type=filter_basis_type,
        filter_bases=filter_bases,
        filter_basis_norm_mode=filter_basis_norm_mode,
        path_drop_rate=path_drop_rate,
        mlp_drop_rate=mlp_drop_rate,
        context_config=context_config,
        mlp_ratio=mlp_ratio,
        layer_scale=layer_scale,
        layer_scale_init=layer_scale_init,
        theta_cutoff=theta_cutoff,
    )


@ModuleSelector.register("SphericalResNet")
@dataclasses.dataclass
class SphericalResNetBuilder(ModuleConfig):
    """Configuration for the spherical ResNet (unconditional)."""

    embed_dim: int = 64
    depth: int = 2
    grid: Literal["legendre-gauss", "equiangular"] = "legendre-gauss"
    activation_function: Literal["relu", "gelu", "identity"] = "gelu"
    kernel_shape: list[int] = dataclasses.field(default_factory=lambda: [3, 3])
    filter_basis_type: Literal[
        "harmonic",
        "piecewise linear",
        "zernike",
        "fourier-bessel",
        "morlet",
        "isotropic morlet",
    ] = "harmonic"
    filter_bases: list[DiscoBasisSpecConfig] | None = None
    filter_basis_norm_mode: Literal[
        "nodal", "modal", "mean", "support", "geometric", "none"
    ] = "nodal"
    path_drop_rate: float = 0.0
    mlp_drop_rate: float = 0.0
    mlp_ratio: float = 2.0
    layer_scale: bool = True
    layer_scale_init: float = 0.1
    # Radians; applied to every DISCO conv. None = kernel_shape/nlat heuristic.
    theta_cutoff: float | None = None

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ) -> nn.Module:
        if len(dataset_info.all_labels) > 0:
            raise ValueError("SphericalResNet does not support labels")
        return _build_spherical_resnet(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            img_shape=dataset_info.img_shape,
            context_config=SphericalResNetContextConfig(),
            embed_dim=self.embed_dim,
            depth=self.depth,
            grid=self.grid,
            activation_function=self.activation_function,
            kernel_shape=_kernel_shape_pair(self.kernel_shape),
            filter_basis_type=self.filter_basis_type,
            filter_bases=_filter_bases_from_config(self.filter_bases),
            filter_basis_norm_mode=self.filter_basis_norm_mode,
            path_drop_rate=self.path_drop_rate,
            mlp_drop_rate=self.mlp_drop_rate,
            mlp_ratio=self.mlp_ratio,
            layer_scale=self.layer_scale,
            layer_scale_init=self.layer_scale_init,
            theta_cutoff=self.theta_cutoff,
        )


@ModuleSelector.register("NoiseConditionedSphericalResNet")
@dataclasses.dataclass
class NoiseConditionedSphericalResNetBuilder(ModuleConfig):
    """Noise-conditioned spherical ResNet wrapped in ``NoiseConditionedModel``."""

    embed_dim: int = 64
    depth: int = 2
    grid: Literal["legendre-gauss", "equiangular"] = "legendre-gauss"
    activation_function: Literal["relu", "gelu", "identity"] = "gelu"
    kernel_shape: list[int] = dataclasses.field(default_factory=lambda: [3, 3])
    filter_basis_type: Literal[
        "harmonic",
        "piecewise linear",
        "zernike",
        "fourier-bessel",
        "morlet",
        "isotropic morlet",
    ] = "harmonic"
    filter_bases: list[DiscoBasisSpecConfig] | None = None
    filter_basis_norm_mode: Literal[
        "nodal", "modal", "mean", "support", "geometric", "none"
    ] = "nodal"
    path_drop_rate: float = 0.0
    mlp_drop_rate: float = 0.0
    mlp_ratio: float = 2.0
    layer_scale: bool = True
    layer_scale_init: float = 0.1
    # Radians; applied to every DISCO conv. None = kernel_shape/nlat heuristic.
    theta_cutoff: float | None = None
    noise_embed_dim: int = 0
    label_embed_dim: int = 0
    noise_type: Literal["isotropic", "gaussian"] = "gaussian"

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ) -> nn.Module:
        n_labels = len(dataset_info.all_labels)
        label_embed_dim = self.label_embed_dim
        effective_label_dim = label_embed_dim if label_embed_dim > 0 else n_labels
        context_config = SphericalResNetContextConfig(
            embed_dim_labels=effective_label_dim,
            embed_dim_noise=self.noise_embed_dim,
        )
        net = _build_spherical_resnet(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            img_shape=dataset_info.img_shape,
            context_config=context_config,
            embed_dim=self.embed_dim,
            depth=self.depth,
            grid=self.grid,
            activation_function=self.activation_function,
            kernel_shape=_kernel_shape_pair(self.kernel_shape),
            filter_basis_type=self.filter_basis_type,
            filter_bases=_filter_bases_from_config(self.filter_bases),
            filter_basis_norm_mode=self.filter_basis_norm_mode,
            path_drop_rate=self.path_drop_rate,
            mlp_drop_rate=self.mlp_drop_rate,
            mlp_ratio=self.mlp_ratio,
            layer_scale=self.layer_scale,
            layer_scale_init=self.layer_scale_init,
            theta_cutoff=self.theta_cutoff,
        )
        if self.noise_type == "isotropic" and self.noise_embed_dim > 0:
            inverse_sht = InverseRealSHT(*dataset_info.img_shape, grid=self.grid)
            lmax = inverse_sht.lmax
            mmax = inverse_sht.mmax
        else:
            inverse_sht = None
            lmax = 0
            mmax = 0
        return NoiseConditionedModel(
            net,
            img_shape=dataset_info.img_shape,
            embed_dim_noise=self.noise_embed_dim,
            n_labels=n_labels,
            label_embed_dim=label_embed_dim,
            inverse_sht=inverse_sht,
            lmax=lmax,
            mmax=mmax,
        )
