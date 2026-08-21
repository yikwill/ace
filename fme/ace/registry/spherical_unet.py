import dataclasses
from typing import Literal

from torch import nn
from torch_harmonics import InverseRealSHT

from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.ace.registry.stochastic_sfno import NoiseConditionedModel
from fme.core.dataset_info import DatasetInfo
from fme.core.models.spherical_unet import (
    DiscoBasisSpec,
    SphericalUNet,
    SphericalUNetContextConfig,
)
from fme.core.models.spherical_unet.s2unet import BasisNormMode, FilterBasisType


@dataclasses.dataclass
class DiscoBasisSpecConfig:
    basis_type: Literal[
        "harmonic",
        "piecewise linear",
        "zernike",
        "fourier-bessel",
        "morlet",
        "isotropic morlet",
    ] = "harmonic"
    kernel_shape: list[int] = dataclasses.field(default_factory=lambda: [3, 3])


def _filter_bases_from_config(
    filter_bases: list[DiscoBasisSpecConfig] | None,
) -> list[DiscoBasisSpec] | None:
    if filter_bases is None:
        return None
    return [
        DiscoBasisSpec(
            basis_type=spec.basis_type,
            kernel_shape=tuple(spec.kernel_shape),
        )
        for spec in filter_bases
    ]


def _kernel_shape_pair(kernel_shape: list[int]) -> tuple[int, int]:
    if len(kernel_shape) == 1:
        return (kernel_shape[0], kernel_shape[0])
    return (kernel_shape[0], kernel_shape[1])


def _build_spherical_unet(
    *,
    n_in_channels: int,
    n_out_channels: int,
    img_shape: tuple[int, int],
    context_config: SphericalUNetContextConfig | None,
    embed_dims: list[int],
    depths: list[int],
    scale_factor: int,
    grid: str,
    grid_internal: str,
    activation_function: Literal["relu", "gelu", "identity"],
    kernel_shape: tuple[int, int],
    filter_basis_type: FilterBasisType,
    filter_bases: list[DiscoBasisSpec] | None,
    filter_basis_norm_mode: BasisNormMode,
    transform_skip: bool,
    path_drop_rate: float,
    mlp_drop_rate: float,
    downsampling_mode: str,
    upsampling_mode: str,
    mlp_ratio: float = 2.0,
    layer_scale: bool = True,
    layer_scale_init: float = 0.1,
) -> SphericalUNet:
    return SphericalUNet(
        img_size=img_shape,
        grid=grid,
        grid_internal=grid_internal,
        in_chans=n_in_channels,
        out_chans=n_out_channels,
        embed_dims=embed_dims,
        depths=depths,
        scale_factor=scale_factor,
        activation_function=activation_function,
        kernel_shape=kernel_shape,
        filter_basis_type=filter_basis_type,
        filter_bases=filter_bases,
        filter_basis_norm_mode=filter_basis_norm_mode,
        transform_skip=transform_skip,
        path_drop_rate=path_drop_rate,
        mlp_drop_rate=mlp_drop_rate,
        downsampling_mode=downsampling_mode,
        upsampling_mode=upsampling_mode,
        context_config=context_config,
        mlp_ratio=mlp_ratio,
        layer_scale=layer_scale,
        layer_scale_init=layer_scale_init,
    )


@ModuleSelector.register("SphericalUNet")
@dataclasses.dataclass
class SphericalUNetBuilder(ModuleConfig):
    """Configuration for the spherical U-Net (unconditional)."""

    embed_dims: list[int] = dataclasses.field(default_factory=lambda: [64, 128, 256])
    depths: list[int] = dataclasses.field(default_factory=lambda: [2, 2, 2])
    scale_factor: int = 2
    grid: Literal["legendre-gauss", "equiangular"] = "legendre-gauss"
    grid_internal: Literal["legendre-gauss", "equiangular"] = "legendre-gauss"
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
    transform_skip: bool = False
    path_drop_rate: float = 0.0
    mlp_drop_rate: float = 0.0
    downsampling_mode: Literal["conv", "bilinear"] = "conv"
    upsampling_mode: Literal["bilinear", "conv"] = "bilinear"
    mlp_ratio: float = 2.0
    layer_scale: bool = True
    layer_scale_init: float = 0.1

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ) -> nn.Module:
        if len(dataset_info.all_labels) > 0:
            raise ValueError("SphericalUNet does not support labels")
        context_config = SphericalUNetContextConfig()
        return _build_spherical_unet(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            img_shape=dataset_info.img_shape,
            context_config=context_config,
            embed_dims=self.embed_dims,
            depths=self.depths,
            scale_factor=self.scale_factor,
            grid=self.grid,
            grid_internal=self.grid_internal,
            activation_function=self.activation_function,
            kernel_shape=_kernel_shape_pair(self.kernel_shape),
            filter_basis_type=self.filter_basis_type,
            filter_bases=_filter_bases_from_config(self.filter_bases),
            filter_basis_norm_mode=self.filter_basis_norm_mode,
            transform_skip=self.transform_skip,
            path_drop_rate=self.path_drop_rate,
            mlp_drop_rate=self.mlp_drop_rate,
            downsampling_mode=self.downsampling_mode,
            upsampling_mode=self.upsampling_mode,
            mlp_ratio=self.mlp_ratio,
            layer_scale=self.layer_scale,
            layer_scale_init=self.layer_scale_init,
        )


@ModuleSelector.register("NoiseConditionedSphericalUNet")
@dataclasses.dataclass
class NoiseConditionedSphericalUNetBuilder(ModuleConfig):
    """Noise-conditioned spherical U-Net wrapped in ``NoiseConditionedModel``."""

    embed_dims: list[int] = dataclasses.field(default_factory=lambda: [64, 128, 256])
    depths: list[int] = dataclasses.field(default_factory=lambda: [2, 2, 2])
    scale_factor: int = 2
    grid: Literal["legendre-gauss", "equiangular"] = "legendre-gauss"
    grid_internal: Literal["legendre-gauss", "equiangular"] = "legendre-gauss"
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
    transform_skip: bool = False
    path_drop_rate: float = 0.0
    mlp_drop_rate: float = 0.0
    downsampling_mode: Literal["conv", "bilinear"] = "conv"
    upsampling_mode: Literal["bilinear", "conv"] = "bilinear"
    mlp_ratio: float = 2.0
    layer_scale: bool = True
    layer_scale_init: float = 0.1
    noise_embed_dim: int = 256
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
        context_config = SphericalUNetContextConfig(
            embed_dim_labels=effective_label_dim,
            embed_dim_noise=self.noise_embed_dim,
        )
        net = _build_spherical_unet(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            img_shape=dataset_info.img_shape,
            context_config=context_config,
            embed_dims=self.embed_dims,
            depths=self.depths,
            scale_factor=self.scale_factor,
            grid=self.grid,
            grid_internal=self.grid_internal,
            activation_function=self.activation_function,
            kernel_shape=_kernel_shape_pair(self.kernel_shape),
            filter_basis_type=self.filter_basis_type,
            filter_bases=_filter_bases_from_config(self.filter_bases),
            filter_basis_norm_mode=self.filter_basis_norm_mode,
            transform_skip=self.transform_skip,
            path_drop_rate=self.path_drop_rate,
            mlp_drop_rate=self.mlp_drop_rate,
            downsampling_mode=self.downsampling_mode,
            upsampling_mode=self.upsampling_mode,
            mlp_ratio=self.mlp_ratio,
            layer_scale=self.layer_scale,
            layer_scale_init=self.layer_scale_init,
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
