# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Ported from torch-harmonics/torch_harmonics/examples/models/s2unet.py with ACE
# DropPath and optional ConditionalLayerNorm noise conditioning.

from __future__ import annotations

import dataclasses
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_harmonics import (
    DiscreteContinuousConvS2,
    DiscreteContinuousConvTransposeS2,
    ResampleS2,
)

from fme.ace.models.modulus.layers import DropPath
from fme.core.models.conditional_sfno.layers import (
    MLP,
    ConditionalLayerNorm,
    Context,
    ContextConfig,
)

FilterBasisType = Literal[
    "harmonic",
    "piecewise linear",
    "zernike",
    "fourier-bessel",
    "morlet",
    "isotropic morlet",
]
BasisNormMode = Literal["nodal", "modal", "mean", "support", "geometric", "none"]


@dataclasses.dataclass
class DiscoBasisSpec:
    basis_type: FilterBasisType = "harmonic"
    kernel_shape: tuple[int, ...] = (3, 3)


@dataclasses.dataclass
class SphericalUNetContextConfig:
    """CLN conditioning for SphericalUNet.

    Supports noise, labels, and scalar inputs only (no positional embedding).
    """

    embed_dim_scalar: int = 0
    embed_dim_labels: int = 0
    embed_dim_noise: int = 0

    def to_context_config(self) -> ContextConfig:
        return ContextConfig(
            embed_dim_scalar=self.embed_dim_scalar,
            embed_dim_labels=self.embed_dim_labels,
            embed_dim_noise=self.embed_dim_noise,
            embed_dim_pos=0,
        )


_THETA_CUTOFF_FACTOR: dict[str, float] = {
    "piecewise linear": 0.5,
    "harmonic": 0.5,
    "morlet": 0.5,
    "isotropic morlet": 0.5,
    "zernike": math.sqrt(2.0),
    "fourier-bessel": 0.5,
}


def _normalize_kernel_shape(
    kernel_shape: int | list[int] | tuple[int, ...],
) -> tuple[int, ...]:
    if isinstance(kernel_shape, int):
        return (kernel_shape,)
    return tuple(kernel_shape)


def _resolve_filter_bases(
    filter_bases: list[DiscoBasisSpec] | None,
    filter_basis_type: FilterBasisType,
    kernel_shape: tuple[int, int],
) -> list[DiscoBasisSpec]:
    if filter_bases:
        return filter_bases
    return [
        DiscoBasisSpec(
            basis_type=filter_basis_type,
            kernel_shape=_normalize_kernel_shape(kernel_shape),
        )
    ]


def _compute_cutoff_radius(
    nlat: int, kernel_shape: tuple[int, ...], basis_type: str
) -> float:
    if basis_type not in _THETA_CUTOFF_FACTOR:
        raise ValueError(f"Unknown basis_type {basis_type!r}")
    return (
        (kernel_shape[0] + 1)
        * _THETA_CUTOFF_FACTOR[basis_type]
        * math.pi
        / float(nlat - 1)
    )


def _default_context_config() -> SphericalUNetContextConfig:
    return SphericalUNetContextConfig()


def _empty_context() -> Context:
    return Context(
        embedding_scalar=None,
        embedding_pos=None,
        labels=None,
        noise=None,
    )


def _context_at_shape(context: Context | None, shape: tuple[int, int]) -> Context:
    if context is None:
        return _empty_context()
    if context.noise is None or context.noise.shape[1] == 0:
        return context
    noise = context.noise
    if noise.shape[-2:] != shape:
        noise = F.interpolate(noise, size=shape, mode="bilinear", align_corners=False)
    return dataclasses.replace(context, noise=noise)


class _LayerScale(nn.Module):
    """Per-channel scaling before the residual add (ConvNeXt / FourCastNet3 style)."""

    def __init__(self, num_channels: int, init_value: float = 0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))
        nn.init.constant_(self.weight, init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class _MultiBasisDiscreteContinuousConvS2(nn.Module):
    """Parallel DISCO branches with per-branch LayerScale; outputs summed."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: tuple[int, int],
        out_shape: tuple[int, int],
        filter_bases: list[DiscoBasisSpec],
        basis_norm_mode: BasisNormMode,
        grid_in: str,
        grid_out: str,
        nlat_for_cutoff: int,
        layer_scale_init: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        if len(filter_bases) < 2:
            raise ValueError("Multi-basis wrapper requires at least 2 filter bases")
        self.branches = nn.ModuleList()
        self.branch_scales = nn.ModuleList()
        for spec in filter_bases:
            theta_cutoff = _compute_cutoff_radius(
                nlat_for_cutoff, spec.kernel_shape, spec.basis_type
            )
            self.branches.append(
                DiscreteContinuousConvS2(
                    in_channels,
                    out_channels,
                    in_shape=in_shape,
                    out_shape=out_shape,
                    kernel_shape=spec.kernel_shape,
                    basis_type=spec.basis_type,
                    basis_norm_mode=basis_norm_mode,
                    grid_in=grid_in,
                    grid_out=grid_out,
                    bias=bias,
                    theta_cutoff=theta_cutoff,
                )
            )
            self.branch_scales.append(
                _LayerScale(out_channels, init_value=layer_scale_init)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.branch_scales[0](self.branches[0](x))
        for branch, scale in zip(self.branches[1:], self.branch_scales[1:]):
            out = out + scale(branch(x))
        return out


class _MultiBasisDiscreteContinuousConvTransposeS2(nn.Module):
    """Parallel DISCO transpose branches with per-branch LayerScale; outputs summed."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: tuple[int, int],
        out_shape: tuple[int, int],
        filter_bases: list[DiscoBasisSpec],
        basis_norm_mode: BasisNormMode,
        grid_in: str,
        grid_out: str,
        nlat_for_cutoff: int,
        layer_scale_init: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        if len(filter_bases) < 2:
            raise ValueError("Multi-basis wrapper requires at least 2 filter bases")
        self.branches = nn.ModuleList()
        self.branch_scales = nn.ModuleList()
        for spec in filter_bases:
            theta_cutoff = _compute_cutoff_radius(
                nlat_for_cutoff, spec.kernel_shape, spec.basis_type
            )
            self.branches.append(
                DiscreteContinuousConvTransposeS2(
                    in_channels,
                    out_channels,
                    in_shape=in_shape,
                    out_shape=out_shape,
                    kernel_shape=spec.kernel_shape,
                    basis_type=spec.basis_type,
                    basis_norm_mode=basis_norm_mode,
                    grid_in=grid_in,
                    grid_out=grid_out,
                    bias=bias,
                    theta_cutoff=theta_cutoff,
                )
            )
            self.branch_scales.append(
                _LayerScale(out_channels, init_value=layer_scale_init)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.branch_scales[0](self.branches[0](x))
        for branch, scale in zip(self.branches[1:], self.branch_scales[1:]):
            out = out + scale(branch(x))
        return out


def _make_disco_conv(
    in_channels: int,
    out_channels: int,
    in_shape: tuple[int, int],
    out_shape: tuple[int, int],
    filter_bases: list[DiscoBasisSpec],
    basis_norm_mode: BasisNormMode,
    grid_in: str,
    grid_out: str,
    nlat_for_cutoff: int,
    layer_scale_init: float = 0.1,
    bias: bool = False,
) -> nn.Module:
    if len(filter_bases) == 1:
        spec = filter_bases[0]
        theta_cutoff = _compute_cutoff_radius(
            nlat_for_cutoff, spec.kernel_shape, spec.basis_type
        )
        return DiscreteContinuousConvS2(
            in_channels,
            out_channels,
            in_shape=in_shape,
            out_shape=out_shape,
            kernel_shape=spec.kernel_shape,
            basis_type=spec.basis_type,
            basis_norm_mode=basis_norm_mode,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=bias,
            theta_cutoff=theta_cutoff,
        )
    return _MultiBasisDiscreteContinuousConvS2(
        in_channels,
        out_channels,
        in_shape=in_shape,
        out_shape=out_shape,
        filter_bases=filter_bases,
        basis_norm_mode=basis_norm_mode,
        grid_in=grid_in,
        grid_out=grid_out,
        nlat_for_cutoff=nlat_for_cutoff,
        layer_scale_init=layer_scale_init,
        bias=bias,
    )


def _make_disco_transpose_conv(
    in_channels: int,
    out_channels: int,
    in_shape: tuple[int, int],
    out_shape: tuple[int, int],
    filter_bases: list[DiscoBasisSpec],
    basis_norm_mode: BasisNormMode,
    grid_in: str,
    grid_out: str,
    nlat_for_cutoff: int,
    layer_scale_init: float = 0.1,
    bias: bool = False,
) -> nn.Module:
    if len(filter_bases) == 1:
        spec = filter_bases[0]
        theta_cutoff = _compute_cutoff_radius(
            nlat_for_cutoff, spec.kernel_shape, spec.basis_type
        )
        return DiscreteContinuousConvTransposeS2(
            in_channels,
            out_channels,
            in_shape=in_shape,
            out_shape=out_shape,
            kernel_shape=spec.kernel_shape,
            basis_type=spec.basis_type,
            basis_norm_mode=basis_norm_mode,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=bias,
            theta_cutoff=theta_cutoff,
        )
    return _MultiBasisDiscreteContinuousConvTransposeS2(
        in_channels,
        out_channels,
        in_shape=in_shape,
        out_shape=out_shape,
        filter_bases=filter_bases,
        basis_norm_mode=basis_norm_mode,
        grid_in=grid_in,
        grid_out=grid_out,
        nlat_for_cutoff=nlat_for_cutoff,
        layer_scale_init=layer_scale_init,
        bias=bias,
    )


class _DiscoConvNeXtBlock(nn.Module):
    """DISCO spatial conv + CLN + pointwise MLP + layer scale + residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: tuple[int, int],
        grid: str,
        filter_bases: list[DiscoBasisSpec],
        basis_norm_mode: BasisNormMode,
        activation: type[nn.Module],
        context_config: SphericalUNetContextConfig,
        mlp_ratio: float = 2.0,
        mlp_drop_rate: float = 0.0,
        path_drop_rate: float = 0.0,
        layer_scale: bool = True,
        layer_scale_init: float = 0.1,
    ):
        super().__init__()
        self.conv = _make_disco_conv(
            in_channels,
            in_channels,
            in_shape=in_shape,
            out_shape=in_shape,
            filter_bases=filter_bases,
            basis_norm_mode=basis_norm_mode,
            grid_in=grid,
            grid_out=grid,
            nlat_for_cutoff=in_shape[0],
            layer_scale_init=layer_scale_init,
            bias=False,
        )
        self.norm = ConditionalLayerNorm(
            in_channels, in_shape, context_config.to_context_config()
        )
        hidden_features = int(in_channels * mlp_ratio)
        self.mlp = MLP(
            in_features=in_channels,
            hidden_features=hidden_features,
            out_features=out_channels,
            act_layer=activation,
            drop_rate=mlp_drop_rate,
        )
        self.drop_path = (
            DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()
        )
        if layer_scale:
            self.layer_scale: nn.Module = _LayerScale(
                out_channels, init_value=layer_scale_init
            )
        else:
            self.layer_scale = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, context: Context) -> torch.Tensor:
        residual = self.skip(x)
        dx = self.conv(x)
        dx = self.norm(dx, context)
        dx = self.mlp(dx)
        dx = self.drop_path(dx)
        return residual + self.layer_scale(dx)


class _TransposeConvNormActConv(nn.Module):
    """DISCO transpose conv, CLN, activation, and follow-up DISCO conv."""

    def __init__(
        self,
        transpose_conv: nn.Module,
        conv: nn.Module,
        out_channels: int,
        out_shape: tuple[int, int],
        activation: type[nn.Module],
        context_config: SphericalUNetContextConfig,
    ):
        super().__init__()
        self.transpose_conv = transpose_conv
        self.norm = ConditionalLayerNorm(
            out_channels, out_shape, context_config.to_context_config()
        )
        self.activation = activation()
        self.conv = conv

    def forward(self, x: torch.Tensor, context: Context) -> torch.Tensor:
        x = self.transpose_conv(x)
        x = self.norm(x, context)
        x = self.activation(x)
        x = self.conv(x)
        return x


class DownsamplingBlock(nn.Module):
    """Down block: DISCO downsample, then ConvNeXt blocks at lower resolution."""

    def __init__(
        self,
        in_shape: tuple[int, int],
        out_shape: tuple[int, int],
        in_channels: int,
        out_channels: int,
        grid_in: str = "equiangular",
        grid_out: str = "equiangular",
        nrep: int = 1,
        filter_bases: list[DiscoBasisSpec] | None = None,
        basis_norm_mode: BasisNormMode = "nodal",
        activation: type[nn.Module] = nn.ReLU,
        transform_skip: bool = False,
        path_drop_rates: list[float] | None = None,
        downsampling_mode: str = "conv",
        context_config: SphericalUNetContextConfig | None = None,
        mlp_ratio: float = 2.0,
        mlp_drop_rate: float = 0.0,
        layer_scale: bool = True,
        layer_scale_init: float = 0.1,
    ):
        super().__init__()
        if context_config is None:
            context_config = _default_context_config()
        if filter_bases is None:
            raise ValueError("filter_bases must be provided")
        if path_drop_rates is None or len(path_drop_rates) != nrep:
            raise ValueError(
                f"path_drop_rates must have length nrep={nrep}, "
                f"got {None if path_drop_rates is None else len(path_drop_rates)}"
            )

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_in = grid_in
        self.grid_out = grid_out
        self.downsampling_mode = downsampling_mode

        if downsampling_mode == "conv":
            # FCN3 encoder: DISCO conv changes resolution and channel width first.
            self.downsample = _make_disco_conv(
                in_channels,
                out_channels,
                in_shape=in_shape,
                out_shape=out_shape,
                filter_bases=filter_bases,
                basis_norm_mode=basis_norm_mode,
                grid_in=grid_in,
                grid_out=grid_out,
                nlat_for_cutoff=in_shape[0],
                layer_scale_init=layer_scale_init,
                bias=False,
            )
            process_shape = out_shape
            process_grid = grid_out
            block_in_channels = out_channels
        elif downsampling_mode == "bilinear":
            self.downsample = ResampleS2(
                nlat_in=in_shape[0],
                nlon_in=in_shape[1],
                nlat_out=out_shape[0],
                nlon_out=out_shape[1],
                grid_in=grid_in,
                grid_out=grid_out,
                mode="bilinear",
            )
            process_shape = in_shape
            process_grid = grid_out
            block_in_channels = in_channels
        else:
            raise ValueError(
                f"Unknown downsampling_mode {downsampling_mode!r}; "
                "expected 'conv' or 'bilinear'"
            )

        self.fwd = nn.ModuleList()
        for i in range(nrep):
            layer_in_channels = block_in_channels if i == 0 else out_channels
            self.fwd.append(
                _DiscoConvNeXtBlock(
                    in_channels=layer_in_channels,
                    out_channels=out_channels,
                    in_shape=process_shape,
                    grid=process_grid,
                    filter_bases=filter_bases,
                    basis_norm_mode=basis_norm_mode,
                    activation=activation,
                    context_config=context_config,
                    mlp_ratio=mlp_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    path_drop_rate=path_drop_rates[i],
                    layer_scale=layer_scale,
                    layer_scale_init=layer_scale_init,
                )
            )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, context: Context | None = None) -> torch.Tensor:
        if self.downsampling_mode == "conv":
            x = self.downsample(x)
            block_context = _context_at_shape(context, self.out_shape)
            for layer in self.fwd:
                x = layer(x, block_context)
        else:
            block_context = _context_at_shape(context, self.in_shape)
            for layer in self.fwd:
                x = layer(x, block_context)
            x = self.downsample(x)
        return x


class UpsamplingBlock(nn.Module):
    """Up block: bilinear upsample, then ConvNeXt blocks at higher resolution."""

    def __init__(
        self,
        in_shape: tuple[int, int],
        out_shape: tuple[int, int],
        in_channels: int,
        out_channels: int,
        grid_in: str = "equiangular",
        grid_out: str = "equiangular",
        nrep: int = 1,
        filter_bases: list[DiscoBasisSpec] | None = None,
        basis_norm_mode: BasisNormMode = "nodal",
        activation: type[nn.Module] = nn.ReLU,
        transform_skip: bool = False,
        path_drop_rates: list[float] | None = None,
        upsampling_mode: str = "bilinear",
        context_config: SphericalUNetContextConfig | None = None,
        mlp_ratio: float = 2.0,
        mlp_drop_rate: float = 0.0,
        layer_scale: bool = True,
        layer_scale_init: float = 0.1,
    ):
        super().__init__()
        if context_config is None:
            context_config = _default_context_config()
        if filter_bases is None:
            raise ValueError("filter_bases must be provided")
        if path_drop_rates is None or len(path_drop_rates) != nrep:
            raise ValueError(
                f"path_drop_rates must have length nrep={nrep}, "
                f"got {None if path_drop_rates is None else len(path_drop_rates)}"
            )

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsampling_mode = upsampling_mode

        if in_shape != out_shape:
            if upsampling_mode == "bilinear":
                self.upsample: nn.Module = ResampleS2(
                    nlat_in=in_shape[0],
                    nlon_in=in_shape[1],
                    nlat_out=out_shape[0],
                    nlon_out=out_shape[1],
                    grid_in=grid_in,
                    grid_out=grid_out,
                    mode="bilinear",
                )
                process_shape = out_shape
                process_grid = grid_out
            elif upsampling_mode == "conv":
                self.upsample = _TransposeConvNormActConv(
                    _make_disco_transpose_conv(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        in_shape=in_shape,
                        out_shape=out_shape,
                        filter_bases=filter_bases,
                        basis_norm_mode=basis_norm_mode,
                        grid_in=grid_in,
                        grid_out=grid_out,
                        nlat_for_cutoff=in_shape[0],
                        layer_scale_init=layer_scale_init,
                        bias=False,
                    ),
                    _make_disco_conv(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        in_shape=out_shape,
                        out_shape=out_shape,
                        filter_bases=filter_bases,
                        basis_norm_mode=basis_norm_mode,
                        grid_in=grid_in,
                        grid_out=grid_out,
                        nlat_for_cutoff=out_shape[0],
                        layer_scale_init=layer_scale_init,
                        bias=False,
                    ),
                    out_channels,
                    out_shape,
                    activation,
                    context_config,
                )
                process_shape = in_shape
                process_grid = grid_in
            else:
                raise ValueError(
                    f"Unknown upsampling_mode {upsampling_mode!r}; "
                    "expected 'bilinear' or 'conv'"
                )
        else:
            self.upsample = nn.Identity()
            process_shape = in_shape
            process_grid = grid_in

        self.fwd = nn.ModuleList()
        for i in range(nrep):
            layer_in_channels = in_channels
            layer_out_channels = out_channels if i == nrep - 1 else in_channels
            self.fwd.append(
                _DiscoConvNeXtBlock(
                    in_channels=layer_in_channels,
                    out_channels=layer_out_channels,
                    in_shape=process_shape,
                    grid=process_grid,
                    filter_bases=filter_bases,
                    basis_norm_mode=basis_norm_mode,
                    activation=activation,
                    context_config=context_config,
                    mlp_ratio=mlp_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    path_drop_rate=path_drop_rates[i],
                    layer_scale=layer_scale,
                    layer_scale_init=layer_scale_init,
                )
            )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, context: Context | None = None) -> torch.Tensor:
        if self.upsampling_mode == "bilinear":
            if self.in_shape != self.out_shape:
                x = self.upsample(x)
            block_context = _context_at_shape(context, self.out_shape)
            for layer in self.fwd:
                x = layer(x, block_context)
        else:
            block_context = _context_at_shape(context, self.in_shape)
            for layer in self.fwd:
                x = layer(x, block_context)
            if isinstance(self.upsample, _TransposeConvNormActConv):
                upsample_context = _context_at_shape(context, self.out_shape)
                x = self.upsample(x, upsample_context)
            elif self.in_shape != self.out_shape:
                x = self.upsample(x)
        return x


class SphericalUNet(nn.Module):
    """Spherical U-Net with ConditionalLayerNorm (noise/labels/scalar only)."""

    def __init__(
        self,
        img_size: tuple[int, int] = (128, 256),
        grid: str = "equiangular",
        grid_internal: str = "legendre-gauss",
        in_chans: int = 3,
        out_chans: int = 3,
        embed_dims: list[int] | None = None,
        depths: list[int] | None = None,
        scale_factor: int = 2,
        activation_function: Literal["relu", "gelu", "identity"] = "relu",
        kernel_shape: tuple[int, int] = (3, 3),
        filter_basis_type: FilterBasisType = "harmonic",
        filter_bases: list[DiscoBasisSpec] | None = None,
        filter_basis_norm_mode: BasisNormMode = "nodal",
        transform_skip: bool = False,
        path_drop_rate: float = 0.0,
        mlp_drop_rate: float = 0.0,
        downsampling_mode: str = "conv",
        upsampling_mode: str = "bilinear",
        context_config: SphericalUNetContextConfig | None = None,
        mlp_ratio: float = 2.0,
        layer_scale: bool = True,
        layer_scale_init: float = 0.1,
    ):
        super().__init__()

        if embed_dims is None:
            embed_dims = [64, 128, 256]
        if depths is None:
            depths = [2, 2, 2]

        self.img_size = img_size
        self.grid = grid
        self.grid_internal = grid_internal
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dims = embed_dims
        self.num_blocks = len(self.embed_dims)
        self.depths = depths
        self.kernel_shape = kernel_shape
        self.filter_bases = _resolve_filter_bases(
            filter_bases, filter_basis_type, kernel_shape
        )
        if context_config is None:
            context_config = _default_context_config()
        self.context_config = context_config
        self.mlp_ratio = mlp_ratio
        self.mlp_drop_rate = mlp_drop_rate
        self.layer_scale = layer_scale
        self.layer_scale_init = layer_scale_init

        if len(self.depths) != self.num_blocks:
            raise ValueError(
                f"depths must have length num_blocks={self.num_blocks}, "
                f"got {len(self.depths)}"
            )

        if activation_function == "relu":
            self.activation_function: type[nn.Module] = nn.ReLU
        elif activation_function == "gelu":
            self.activation_function = nn.GELU
        elif activation_function == "identity":
            self.activation_function = nn.Identity
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        dpr = [
            x.item() for x in torch.linspace(0, path_drop_rate, 2 * sum(self.depths))
        ]
        dpr_idx = 0

        self.dblocks = nn.ModuleList()
        out_shape = img_size
        grid_in = grid
        grid_out = grid_internal
        in_channels = in_chans
        for i in range(self.num_blocks):
            out_shape_new = (
                out_shape[0] // scale_factor,
                out_shape[1] // scale_factor,
            )
            out_channels = self.embed_dims[i]
            n = self.depths[i]
            block_dpr = dpr[dpr_idx : dpr_idx + n]
            dpr_idx += n
            self.dblocks.append(
                DownsamplingBlock(
                    in_shape=out_shape,
                    out_shape=out_shape_new,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    grid_in=grid_in,
                    grid_out=grid_out,
                    nrep=n,
                    filter_bases=self.filter_bases,
                    basis_norm_mode=filter_basis_norm_mode,
                    activation=self.activation_function,
                    path_drop_rates=block_dpr,
                    transform_skip=transform_skip,
                    downsampling_mode=downsampling_mode,
                    context_config=context_config,
                    mlp_ratio=mlp_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    layer_scale=layer_scale,
                    layer_scale_init=layer_scale_init,
                )
            )
            out_shape = out_shape_new
            grid_in = grid_internal
            in_channels = out_channels

        self.ublocks = nn.ModuleList()
        for i in range(self.num_blocks - 1, -1, -1):
            block_in_shape = self.dblocks[i].out_shape
            block_out_shape = self.dblocks[i].in_shape
            block_in_channels = self.dblocks[i].out_channels
            if i != self.num_blocks - 1:
                block_in_channels = 2 * block_in_channels
            block_out_channels = self.dblocks[i].in_channels
            if i == 0:
                block_out_channels = self.embed_dims[0]
            block_grid_in = self.dblocks[i].grid_out
            block_grid_out = self.dblocks[i].grid_in
            n = self.depths[i]
            block_dpr = dpr[dpr_idx : dpr_idx + n]
            dpr_idx += n
            self.ublocks.append(
                UpsamplingBlock(
                    in_shape=block_in_shape,
                    out_shape=block_out_shape,
                    in_channels=block_in_channels,
                    out_channels=block_out_channels,
                    grid_in=block_grid_in,
                    grid_out=block_grid_out,
                    nrep=n,
                    filter_bases=self.filter_bases,
                    basis_norm_mode=filter_basis_norm_mode,
                    activation=self.activation_function,
                    path_drop_rates=block_dpr,
                    transform_skip=transform_skip,
                    upsampling_mode=upsampling_mode,
                    context_config=context_config,
                    mlp_ratio=mlp_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    layer_scale=layer_scale,
                    layer_scale_init=layer_scale_init,
                )
            )

        self.head = nn.Conv2d(
            self.embed_dims[0], self.out_chans, kernel_size=1, bias=True
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, context: Context | None = None) -> torch.Tensor:
        block_context = _context_at_shape(context, self.img_size)
        features = []
        feat = x
        for dblock in self.dblocks:
            feat = dblock(feat, block_context)
            features.append(feat)

        features = features[::-1]

        ufeat = self.ublocks[0](features[0], block_context)
        for feat, ublock in zip(features[1:], self.ublocks[1:]):
            ufeat = ublock(torch.cat([feat, ufeat], dim=1), block_context)

        return self.head(ufeat)
