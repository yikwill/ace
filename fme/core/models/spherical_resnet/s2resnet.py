# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Full-resolution spherical ResNet: DISCO ConvNeXt blocks without U-Net pooling.

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from fme.core.models.conditional_sfno.layers import Context
from fme.core.models.spherical_unet.s2unet import (
    BasisNormMode,
    DiscoBasisSpec,
    FilterBasisType,
    SphericalUNetContextConfig,
    _context_at_shape,
    _default_context_config,
    _DiscoConvNeXtBlock,
    _make_disco_conv,
    _resolve_filter_bases,
)


class SphericalResNet(nn.Module):
    """Small full-resolution spherical ResNet with DISCO ConvNeXt blocks."""

    def __init__(
        self,
        img_size: tuple[int, int] = (180, 360),
        grid: str = "legendre-gauss",
        in_chans: int = 3,
        out_chans: int = 3,
        embed_dim: int = 64,
        depth: int = 2,
        activation_function: Literal["relu", "gelu", "identity"] = "gelu",
        kernel_shape: tuple[int, int] = (3, 3),
        filter_basis_type: FilterBasisType = "harmonic",
        filter_bases: list[DiscoBasisSpec] | None = None,
        filter_basis_norm_mode: BasisNormMode = "nodal",
        path_drop_rate: float = 0.0,
        mlp_drop_rate: float = 0.0,
        context_config: SphericalUNetContextConfig | None = None,
        mlp_ratio: float = 2.0,
        layer_scale: bool = True,
        layer_scale_init: float = 0.1,
        # Shared DISCO support radius (radians). None keeps heuristic.
        theta_cutoff: float | None = None,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.img_size = img_size
        self.grid = grid
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.theta_cutoff = theta_cutoff
        self.filter_bases = _resolve_filter_bases(
            filter_bases, filter_basis_type, kernel_shape
        )
        if context_config is None:
            context_config = _default_context_config()
        self.context_config = context_config

        if activation_function == "relu":
            activation: type[nn.Module] = nn.ReLU
        elif activation_function == "gelu":
            activation = nn.GELU
        elif activation_function == "identity":
            activation = nn.Identity
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        self.stem = _make_disco_conv(
            in_chans,
            embed_dim,
            in_shape=img_size,
            out_shape=img_size,
            filter_bases=self.filter_bases,
            basis_norm_mode=filter_basis_norm_mode,
            grid_in=grid,
            grid_out=grid,
            nlat_for_cutoff=img_size[0],
            layer_scale_init=layer_scale_init,
            bias=False,
            theta_cutoff=theta_cutoff,
        )

        dpr = [x.item() for x in torch.linspace(0, path_drop_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                _DiscoConvNeXtBlock(
                    embed_dim,
                    embed_dim,
                    in_shape=img_size,
                    grid=grid,
                    filter_bases=self.filter_bases,
                    basis_norm_mode=filter_basis_norm_mode,
                    activation=activation,
                    context_config=context_config,
                    mlp_ratio=mlp_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    path_drop_rate=dpr[i],
                    layer_scale=layer_scale,
                    layer_scale_init=layer_scale_init,
                    theta_cutoff=theta_cutoff,
                )
                for i in range(depth)
            ]
        )
        self.head = nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, context: Context | None = None) -> torch.Tensor:
        block_context = _context_at_shape(context, self.img_size)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x, block_context)
        return self.head(x)


# Re-export for registry convenience.
SphericalResNetContextConfig = SphericalUNetContextConfig

__all__ = [
    "DiscoBasisSpec",
    "SphericalResNet",
    "SphericalResNetContextConfig",
]
