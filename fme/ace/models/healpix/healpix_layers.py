# flake8: noqa
# Copied from https://github.com/NVIDIA/modulus/commit/89a6091bd21edce7be4e0539cbd91507004faf08
# Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
HEALPix convolution wrapper and re-exports for padding utilities.

Face layout and padding modes are documented in ``healpix_paddings``.
"""

from __future__ import annotations

import torch as th
import torch.nn as nn

from .healpix_paddings import (
    HEALPixFoldFaces,
    HEALPixPadding,
    HEALPixPaddingIsolatitude,
    HEALPixPaddingv2,
    HEALPixUnfoldFaces,
    build_isolatitude_gather_index,
    have_earth2grid,
    isolatitude_pad_folded,
    make_hpx_padding_layer,
    pop_deprecated_enable_healpixpad_from_kwargs,
    warn_deprecated_enable_healpixpad,
)


def _symmetric_pad_width(kernel_size, dilation) -> int:
    """Effective HEALPix edge pad (per side) for conv-like geometry."""
    if isinstance(kernel_size, int):
        ks = (kernel_size,)
    else:
        ks = tuple(kernel_size)
    if isinstance(dilation, int):
        dil = (dilation,) * len(ks)
    else:
        dil = tuple(dilation)
        if len(dil) == 1 and len(ks) > 1:
            dil = dil * len(ks)
    if len(dil) != len(ks):
        raise ValueError("dilation must be int, or tuple matching kernel_size")
    return max((((k - 1) // 2) * d) for k, d in zip(ks, dil))


class HEALPixLayer(nn.Module):
    """
    Apply a base ``torch.nn.Module`` on data laid out as HEALPix faces.

    Expected layout: folded ``[N * 12, C, H, W]``. When computed edge padding is
    positive, a HEALPix-aware padding module is prepended, then the base layer.
    """

    def __init__(
        self,
        layer,
        hpx_padding_mode: str | None = "earth2grid",
        nside: int | None = None,
        compile_padding: bool = False,
        **kwargs,
    ):
        """
        Args:
            layer: Base ``torch.nn`` layer class to wrap (e.g. ``nn.Conv2d``).
            hpx_padding_mode: HEALPix padding backend. ``"earth2grid"`` matches legacy
                ``enable_healpixpad=True`` behavior; ``"karlbauer"`` and
                ``"isolatitude"`` select pure-PyTorch implementations.
            nside: Native face size, required for ``hpx_padding_mode="isolatitude"``.
            compile_padding: If True, compile the HEALPix padding module.
            **kwargs: Keyword args forwarded to ``layer`` and HEALPix wrapper options.
        """
        super().__init__()
        layers_list: list[nn.Module] = []

        legacy_enable_healpixpad = pop_deprecated_enable_healpixpad_from_kwargs(kwargs)
        hpx_padding_mode = warn_deprecated_enable_healpixpad(
            legacy_enable_healpixpad, hpx_padding_mode
        )
        self.hpx_padding_mode = hpx_padding_mode

        if "nside" in kwargs:
            _ns = kwargs.pop("nside")
            nside = int(_ns) if _ns is not None else None
        if "compile_padding" in kwargs:
            compile_padding = bool(kwargs.pop("compile_padding"))

        if "enable_nhwc" in kwargs:
            enable_nhwc = kwargs["enable_nhwc"]
            del kwargs["enable_nhwc"]
        else:
            enable_nhwc = False

        if "enable_healpixpad" in kwargs and kwargs["enable_healpixpad"]:
            raise NotImplementedError(
                "HEALPixPaddingv2 is not available in this environment"
            )

        if "enable_healpixpad" in kwargs:
            del kwargs["enable_healpixpad"]

        if not isinstance(layer, type) or not issubclass(layer, th.nn.Module):
            raise TypeError(
                f"Expected a subclass of torch.nn.Module, got {type(layer).__name__}"
            )
        # Define a HEALPixPadding layer if the given layer is a convolution layer
        if layer.__bases__[0] is nn.modules.conv._ConvNd and kwargs["kernel_size"] > 1:
            kwargs["padding"] = 0  # Disable native padding
            kernel_size = 3 if "kernel_size" not in kwargs else kwargs["kernel_size"]
            dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
            padding = ((kernel_size - 1) // 2) * dilation
            layers_list.append(HEALPixPadding(padding=padding, enable_nhwc=enable_nhwc))

        layers_list.append(layer(**kwargs))
        self.layers = nn.Sequential(*layers_list)

        if enable_nhwc:
            self.layers = self.layers.to(memory_format=th.channels_last)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.layers(x)


__all__ = [
    "HEALPixFoldFaces",
    "HEALPixLayer",
    "HEALPixPadding",
    "HEALPixPaddingIsolatitude",
    "HEALPixPaddingv2",
    "HEALPixUnfoldFaces",
    "build_isolatitude_gather_index",
    "have_earth2grid",
    "isolatitude_pad_folded",
    "make_hpx_padding_layer",
    "pop_deprecated_enable_healpixpad_from_kwargs",
    "warn_deprecated_enable_healpixpad",
]
