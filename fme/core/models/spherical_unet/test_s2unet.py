import warnings

import pytest
import torch
import torch.nn as nn
from torch_harmonics import DiscreteContinuousConvS2

import fme
from fme.core.models.conditional_sfno.layers import MLP, ConditionalLayerNorm, Context
from fme.core.models.spherical_unet import (
    DiscoBasisSpec,
    SphericalUNet,
    SphericalUNetContextConfig,
)
from fme.core.models.spherical_unet.s2unet import (
    _DiscoConvNeXtBlock,
    _LayerScale,
    _MultiBasisDiscreteContinuousConvS2,
)

SMALL_IMG = (32, 64)
SMALL_EMBED = [8, 16, 32]
ERA5_IMG = (180, 360)

MULTI_FILTER_BASES = [
    DiscoBasisSpec(basis_type="harmonic", kernel_shape=(3, 3)),
    DiscoBasisSpec(basis_type="zernike", kernel_shape=(4,)),
]


def _small_unet(
    *,
    img_size: tuple[int, int] = SMALL_IMG,
    in_chans: int = 5,
    out_chans: int = 3,
    embed_dims: list[int] | None = None,
    depths: list[int] | None = None,
    scale_factor: int = 2,
    path_drop_rate: float = 0.0,
    mlp_drop_rate: float = 0.0,
    context_config: SphericalUNetContextConfig | None = None,
    filter_bases: list[DiscoBasisSpec] | None = None,
    downsampling_mode: str = "conv",
    upsampling_mode: str = "bilinear",
    theta_cutoff: list[float | None] | None = None,
    unet_layout: str = "downsample_first",
) -> SphericalUNet:
    return SphericalUNet(
        img_size=img_size,
        in_chans=in_chans,
        out_chans=out_chans,
        embed_dims=embed_dims if embed_dims is not None else SMALL_EMBED,
        depths=depths if depths is not None else [1, 1, 1],
        scale_factor=scale_factor,
        path_drop_rate=path_drop_rate,
        mlp_drop_rate=mlp_drop_rate,
        context_config=context_config,
        filter_bases=filter_bases,
        downsampling_mode=downsampling_mode,
        upsampling_mode=upsampling_mode,
        theta_cutoff=theta_cutoff,
        unet_layout=unet_layout,  # type: ignore[arg-type]
    )


def test_forward_shape_small():
    model = _small_unet().to(fme.get_device())
    x = torch.randn(2, 5, *SMALL_IMG, device=fme.get_device())
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        out = model(x)
    assert out.shape == (2, 3, *SMALL_IMG)


@pytest.mark.gpu
def test_forward_shape_era5():
    model = SphericalUNet(
        img_size=ERA5_IMG,
        in_chans=4,
        out_chans=2,
        embed_dims=[16, 32, 64],
        depths=[1, 1, 1],
        path_drop_rate=0.0,
        mlp_drop_rate=0.0,
    ).to(fme.get_device())
    x = torch.randn(1, 4, *ERA5_IMG, device=fme.get_device())
    out = model(x)
    assert out.shape == (1, 2, *ERA5_IMG)


def test_uses_conditional_layer_norm_not_batch_norm():
    model = _small_unet()
    for module in model.modules():
        assert not isinstance(module, nn.BatchNorm2d)
    cln_count = sum(
        1 for module in model.modules() if isinstance(module, ConditionalLayerNorm)
    )
    assert cln_count > 0


def test_convnext_block_structure():
    model = _small_unet()
    blocks = [m for m in model.modules() if isinstance(m, _DiscoConvNeXtBlock)]
    assert len(blocks) > 0
    block = blocks[0]
    assert isinstance(block.norm, ConditionalLayerNorm)
    assert isinstance(block.mlp, MLP)
    assert isinstance(block.layer_scale, _LayerScale)


def test_context_config_has_no_positional_embedding():
    config = SphericalUNetContextConfig(embed_dim_noise=8)
    assert not hasattr(config, "embed_dim_pos")
    assert config.to_context_config().embed_dim_pos == 0


def test_forward_with_cln():
    noise_dim = 8
    context_config = SphericalUNetContextConfig(embed_dim_noise=noise_dim)
    model = _small_unet(context_config=context_config).to(fme.get_device())
    x = torch.randn(2, 5, *SMALL_IMG, device=fme.get_device())
    noise = torch.randn(2, noise_dim, *SMALL_IMG, device=fme.get_device())
    context = Context(
        embedding_scalar=None,
        embedding_pos=None,
        labels=None,
        noise=noise,
    )
    out = model(x, context)
    assert out.shape == (2, 3, *SMALL_IMG)


def test_forward_unconditional_ignores_context():
    model = _small_unet().to(fme.get_device())
    x = torch.randn(2, 5, *SMALL_IMG, device=fme.get_device())
    noise = torch.randn(2, 8, *SMALL_IMG, device=fme.get_device())
    context = Context(
        embedding_scalar=None,
        embedding_pos=None,
        labels=None,
        noise=noise,
    )
    out_with = model(x, context)
    out_without = model(x)
    assert out_with.shape == out_without.shape


def test_single_basis_backward_compat():
    model = _small_unet().to(fme.get_device())
    blocks = [m for m in model.modules() if isinstance(m, _DiscoConvNeXtBlock)]
    assert len(blocks) > 0
    assert isinstance(blocks[0].conv, DiscreteContinuousConvS2)
    assert not isinstance(blocks[0].conv, _MultiBasisDiscreteContinuousConvS2)


def test_multi_basis_forward_shape():
    model = _small_unet(filter_bases=MULTI_FILTER_BASES).to(fme.get_device())
    x = torch.randn(2, 5, *SMALL_IMG, device=fme.get_device())
    out = model(x)
    assert out.shape == (2, 3, *SMALL_IMG)


def test_multi_basis_uses_wrapper():
    model = _small_unet(filter_bases=MULTI_FILTER_BASES)
    blocks = [m for m in model.modules() if isinstance(m, _DiscoConvNeXtBlock)]
    assert len(blocks) > 0
    conv = blocks[0].conv
    assert isinstance(conv, _MultiBasisDiscreteContinuousConvS2)
    assert len(conv.branches) == 2
    assert len(conv.branch_scales) == 2
    assert all(isinstance(scale, _LayerScale) for scale in conv.branch_scales)


def test_conv_upsample_multi_basis():
    model = _small_unet(
        filter_bases=MULTI_FILTER_BASES,
        downsampling_mode="conv",
        upsampling_mode="conv",
    ).to(fme.get_device())
    x = torch.randn(2, 5, *SMALL_IMG, device=fme.get_device())
    out = model(x)
    assert out.shape == (2, 3, *SMALL_IMG)


ISO_MORLET_BASE = [
    DiscoBasisSpec(basis_type="isotropic morlet", kernel_shape=(8,)),
]


def test_forward_isotropic_morlet():
    model = _small_unet(filter_bases=ISO_MORLET_BASE).to(fme.get_device())
    x = torch.randn(2, 5, *SMALL_IMG, device=fme.get_device())
    out = model(x)
    assert out.shape == (2, 3, *SMALL_IMG)


def test_isotropic_morlet_conv_lat_flip_symmetry():
    from fme.core.models.spherical_unet.s2unet import _compute_cutoff_radius

    embed_dim = 4
    img_shape = (16, 32)
    lat_dim = 2
    kernel_shape = (8,)

    theta_cutoff = _compute_cutoff_radius(
        nlat=img_shape[0],
        kernel_shape=kernel_shape,
        basis_type="isotropic morlet",
    )
    conv = DiscreteContinuousConvS2(
        embed_dim,
        embed_dim,
        in_shape=img_shape,
        out_shape=img_shape,
        kernel_shape=kernel_shape,
        basis_type="isotropic morlet",
        basis_norm_mode="nodal",
        groups=1,
        grid_in="equiangular",
        grid_out="equiangular",
        bias=False,
        theta_cutoff=theta_cutoff,
    )

    x = torch.randn(1, embed_dim, *img_shape)
    x_flipped = torch.flip(x, dims=[lat_dim])
    with torch.no_grad():
        out = conv(x)
        out_from_flipped = conv(x_flipped)
    torch.testing.assert_close(
        torch.flip(out, dims=[lat_dim]),
        out_from_flipped,
    )


def test_normalize_theta_cutoff_per_level():
    from fme.core.models.spherical_unet import s2unet as s2unet_mod

    assert s2unet_mod._normalize_theta_cutoff_per_level(None, 3) == [None, None, None]
    assert s2unet_mod._normalize_theta_cutoff_per_level([0.1, None, 0.3], 3) == [
        0.1,
        None,
        0.3,
    ]
    with pytest.raises(ValueError, match="num_blocks"):
        s2unet_mod._normalize_theta_cutoff_per_level([0.1, 0.2], 3)
    with pytest.raises(ValueError, match="positive"):
        s2unet_mod._normalize_theta_cutoff_per_level([0.1, 0.0, 0.3], 3)


def test_theta_cutoff_per_level_reaches_disco_convs():
    """Each U-Net level's DISCO layers must receive that level's override."""
    overrides: list[float | None] = [0.11, 0.22, 0.33]
    model = _small_unet(theta_cutoff=overrides)
    assert model.theta_cutoff == overrides

    for i, dblock in enumerate(model.dblocks):
        expected = overrides[i]
        # conv downsample + ConvNeXt spatial convs at this level
        disco = [m for m in dblock.modules() if isinstance(m, DiscreteContinuousConvS2)]
        assert disco, f"expected DISCO layers in dblock {i}"
        for layer in disco:
            assert layer.theta_cutoff == expected

    # ublocks are built deepest-first; still keyed by level index i
    for ublock, expected in zip(reversed(list(model.ublocks)), overrides):
        disco = [m for m in ublock.modules() if isinstance(m, DiscreteContinuousConvS2)]
        # bilinear upsampling may have no DISCO in the upsample path
        for layer in disco:
            assert layer.theta_cutoff == expected


def test_theta_cutoff_none_uses_heuristic_not_uniform():
    """Default None must not force one cutoff across differently-resolved levels."""
    model = _small_unet(theta_cutoff=None)
    cutoffs = [
        m.theta_cutoff
        for m in model.modules()
        if isinstance(m, DiscreteContinuousConvS2)
    ]
    assert cutoffs
    assert len(set(cutoffs)) > 1


def test_default_layout_is_downsample_first():
    model = _small_unet()
    assert model.unet_layout == "downsample_first"
    assert len(model.dblocks) == 3
    assert len(model.ublocks) == 3
    assert isinstance(model.stem, nn.Identity)


def test_classic_forward_shape():
    model = _small_unet(unet_layout="classic").to(fme.get_device())
    x = torch.randn(2, 5, *SMALL_IMG, device=fme.get_device())
    out = model(x)
    assert out.shape == (2, 3, *SMALL_IMG)


def test_classic_structure_n_process_n_minus_1_downs():
    model = _small_unet(unet_layout="classic", embed_dims=[8, 16, 32])
    assert model.unet_layout == "classic"
    assert len(model.encoder_stages) == 3
    assert len(model.downs) == 2
    assert len(model.ups) == 2
    assert len(model.decoder_stages) == 2
    assert len(model.dblocks) == 0
    assert len(model.ublocks) == 0
    assert not isinstance(model.stem, nn.Identity)
    assert model.level_shapes == [
        SMALL_IMG,
        (SMALL_IMG[0] // 2, SMALL_IMG[1] // 2),
        (SMALL_IMG[0] // 4, SMALL_IMG[1] // 4),
    ]
    # Spatial-only downs keep channel width.
    for i, down in enumerate(model.downs):
        if isinstance(down, DiscreteContinuousConvS2):
            assert down.weight.shape[0] == model.embed_dims[i]
            assert down.groupsize * down.groups == model.embed_dims[i]


def test_classic_theta_cutoff_per_level():
    overrides: list[float | None] = [0.11, 0.22, 0.33]
    model = _small_unet(unet_layout="classic", theta_cutoff=overrides)
    assert model.theta_cutoff == overrides
    for i, stage in enumerate(model.encoder_stages):
        disco = [m for m in stage.modules() if isinstance(m, DiscreteContinuousConvS2)]
        assert disco
        for layer in disco:
            assert layer.theta_cutoff == overrides[i]
    stem_disco = [
        m for m in model.stem.modules() if isinstance(m, DiscreteContinuousConvS2)
    ]
    if isinstance(model.stem, DiscreteContinuousConvS2):
        stem_disco = [model.stem]
    assert stem_disco
    for layer in stem_disco:
        assert layer.theta_cutoff == overrides[0]


def test_classic_multi_basis_forward():
    model = _small_unet(unet_layout="classic", filter_bases=MULTI_FILTER_BASES).to(
        fme.get_device()
    )
    x = torch.randn(2, 5, *SMALL_IMG, device=fme.get_device())
    out = model(x)
    assert out.shape == (2, 3, *SMALL_IMG)
