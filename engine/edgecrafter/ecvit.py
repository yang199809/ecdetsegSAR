"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DINOv3 (https://github.com/facebookresearch/dinov3)
Modified from https://huggingface.co/spaces/Hila/RobustViT/blob/main/ViT/ViT_new.py

"""
import math
import warnings
from functools import partial
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

from ..core import register
from ..misc import dist_utils
from .hybrid_encoder import ConvNormLayer_fuse

__all__ = ['ViTAdapter', ]


def safe_get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        assert head_dim % 4 == 0, "Head dimension must be divisible by 4 for 2D RoPE"
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = head_dim
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype
        self.register_buffer(
            "periods",
            torch.empty(head_dim // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.periods.device
        dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        dd = {"device": device, "dtype": dtype}

        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW
            coords_w = torch.arange(0.5, W, **dd) / max_HW
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else: # min
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW
            coords_w = torch.arange(0.5, W, **dd) / min_HW

        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0

        if self.training and self.shift_coords is not None:
            coords += torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)[None, :]
        if self.training and self.jitter_coords is not None:
            jitter = (torch.empty(2, **dd).uniform_(-np.log(self.jitter_coords), np.log(self.jitter_coords))).exp()
            coords *= jitter[None, :]
        if self.training and self.rescale_coords is not None:
            rescale = (torch.empty(1, **dd).uniform_(-np.log(self.rescale_coords), np.log(self.rescale_coords))).exp()
            coords *= rescale

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).repeat(1, 2)

        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return sin.unsqueeze(0).unsqueeze(0), cos.unsqueeze(0).unsqueeze(0)

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        if self.base is not None:
            periods = self.base ** (2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2))
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)
            periods = self.max_period * (base ** (exponents - 1))
        self.periods.data.copy_(periods)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, sin, cos):
    """Applies RoPE to the input tensor."""
    return (x * cos) + (rotate_half(x) * sin)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.act(self.fc1(x)) 
        x = self.drop(x) 
        x = self.fc2(x) 
        x = self.drop(x)
        return x

    
class ConvPyramidPatchEmbed(nn.Module):
    def __init__(self, embed_dim=192, patch_size=16, act='relu'):
        super().__init__()
        
        assert patch_size==16, "Only support patch_size=16 for ConvPyramidPatchEmbed"
        
        num_stages = int(math.log2(patch_size)) - 1
        ratios = [2 ** i for i in range(num_stages, 0, -1)]
        channels = [embed_dim // r for r in ratios]
        
        self.convs = nn.ModuleList([
            ConvNormLayer_fuse(in_ch, out_ch, 3, 2, act=act)
            for in_ch, out_ch in zip([3] + channels[:-1], channels)
        ])
        
        self.proj = nn.Conv2d(channels[-1], embed_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x): return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std); u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1); tensor.erfinv_(); tensor.mul_(std * math.sqrt(2.)); tensor.add_(mean); tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope_sincos=None):
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.unbind(0)
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads) # .permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

        if rope_sincos is not None:
            sin, cos = rope_sincos
            q_cls, q_patch = q[:, :, :1, :], q[:, :, 1:, :]
            k_cls, k_patch = k[:, :, :1, :], k[:, :, 1:, :]

            q_patch = apply_rope(q_patch, sin, cos)
            k_patch = apply_rope(k_patch, sin, cos)

            q = torch.cat((q_cls, q_patch), dim=2)
            k = torch.cat((k_cls, k_patch), dim=2)

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)
        x = x.transpose(1, 2).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.SiLU, norm_layer=nn.LayerNorm, ffn_layer=Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(in_features=dim, hidden_features=int(dim * ffn_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, rope_sincos=None):
        attn_output = self.attn(self.norm1(x), rope_sincos=rope_sincos)
        x = x + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=192, depth=12,
        num_heads=3, ffn_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0., return_layers=[3, 7, 11], embed_layer=ConvPyramidPatchEmbed,
        norm_layer=None, act_layer=None, ffn_layer=Mlp
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.return_layers = return_layers
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        if embed_layer == PatchEmbed:
            self.patch_embed = embed_layer(
                img_size=img_size, patch_size=patch_size,
                in_chans=in_chans, embed_dim=embed_dim
            )
        else:
            self.patch_embed = embed_layer(embed_dim=embed_dim, patch_size=patch_size)
        self.patch_size = patch_size

        self.register_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer, ffn_layer=ffn_layer,
            ) for i in range(depth)
        ])

        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim, num_heads=num_heads, base=100.0,
            normalize_coords="separate", shift_coords=None, jitter_coords=None,
            rescale_coords=None, dtype=None, device=None,
        )
        self.init_weights()

    def init_weights(self):
        self.apply(self._init_vit_weights)
        self.rope_embed._init_weights()
        trunc_normal_(self.register_token, std=.02)

    def _init_vit_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        outs = []
        x_embed = self.patch_embed(x)
        _, _, H, W = x_embed.shape
        
        x_embed = x_embed.flatten(2).transpose(1, 2)
        register_token = self.register_token.expand(x_embed.shape[0], -1, -1)
        x = torch.cat((register_token, x_embed), dim=1)
        rope_sincos = self.rope_embed(H=H, W=W)

        for i, blk in enumerate(self.blocks):
            x = blk(x, rope_sincos=rope_sincos)
            if i in self.return_layers:
                outs.append(x[:, 1:])
        return outs
    
    

EMBED_LAYER_REGISTRY = {
    "ConvPyramidPatchEmbed": ConvPyramidPatchEmbed,
    "PatchEmbed": PatchEmbed,
}

FFN_LAYER_REGISTRY = {
    "mlp": Mlp,
   # "swigluffn": SwiGLUFFN,  # To be implemented
}


# =========================================================
# 1. Haar DWT / IDWT
# =========================================================

def haar_dwt(x: torch.Tensor):
    """
    Haar DWT.

    Args:
        x: [B, C, H, W]

    Returns:
        bands: [B, 4C, H/2, W/2], order = [LL, LH, HL, HH]
        orig_hw: original spatial size
    """
    B, C, H, W = x.shape

    pad_h = H % 2
    pad_w = W % 2

    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    # Orthonormal-like Haar decomposition.
    LL = (x00 + x01 + x10 + x11) * 0.5
    LH = (-x00 - x01 + x10 + x11) * 0.5
    HL = (-x00 + x01 - x10 + x11) * 0.5
    HH = (x00 - x01 - x10 + x11) * 0.5

    bands = torch.cat([LL, LH, HL, HH], dim=1)
    return bands, (H, W)


def haar_idwt(bands: torch.Tensor, out_hw: Tuple[int, int]):
    """
    Inverse Haar DWT.

    Args:
        bands: [B, 4C, H/2, W/2]
        out_hw: original spatial size

    Returns:
        x: [B, C, H, W]
    """
    B, C4, H2, W2 = bands.shape
    assert C4 % 4 == 0, "The channel number of wavelet bands must be 4C."

    C = C4 // 4
    LL, LH, HL, HH = bands.chunk(4, dim=1)

    x = torch.zeros(
        B,
        C,
        H2 * 2,
        W2 * 2,
        device=bands.device,
        dtype=bands.dtype,
    )

    x[:, :, 0::2, 0::2] = (LL - LH - HL + HH) * 0.5
    x[:, :, 0::2, 1::2] = (LL - LH + HL - HH) * 0.5
    x[:, :, 1::2, 0::2] = (LL + LH - HL - HH) * 0.5
    x[:, :, 1::2, 1::2] = (LL + LH + HL + HH) * 0.5

    H, W = out_hw
    return x[:, :, :H, :W]


# =========================================================
# 2. Spatial Branch
# =========================================================

class SpatialBranch(nn.Module):
    """
    Spatial branch:
        DWConv 5x5
        PWConv 1x1

    All Conv-Norm-Act blocks are aligned with EdgeCrafter:
        ConvNormLayer_fuse = Conv2d + BatchNorm2d + get_activation(act)
    """
    def __init__(self, dim: int, act: str = "silu"):
        super().__init__()

        self.dw5 = ConvNormLayer_fuse(
            dim,
            dim,
            kernel_size=5,
            stride=1,
            g=dim,
            act=None,
        )

        self.pw = ConvNormLayer_fuse(
            dim,
            dim,
            kernel_size=1,
            stride=1,
            g=1,
            padding=0,
            act=act,
        )

    def forward(self, x):
        x = self.dw5(x)
        x = self.pw(x)
        return x


# =========================================================
# 3. Wavelet Branch
# =========================================================

class FreqBranch(nn.Module):
    """
    Wavelet branch:
        DWT
        -> LL/LH/HL/HH
        -> sub-band gate
        -> Conv fusion
        -> IDWT / resize fusion

    Norm/activation follow EdgeCrafter:
        ConvNormLayer_fuse + act='silu'
    """
    def __init__(
        self,
        dim: int,
        reduction: int = 4,
        reconstruct: str = "idwt",
        act: str = "silu",
    ):
        super().__init__()

        assert reconstruct in ["idwt", "resize"]
        self.reconstruct = reconstruct

        hidden = max(dim // reduction, 8)

        # Gate branch uses Conv2d because it is channel attention rather than Conv-Norm-Act.
        # Sigmoid is necessary for gating and should be kept.
        self.band_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * dim, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 4 * dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.band_reduce = ConvNormLayer_fuse(
            4 * dim,
            4 * dim,
            kernel_size=1,
            stride=1,
            padding=0,
            act=act,
        )

        self.band_dw = ConvNormLayer_fuse(
            4 * dim,
            4 * dim,
            kernel_size=3,
            stride=1,
            g=4 * dim,
            act=act,
        )

        # Keep channel = 4C for IDWT.
        # Final act=None is more stable before reconstruction.
        self.band_expand = ConvNormLayer_fuse(
            4 * dim,
            4 * dim,
            kernel_size=1,
            stride=1,
            padding=0,
            act=None,
        )

        if reconstruct == "resize":
            self.resize_proj = ConvNormLayer_fuse(
                4 * dim,
                dim,
                kernel_size=1,
                stride=1,
                padding=0,
                act=act,
            )

        self.out_proj = ConvNormLayer_fuse(
            dim,
            dim,
            kernel_size=1,
            stride=1,
            padding=0,
            act=act,
        )

    def forward(self, x):
        bands, orig_hw = haar_dwt(x)

        gate = self.band_gate(bands)
        bands = bands * gate

        bands = self.band_reduce(bands)
        bands = self.band_dw(bands)
        bands = self.band_expand(bands)

        if self.reconstruct == "idwt":
            y = haar_idwt(bands, orig_hw)
        else:
            y = self.resize_proj(bands)
            y = F.interpolate(
                y,
                size=orig_hw,
                mode="bilinear",
                align_corners=False,
            )

        y = self.out_proj(y)
        return y


# =========================================================
# 4. Band-Decoupled Wavelet Branch
# =========================================================

class SpatialSubBandGate(nn.Module):
    """Spatially preserved gate over LL/LH/HL/HH wavelet bands."""
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(4 * dim, hidden, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 4 * dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, bands):
        return self.gate(bands)


class BandDecoupledFreqBranch(nn.Module):
    """
    Band-decoupled wavelet branch:
        DWT
        -> spatial sub-band gate
        -> independent LL/LH/HL/HH processing
        -> IDWT
        -> output projection
    """
    def __init__(self, dim: int, reduction: int = 4, act: str = "silu"):
        super().__init__()
        self.band_gate = SpatialSubBandGate(dim, reduction=reduction)
        self.ll_branch = self._make_subband_branch(dim, act)
        self.lh_branch = self._make_subband_branch(dim, act)
        self.hl_branch = self._make_subband_branch(dim, act)
        self.hh_branch = self._make_subband_branch(dim, act)
        self.out_proj = ConvNormLayer_fuse(
            dim,
            dim,
            kernel_size=1,
            stride=1,
            padding=0,
            act=act,
        )

    @staticmethod
    def _make_subband_branch(dim: int, act: str):
        return nn.Sequential(
            ConvNormLayer_fuse(
                dim,
                dim,
                kernel_size=1,
                stride=1,
                padding=0,
                act=act,
            ),
            ConvNormLayer_fuse(
                dim,
                dim,
                kernel_size=3,
                stride=1,
                padding=1,
                g=dim,
                act=act,
            ),
        )

    def forward(self, x):
        bands, orig_hw = haar_dwt(x)
        weights = self.band_gate(bands)

        LL, LH, HL, HH = bands.chunk(4, dim=1)
        w_LL, w_LH, w_HL, w_HH = weights.chunk(4, dim=1)

        LL = self.ll_branch(LL * w_LL)
        LH = self.lh_branch(LH * w_LH)
        HL = self.hl_branch(HL * w_HL)
        HH = self.hh_branch(HH * w_HH)

        bands = torch.cat([LL, LH, HL, HH], dim=1)
        y = haar_idwt(bands, orig_hw)
        y = self.out_proj(y)
        return y

# =========================================================
# 5. FSAS Branch
# =========================================================

class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class FSASBranch(nn.Module):
    """
    FSAS-like branch:
        Q, K, V
        -> FFT(Q), FFT(K)
        -> frequency-domain correlation
        -> IFFT
        -> correlation map modulates V

    Recommended:
        C2: use_fsas=False
        C3/C4: use_fsas=True
    """
    def __init__(self, dim: int, act: str = "silu"):
        super().__init__()

        # Use raw conv here because qkv generation should not be normalized separately
        # before splitting into Q/K/V.
        self.qkv_pw = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)

        self.qkv_dw = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=False,
        )

        self.corr_norm = LayerNorm2d(dim)
        self.log_scale = nn.Parameter(torch.zeros(1))

        self.proj = ConvNormLayer_fuse(
            dim,
            dim,
            kernel_size=1,
            stride=1,
            padding=0,
            act=act,
        )

    def forward(self, x):
        dtype = x.dtype
        B, C, H, W = x.shape

        qkv = self.qkv_dw(self.qkv_pw(x))
        q, k, v = qkv.chunk(3, dim=1)

        qf = torch.fft.rfft2(q.float(), norm="ortho")
        kf = torch.fft.rfft2(k.float(), norm="ortho")

        corr = torch.fft.irfft2(
            qf * torch.conj(kf),
            s=(H, W),
            norm="ortho",
        )

        corr = self.corr_norm(corr.to(dtype))
        scale_factor = torch.exp(0.5 * torch.tanh(self.log_scale))
        corr = torch.sigmoid(corr * scale_factor)

        y = v * corr
        y = self.proj(y)
        return y


# =========================================================
# 6. Full FSEM Block
# =========================================================

class FSEMBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        out_dim: int,
        use_fsas: bool = False,
        base_fft_size: Tuple[int, int] = (32, 32),
        freq_reconstruct: str = "idwt",
        act: str = "silu",
        init_scale: float = 1e-3,
    ):
        super().__init__()

        self.dw3 = ConvNormLayer_fuse(
            dim,
            dim,
            kernel_size=3,
            stride=2,
            padding=1,
            g=dim,
            act=None,
        )
        self.pw = ConvNormLayer_fuse(
            dim,
            out_dim,
            kernel_size=1,
            stride=1,
            g=1,
            padding=0,
            act=act,
        )

        self.spatial_branch = SpatialBranch(out_dim, act=act)

        self.freq_branch = (
            BandDecoupledFreqBranch(out_dim, act=act)
            if not use_fsas
            else FSASBranch(out_dim, act=act)
        )

        fuse_in = out_dim * 2

        # Final fusion should be linear Conv-BN without activation.
        # This is more stable for residual enhancement.
        self.fusion = ConvNormLayer_fuse(
            fuse_in,
            out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            act=None,
        )

        self.alpha = nn.Parameter(torch.ones(1) * init_scale)

    def forward(self, x):
        x = self.pw(self.dw3(x))
        x_spatial = self.spatial_branch(x)
        x_freq = self.freq_branch(x)
        # x_global = self.global_frequency_branch(x)

        feats = [x_spatial, x_freq]

        delta = self.fusion(torch.cat(feats, dim=1))

        return x + self.alpha * delta

class Stem(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=1, bias=False)
        self.norm = nn.SyncBatchNorm(out_channels)
        self.gelu = nn.GELU()
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
        self.conv_3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.conv_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=1)
        self.use_norm = use_norm

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_norm:
            # out = self.norm(out)
            if hasattr(self, 'norm'):
                out = self.norm(out)

        out = self.gelu(out)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)

        return out

class SpatialFreqModule(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()

        # 1/4
        self.stem = Stem(3, inplanes, use_norm=True)
        # 1/8
        self.conv2 = FSEMBlock(inplanes, out_dim= 2 * inplanes, use_fsas=False)
        # 1/16
        self.conv3 = FSEMBlock(2 * inplanes, out_dim= 4 * inplanes, use_fsas=False)
        # 1/32
        self.conv4 = FSEMBlock(4 * inplanes, out_dim= 4 * inplanes, use_fsas=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)     # 1/8
        c3 = self.conv3(c2)     # 1/16
        c4 = self.conv4(c3)     # 1/32

        return c2, c3, c4


class SemanticC2Refinement(nn.Module):
    """
    Lightweight residual refinement for the upsampled C2 semantic feature.

    This module is intentionally applied only to the ViT semantic branch before
    concatenating it with the C2 detail feature from SpatialFreqModule. It keeps
    the A3 [10, 11] semantic source unchanged and only mitigates the smoothness
    introduced by bilinear upsampling from the 1/16 ViT token map to the 1/8 C2
    semantic map.
    """
    def __init__(self, dim: int, act: str = "silu", init_scale: float = 1e-3):
        super().__init__()
        self.dw3 = ConvNormLayer_fuse(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            g=dim,
            act=act,
        )
        self.pw = ConvNormLayer_fuse(
            dim,
            dim,
            kernel_size=1,
            stride=1,
            padding=0,
            act=None,
        )
        self.beta = nn.Parameter(torch.ones(1) * init_scale)

    def forward(self, x):
        return x + self.beta * self.pw(self.dw3(x))


@register()
class ViTAdapter(nn.Module):
    
    ecvit_url = {
        # detection backbone
        "ecvitt": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvitt.pth",
        "ecvittplus": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvittplus.pth",
        "ecvits": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvits.pth",
        "ecvitsplus": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvitsplus.pth",
        
        # segmentation backbone
        "ecseg_vitt": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_vitt.pth",
        "ecseg_vittplus": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_vittplus.pth",
        "ecseg_vits": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_vits.pth",
        "ecseg_vitsplus": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_vitsplus.pth",}
    
    def __init__(
        self,
        name,
        weights_path=None,
        interaction_indexes=[10, 11],
        embed_dim=192,
        num_heads=3,
        patch_size=16,
        proj_dim=None,
        num_levels=3,
        embed_layer='ConvPyramidPatchEmbed',
        use_sta=True,
        conv_inplane=16,
        hidden_dim=None,
        ffn_layer='mlp',
        ffn_ratio=4,
        skip_load_backbone=False,
        use_c2_sem_refine=False,
        c2_sem_refine_init_scale=1e-3,
        **kwargs
    ):
        super().__init__()
        
        self.name = name
        
        if embed_layer not in EMBED_LAYER_REGISTRY:
            raise ValueError(f"Unknown embed_layer: {embed_layer}. Available: {list(EMBED_LAYER_REGISTRY)}")
        if ffn_layer not in FFN_LAYER_REGISTRY:
            raise ValueError(f"Unknown ffn_layer: {ffn_layer}. Available: {list(FFN_LAYER_REGISTRY)}")
        embed_layer = EMBED_LAYER_REGISTRY[embed_layer]
        ffn_layer = FFN_LAYER_REGISTRY[ffn_layer]
        
        
        self.backbone = VisionTransformer(embed_dim=embed_dim, 
                                          num_heads=num_heads, 
                                          return_layers=interaction_indexes, 
                                          patch_size=patch_size, 
                                          embed_layer=embed_layer,
                                          ffn_layer=ffn_layer,
                                          ffn_ratio=ffn_ratio,
                                          **kwargs)
        if not skip_load_backbone:
            self._load_weights(weights_path)
            
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        self.num_levels = num_levels
        
        if num_levels != 3:
            raise NotImplementedError("Only support num_levels=3 for ViTAdapter now.")

        self.use_c2_sem_refine = use_c2_sem_refine
        self.c2_sem_refine = (
            SemanticC2Refinement(
                embed_dim,
                init_scale=c2_sem_refine_init_scale,
            )
            if use_c2_sem_refine
            else nn.Identity()
        )
        if use_c2_sem_refine:
            print(
                f"Using C2 semantic refinement with "
                f"init_scale={c2_sem_refine_init_scale}"
            )

        # self.proj_dim = [proj_dim] * num_levels if proj_dim is not None else [embed_dim]

        # self.projector = nn.ModuleList([ConvNormLayer_fuse(embed_dim, dim, kernel_size=1, stride=1) for dim in self.proj_dim])

        self.use_sta = use_sta
        if use_sta:
            print(f"Using Lite Spatial Prior Module with inplanes={conv_inplane}")
            self.sta = SpatialFreqModule(inplanes=conv_inplane)
        else:
            conv_inplane = 0

        # linear projection
        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.convs = nn.ModuleList([
            nn.Conv2d(embed_dim + conv_inplane*2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        # norm
        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim)
        ])
        
    def _load_weights(self, weights_path):
        if self.name not in self.ecvit_url:
            raise ValueError(f"Unknown model name: {self.name}. Available: {list(self.ecvit_url)}")
        url = self.ecvit_url[self.name]
        
        if weights_path is None:
            print(
                "="*80 + "\n",
                "❌❌❌ WARNING: Pretrained ViT weights not loaded! ❌❌❌\n"
                "The model is running with randomly initialized parameters.\n"
                "This will severely degrade performance and convergence!\n"
                f"Please download the model manually from {url}.\n"
                "If you want to train from scratch, please set `skip_load_backbone=True` to skip this warning.\n"*2,
                "="*80,
                sep="")
            
            return
        
        path = Path(weights_path)
        if path.exists():
            state = torch.load(path, weights_only=True, map_location="cpu")
            self.backbone.load_state_dict(state, strict=True)
            print(
                "=" * 80 + "\n",
                "✅ Pretrained ViT weights loaded successfully!\n"
                f"📦 Weights file: {path}\n",
                "=" * 80
            )
        else:
            model_dir = Path(__file__).resolve().parents[2] / "ecvits"
            print(
                f"\nTrying to load pretrained ViT weights from {url}. "
                "If this fails, please download the model manually.\n"
                "If you have already downloaded the model, please check that `weights_path` is correct and the file exists.")
            
            if safe_get_rank() == 0:
                torch.hub.load_state_dict_from_url(
                    url, map_location="cpu", model_dir=model_dir, weights_only=True,)
                
            if dist_utils.is_dist_available_and_initialized():
                torch.distributed.barrier()

            model_path = model_dir / url.split("/")[-1]
            state = torch.load(model_path, map_location="cpu")
            self.backbone.load_state_dict(state, strict=True)
            print(
                "=" * 80 + "\n",
                "✅ Pretrained ViT weights loaded successfully!\n"
                f"📦 Weights downloaded to: {model_dir}\n",
                "=" * 80,
                sep="")


    
    def forward(self, x):
        
        H_c, W_c = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs = x.shape[0]

        return_layers = self.backbone(x)
        
        # fused_feats = (return_layers[0] + return_layers[1]) / 2
        fused_feats = torch.mean(torch.stack(return_layers), dim=0)

        proj_feats = []
        fused_feats = fused_feats.transpose(1, 2).contiguous().view(bs, -1, H_c, W_c)  # [B, D, H, W]
        for i in range(self.num_levels):
            scale = 2 ** (1 - i)
            resize_H = int(H_c * scale)
            resize_W = int(W_c * scale)
            feature = F.interpolate(
                fused_feats,
                size=[resize_H, resize_W],
                mode="bilinear",
                align_corners=False,
            )
            if i == 0:
                feature = self.c2_sem_refine(feature)
            proj_feats.append(feature)
            
        # fusion
        fused_feats = []
        if self.use_sta:
            detail_feats = self.sta(x)
            for sem_feat, detail_feat in zip(proj_feats, detail_feats):
                # print(sem_feat.shape, detail_feat.shape)
                fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
        else:
            fused_feats = proj_feats

        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))

        backbone_feats = [c2, c3, c4]

        return backbone_feats
        
    
