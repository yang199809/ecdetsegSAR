"""Object-aware scale-adaptive frequency-spatial enhancement modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hybrid_encoder import ConvNormLayer_fuse

__all__ = [
    "haar_dwt",
    "haar_idwt",
    "LayerNorm2d",
    "SpatialBranch",
    "SpatialSubBandGate",
    "ObjectGate",
    "ObjectAwareWaveletBranch",
    "GlobalFrequencyBranch",
    "FSASBranch",
    "OSAFSEMBlock",
]


def haar_dwt(x):
    """Apply an orthonormal Haar DWT and return LL/LH/HL/HH bands."""
    orig_hw = x.shape[-2:]
    pad_h = orig_hw[0] % 2
    pad_w = orig_hw[1] % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (-x00 - x01 + x10 + x11) * 0.5
    hl = (-x00 + x01 - x10 + x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return torch.cat((ll, lh, hl, hh), dim=1), orig_hw


def haar_idwt(bands, orig_hw):
    """Invert bands ordered as LL/LH/HL/HH and crop replicated padding."""
    if bands.shape[1] % 4 != 0:
        raise ValueError("The number of wavelet-band channels must be divisible by 4.")

    ll, lh, hl, hh = bands.chunk(4, dim=1)
    x00 = (ll - lh - hl + hh) * 0.5
    x01 = (ll - lh + hl - hh) * 0.5
    x10 = (ll + lh - hl - hh) * 0.5
    x11 = (ll + lh + hl + hh) * 0.5

    b, c, h, w = ll.shape
    x = bands.new_empty((b, c, h * 2, w * 2))
    x[..., 0::2, 0::2] = x00
    x[..., 0::2, 1::2] = x01
    x[..., 1::2, 0::2] = x10
    x[..., 1::2, 1::2] = x11
    return x[..., : orig_hw[0], : orig_hw[1]]


class LayerNorm2d(nn.Module):
    """Layer normalization over channels at every spatial location."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class SpatialBranch(nn.Module):
    def __init__(self, dim, act="silu"):
        super().__init__()
        self.dw3 = ConvNormLayer_fuse(dim, dim, 3, 1, g=dim, act=act)
        self.dw5 = ConvNormLayer_fuse(dim, dim, 5, 1, g=dim, act=act)
        self.pw = ConvNormLayer_fuse(dim, dim, 1, 1, padding=0, act=act)

    def forward(self, x):
        return self.pw(self.dw3(x) + self.dw5(x))


class SpatialSubBandGate(nn.Module):
    """Predict spatially varying weights for every explicit wavelet band."""

    def __init__(self, dim, act="silu"):
        super().__init__()
        band_dim = 4 * dim
        self.gate = nn.Sequential(
            ConvNormLayer_fuse(band_dim, band_dim, 3, 1, g=band_dim, act=act),
            nn.Conv2d(band_dim, band_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, bands):
        return self.gate(bands)


class ObjectGate(nn.Module):
    """Predict an object-aware spatial map without global pooling."""

    def __init__(self, dim, act="silu"):
        super().__init__()
        self.gate = nn.Sequential(
            ConvNormLayer_fuse(dim, dim, 3, 1, g=dim, act=act),
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.gate(x)


class ObjectAwareWaveletBranch(nn.Module):
    def __init__(self, dim, act="silu"):
        super().__init__()
        band_dim = 4 * dim
        self.band_gate = SpatialSubBandGate(dim, act=act)
        self.object_gate = ObjectGate(dim, act=act)
        self.scatter_proj = nn.Conv2d(dim, 1, 1)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.band_conv = nn.Sequential(
            ConvNormLayer_fuse(band_dim, band_dim, 1, 1, padding=0, act=act),
            ConvNormLayer_fuse(band_dim, band_dim, 3, 1, g=band_dim, act=act),
            ConvNormLayer_fuse(band_dim, band_dim, 1, 1, padding=0, act=None),
        )
        self.out_proj = ConvNormLayer_fuse(dim, dim, 1, 1, padding=0, act=act)

    def forward(self, x, return_aux=False):
        bands, orig_hw = haar_dwt(x)
        ll, lh, hl, hh = bands.chunk(4, dim=1)
        high_response = lh.abs() + hl.abs() + hh.abs()
        scatter_map = torch.sigmoid(self.scatter_proj(high_response))
        scatter_map = F.interpolate(
            scatter_map,
            size=orig_hw,
            mode="bilinear",
            align_corners=False,
        )

        band_weight = self.band_gate(bands)
        w_ll, w_lh, w_hl, w_hh = band_weight.chunk(4, dim=1)
        ll = ll * w_ll
        lh = lh * w_lh
        hl = hl * w_hl
        hh = hh * w_hh

        object_map = self.object_gate(x)
        object_dwt = F.interpolate(
            object_map,
            size=ll.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        gamma = self.gamma.clamp(0.0, 1.0)
        lh = lh * (1.0 + gamma * object_dwt)
        hl = hl * (1.0 + gamma * object_dwt)
        hh = hh * (1.0 + gamma * object_dwt)

        bands = self.band_conv(torch.cat((ll, lh, hl, hh), dim=1))
        y = self.out_proj(haar_idwt(bands, orig_hw))
        if return_aux:
            return y, {"object_map": object_map, "band_weight": band_weight, "scatter_map": scatter_map}
        return y


class GlobalFrequencyBranch(nn.Module):
    def __init__(self, dim, base_fft_size=(32, 32)):
        super().__init__()
        base_h, base_w = base_fft_size
        complex_filter = torch.zeros(1, dim, base_h, base_w // 2 + 1, 2)
        complex_filter[..., 0] = 1.0
        self.complex_filter = nn.Parameter(complex_filter)
        self.scene_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid(),
        )

    def _resize_filter(self, h, w):
        target_w = w // 2 + 1
        base = self.complex_filter.permute(0, 1, 4, 2, 3)
        base = base.reshape(-1, 2, base.shape[-2], base.shape[-1])
        base = F.interpolate(base, size=(h, target_w), mode="bilinear", align_corners=False)
        base = base.reshape(1, self.complex_filter.shape[1], 2, h, target_w)
        return torch.view_as_complex(base.permute(0, 1, 3, 4, 2).contiguous())

    def forward(self, x):
        h, w = x.shape[-2:]
        xf = torch.fft.rfft2(x.float(), norm="ortho")
        complex_filter = self._resize_filter(h, w)
        scene_gate = self.scene_gate(x).float()
        y = torch.fft.irfft2(
            xf * complex_filter * (1.0 + scene_gate),
            s=(h, w),
            norm="ortho",
        )
        return y.to(dtype=x.dtype)


class FSASBranch(nn.Module):
    """Frequency-spatial autocorrelation branch with LayerNorm2d."""

    def __init__(self, dim, act="silu"):
        super().__init__()
        self.qkv_pw = ConvNormLayer_fuse(dim, 3 * dim, 1, 1, padding=0, act=act)
        self.qkv_dw = ConvNormLayer_fuse(
            3 * dim,
            3 * dim,
            3,
            1,
            g=3 * dim,
            act=act,
        )
        self.corr_norm = LayerNorm2d(dim)
        self.temperature = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.out_proj = ConvNormLayer_fuse(dim, dim, 1, 1, padding=0, act=act)

    def forward(self, x):
        q, k, v = self.qkv_dw(self.qkv_pw(x)).chunk(3, dim=1)
        h, w = x.shape[-2:]
        qf = torch.fft.rfft2(q.float(), norm="ortho")
        kf = torch.fft.rfft2(k.float(), norm="ortho")
        corr = torch.fft.irfft2(qf * torch.conj(kf), s=(h, w), norm="ortho")
        corr = self.corr_norm(corr)
        corr = torch.sigmoid(corr / self.temperature.clamp(min=0.1))
        return self.out_proj(v * corr.to(dtype=v.dtype))


class OSAFSEMBlock(nn.Module):
    """Object-aware Scale-Adaptive Frequency-Spatial Enhancement block."""

    def __init__(
        self,
        dim,
        out_dim,
        use_fsas=False,
        base_fft_size=(32, 32),
        act="silu",
        init_scale=1e-3
    ):
        super().__init__()
        self.use_fsas = use_fsas
        self.spatial_branch = SpatialBranch(out_dim, act=act)
        self.wavelet_branch = ObjectAwareWaveletBranch(out_dim, act=act)
        self.global_frequency_branch = GlobalFrequencyBranch(out_dim, base_fft_size)
        if use_fsas:
            self.fsas_branch = FSASBranch(out_dim, act=act)

        fuse_in = out_dim * (4 if use_fsas else 3)
        self.fusion = ConvNormLayer_fuse(fuse_in, out_dim, 1, 1, padding=0, act=None)
        self.alpha = nn.Parameter(torch.ones(1) * init_scale)
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

    def forward(self, x, return_aux=False):
        x = self.pw(self.dw3(x))
        x_spatial = self.spatial_branch(x)
        if return_aux:
            x_wavelet, aux = self.wavelet_branch(x, return_aux=True)
        else:
            x_wavelet = self.wavelet_branch(x, return_aux=False)
        x_global = self.global_frequency_branch(x)

        features = [x_spatial, x_wavelet, x_global]
        if self.use_fsas:
            features.append(self.fsas_branch(x))

        delta = self.fusion(torch.cat(features, dim=1))
        y = x + self.alpha * delta
        # scale = self.max_scale * torch.tanh(self.alpha / self.max_scale)
        # y = x + scale * delta
        if return_aux:
            return y, aux
        return y
