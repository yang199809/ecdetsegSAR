"""
Lightweight query-pixel mask head for SAR instance segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseConvBlock(nn.Module):
    """ConvNeXt-style depthwise block adapted from EdgeCrafter's mask head."""

    def __init__(self, channels: int, layer_scale_init_value: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pwconv = nn.Linear(channels, channels)
        self.act = nn.GELU()
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(channels)) \
            if layer_scale_init_value > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act(self.pwconv(self.norm(x)))
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + x


class QueryMLPBlock(nn.Module):
    """Residual query refinement block used before query-pixel dot product."""

    def __init__(self, channels: int, layer_scale_init_value: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, channels * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(channels * 4, channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(channels)) \
            if layer_scale_init_value > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc2(self.act(self.fc1(self.norm(x))))
        if self.gamma is not None:
            x = self.gamma * x
        return residual + x


class SARSegmentationHead(nn.Module):
    """Query-pixel dot-product mask head.

    Args:
        hidden_dim: Channel dimension of DEIMv2 decoder queries and encoded features.
        mask_hidden_dim: Shared embedding dimension for pixels and queries.
        mask_output_stride: Output stride relative to the input image. Stage 1 uses stride 4.
        mask_num_blocks: Number of lightweight depthwise refinement blocks.
    """

    def __init__(
        self,
        hidden_dim: int = 192,
        mask_hidden_dim: int = 128,
        mask_output_stride: int = 4,
        mask_num_blocks: int = 4,
        use_multiscale_fusion: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mask_hidden_dim = mask_hidden_dim
        self.mask_output_stride = mask_output_stride
        self.use_multiscale_fusion = use_multiscale_fusion

        self.pixel_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, mask_hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(mask_hidden_dim),
            nn.GELU(),
        )
        self.lateral_proj = nn.ModuleList([
            nn.Conv2d(hidden_dim, mask_hidden_dim, kernel_size=1, bias=False)
            for _ in range(2)
        ])
        self.pixel_blocks = nn.ModuleList([DepthwiseConvBlock(mask_hidden_dim) for _ in range(mask_num_blocks)])
        self.query_block = QueryMLPBlock(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, mask_hidden_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        self.scale = mask_hidden_dim ** -0.5

    def build_pixel_embedding(
        self,
        spatial_feature: torch.Tensor,
        multiscale_features: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if spatial_feature.shape[1] != self.hidden_dim:
            raise ValueError(
                f"Expected spatial feature channels={self.hidden_dim}, got {spatial_feature.shape[1]}"
            )

        # The highest-resolution encoded feature is stride 8 in DEIMv2; Stage 1 predicts stride 4 masks.
        if self.mask_output_stride == 4:
            spatial_feature = F.interpolate(
                spatial_feature, scale_factor=2.0, mode="bilinear", align_corners=False
            )

        pixel_embed = self.pixel_proj(spatial_feature)
        if self.use_multiscale_fusion and multiscale_features is not None:
            for proj, feat in zip(self.lateral_proj, multiscale_features[1:3]):
                pixel_embed = pixel_embed + F.interpolate(
                    proj(feat), size=pixel_embed.shape[-2:], mode="bilinear", align_corners=False
                )

        return pixel_embed

    def build_query_embedding(self, query_features: torch.Tensor) -> torch.Tensor:
        if query_features.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected query dim={self.hidden_dim}, got {query_features.shape[-1]}"
            )
        return self.query_proj(self.query_block(query_features))

    def _apply_pixel_block(self, pixel_embed: torch.Tensor, block_index: int | None) -> torch.Tensor:
        if len(self.pixel_blocks) == 0:
            return pixel_embed
        if block_index is None:
            blocks = self.pixel_blocks
        else:
            blocks = [self.pixel_blocks[min(block_index, len(self.pixel_blocks) - 1)]]
        for block in blocks:
            pixel_embed = block(pixel_embed)
        return pixel_embed

    def forward_layers(
        self,
        spatial_feature: torch.Tensor,
        query_layers: torch.Tensor | list[torch.Tensor],
        multiscale_features: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """EdgeCrafter-style progressive mask prediction.

        Dense spatial features are projected and fused once, then refined
        progressively while each decoder layer emits its own query masks.
        """
        if isinstance(query_layers, torch.Tensor):
            if query_layers.dim() != 4:
                raise ValueError(f"Expected query_layers [L, B, Q, C], got {tuple(query_layers.shape)}")
            query_iter = query_layers.unbind(0)
        else:
            query_iter = list(query_layers)
            if not query_iter:
                raise ValueError("query_layers must contain at least one decoder layer.")

        pixel_embed = self.build_pixel_embedding(spatial_feature, multiscale_features)

        masks = []
        for layer_idx, query_features in enumerate(query_iter):
            pixel_embed = self._apply_pixel_block(pixel_embed, layer_idx)
            query_embed = self.build_query_embedding(query_features)
            masks.append(torch.einsum("bchw,bqc->bqhw", pixel_embed, query_embed) * self.scale + self.bias)

        return torch.stack(masks, dim=0)

    def forward(
        self,
        spatial_feature: torch.Tensor,
        query_features: torch.Tensor,
        multiscale_features: list[torch.Tensor] | None = None,
        block_index: int | None = None,
    ) -> torch.Tensor:
        pixel_embed = self.build_pixel_embedding(spatial_feature, multiscale_features)
        pixel_embed = self._apply_pixel_block(pixel_embed, block_index)

        query_embed = self.build_query_embedding(query_features)
        return torch.einsum("bchw,bqc->bqhw", pixel_embed, query_embed) * self.scale + self.bias
