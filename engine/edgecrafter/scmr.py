"""
Query-conditioned Scattering-aware Contextual Mask Refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCMR(nn.Module):
    """Lightweight instance-level residual mask refinement.

    SCMR uses three signals after coarse mask prediction: a DWT-derived
    scattering prior, masked object/background contextual pooling, and
    query-conditioned FiLM. It predicts only a residual correction to the
    coarse logits, so the zero-initialized output keeps the initial behavior
    close to the original mask head.
    """

    def __init__(
        self,
        c_low: int,
        q_dim: int,
        hidden_dim: int = 128,
        ring_kernel: int = 7,
        film_scale: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        if ring_kernel % 2 == 0:
            raise ValueError("SCMR ring_kernel must be odd.")

        self.ring_kernel = ring_kernel
        self.film_scale = film_scale
        self.eps = eps

        self.low_proj = nn.Conv2d(c_low, hidden_dim, kernel_size=1)
        self.mask_proj = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.scatter_proj = nn.Conv2d(1, hidden_dim, kernel_size=1)

        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.context_mlp = nn.Sequential(
            nn.Linear(c_low * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.query_mlp = nn.Sequential(
            nn.LayerNorm(q_dim),
            nn.Linear(q_dim, hidden_dim * 2),
        )
        self.out = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        nn.init.zeros_(self.context_mlp[-1].weight)
        nn.init.zeros_(self.context_mlp[-1].bias)
        nn.init.zeros_(self.query_mlp[-1].weight)
        nn.init.zeros_(self.query_mlp[-1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _masked_avg_pool(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        numerator = (feat * mask).flatten(2).sum(dim=-1)
        denominator = mask.flatten(2).sum(dim=-1).clamp_min(self.eps)
        return numerator / denominator

    def forward(
        self,
        coarse_mask: torch.Tensor,
        low_feat: torch.Tensor,
        scatter_map: torch.Tensor,
        query_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            coarse_mask: [B, N, Hm, Wm] coarse mask logits.
            low_feat: [B, C, Hl, Wl] low-level high-resolution feature.
            scatter_map: [B, 1, Hs, Ws] DWT-derived scattering response.
            query_embed: [B, N, Cq] object queries.

        Returns:
            [B, N, Hl, Wl] refined mask logits.
        """
        if coarse_mask.dim() == 5 and coarse_mask.size(2) == 1:
            coarse_mask = coarse_mask.squeeze(2)
        if coarse_mask.dim() != 4:
            raise ValueError(f"SCMR expects coarse_mask [B, N, H, W], got {tuple(coarse_mask.shape)}")
        if scatter_map.dim() == 3:
            scatter_map = scatter_map.unsqueeze(1)
        if scatter_map.dim() != 4 or scatter_map.size(1) != 1:
            raise ValueError(f"SCMR expects scatter_map [B, 1, H, W], got {tuple(scatter_map.shape)}")

        b, n, _, _ = coarse_mask.shape
        _, c_low, h_low, w_low = low_feat.shape

        coarse_flat = coarse_mask.reshape(b * n, 1, *coarse_mask.shape[-2:])
        coarse_up = F.interpolate(
            coarse_flat,
            size=(h_low, w_low),
            mode="bilinear",
            align_corners=False,
        )
        prob = coarse_up.sigmoid()

        scatter = F.interpolate(
            scatter_map,
            size=(h_low, w_low),
            mode="bilinear",
            align_corners=False,
        )
        scatter_rep = scatter.repeat_interleave(n, dim=0)
        low_rep = low_feat.repeat_interleave(n, dim=0)

        ring_pad = self.ring_kernel // 2
        bg_ring = F.max_pool2d(prob, self.ring_kernel, stride=1, padding=ring_pad) - prob
        bg_ring = bg_ring.clamp(0.0, 1.0)

        # Masked object/background context keeps the instance support explicit.
        f_obj = self._masked_avg_pool(low_rep, prob)
        f_bg = self._masked_avg_pool(low_rep, bg_ring)
        ctx = torch.cat([f_obj, f_bg, f_obj - f_bg, f_obj * f_bg], dim=1)
        gamma_c, beta_c = self.context_mlp(ctx).chunk(2, dim=1)

        query_flat = query_embed.reshape(b * n, -1)
        gamma_q, beta_q = self.query_mlp(query_flat).chunk(2, dim=1)
        gamma = torch.tanh(gamma_c + gamma_q).view(b * n, -1, 1, 1)
        beta = (beta_c + beta_q).view(b * n, -1, 1, 1)

        # DWT scattering prior and query FiLM refine only the residual logits.
        f_low = self.low_proj(low_rep)
        f_mask = self.mask_proj(prob)
        f_scat = self.scatter_proj(scatter_rep)
        feat = self.fuse(torch.cat([f_low, f_mask, f_scat], dim=1))
        feat = feat * (1.0 + self.film_scale * gamma) + self.film_scale * beta

        delta = self.out(feat)
        refined = coarse_up + delta
        return refined.reshape(b, n, h_low, w_low)
