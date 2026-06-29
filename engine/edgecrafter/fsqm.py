"""
Frequency-guided Scattering Query Modulation for SAR instance segmentation.
"""

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FSQM(nn.Module):
    """Frequency-guided Scattering Query Modulation v1.

    FSQM is inserted once before the transformer decoder. It converts the
    FSEM/A3-enhanced multi-scale features into query-specific scattering priors
    through reference-point guided sampling, then injects the prior into object
    queries with a zero-initialized residual scale.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        query_dim: int,
        hidden_ratio: float = 0.25,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.query_dim = query_dim

        hidden_dim = max(int(query_dim * hidden_ratio), 16)
        self.input_proj = nn.ModuleList()
        self.map_heads = nn.ModuleList()

        for c in in_channels:
            if c == query_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(nn.Conv2d(c, query_dim, kernel_size=1))

            self.map_heads.append(
                nn.Sequential(
                    nn.Conv2d(query_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_dim, 1, kernel_size=1),
                )
            )

        self.prior_proj = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.ReLU(inplace=True),
            nn.Linear(query_dim, query_dim),
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.Sigmoid(),
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, queries: torch.Tensor, feats: List[torch.Tensor], ref_points: torch.Tensor):
        if len(feats) != len(self.input_proj):
            raise ValueError(f"FSQM got {len(feats)} feature levels, expected {len(self.input_proj)}")

        if ref_points.size(-1) == 4:
            ref_points = ref_points[..., :2]
        elif ref_points.size(-1) != 2:
            raise ValueError(f"FSQM expects ref_points last dim 2 or 4, got {ref_points.size(-1)}")

        ref_points = ref_points.clamp(0.0, 1.0)
        grid = (ref_points * 2.0 - 1.0).unsqueeze(2)

        scatter_logits = []
        scatter_maps = []
        sampled_scores = []
        sampled_feats = []

        for feat, proj, head in zip(feats, self.input_proj, self.map_heads):
            feat_proj = proj(feat)
            logit = head(feat_proj)
            scatter = torch.sigmoid(logit)

            scatter_logits.append(logit)
            scatter_maps.append(scatter)

            score = F.grid_sample(
                scatter,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(-1).transpose(1, 2)

            local_feat = F.grid_sample(
                feat_proj,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(-1).transpose(1, 2)

            sampled_scores.append(score)
            sampled_feats.append(local_feat)

        scores = torch.cat(sampled_scores, dim=-1)
        weights = scores + 1e-3
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self.eps)

        prior_feat = sampled_feats[0].new_zeros(sampled_feats[0].shape)
        for i, feat_i in enumerate(sampled_feats):
            prior_feat = prior_feat + weights[..., i:i + 1] * feat_i

        prior = self.prior_proj(prior_feat)
        gate = self.gate_proj(prior)
        mod_queries = queries + self.gamma * gate * prior

        aux_outputs = {
            "scatter_maps": scatter_maps,
            "scatter_logits": scatter_logits,
        }
        return mod_queries, aux_outputs


def _per_image_minmax_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    flat = x.flatten(1)
    x_min = flat.amin(dim=1).view(-1, 1, 1, 1)
    x_max = flat.amax(dim=1).view(-1, 1, 1, 1)
    return (x - x_min) / (x_max - x_min + eps)


def _masks_from_targets(gt_masks):
    if isinstance(gt_masks, (list, tuple)) and gt_masks and isinstance(gt_masks[0], dict):
        return [target.get("masks", None) for target in gt_masks]
    return gt_masks


def _to_tensor_masks(mask_item, device: torch.device) -> Optional[torch.Tensor]:
    if mask_item is None:
        return None

    if torch.is_tensor(mask_item):
        masks = mask_item
    elif isinstance(mask_item, (list, tuple)):
        if len(mask_item) == 0:
            return None
        tensors = [m for m in mask_item if m is not None]
        if len(tensors) == 0:
            return None
        masks = torch.stack([torch.as_tensor(m) for m in tensors], dim=0)
    elif hasattr(mask_item, "to_tensor"):
        masks = mask_item.to_tensor(dtype=torch.bool, device=device)
    elif hasattr(mask_item, "masks"):
        masks = torch.as_tensor(mask_item.masks)
    else:
        masks = torch.as_tensor(mask_item)

    masks = masks.to(device=device)
    if masks.numel() == 0:
        return None

    if masks.dim() == 2:
        masks = masks.unsqueeze(0)
    elif masks.dim() == 4 and masks.size(1) == 1:
        masks = masks.squeeze(1)
    elif masks.dim() > 3:
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])

    return masks.float()


def _merge_instance_masks(gt_masks, batch_size: int, image_size: Tuple[int, int], device: torch.device) -> torch.Tensor:
    gt_masks = _masks_from_targets(gt_masks)
    merged_masks = []

    for idx in range(batch_size):
        mask_item = gt_masks[idx] if isinstance(gt_masks, (list, tuple)) and idx < len(gt_masks) else None
        if torch.is_tensor(gt_masks) and gt_masks.dim() >= 3:
            mask_item = gt_masks[idx]

        masks = _to_tensor_masks(mask_item, device)
        if masks is None:
            merged_masks.append(torch.zeros(1, *image_size, device=device))
            continue

        if masks.shape[-2:] != image_size:
            masks = F.interpolate(
                masks.unsqueeze(1),
                size=image_size,
                mode="nearest",
            ).squeeze(1)

        merged_masks.append((masks > 0).any(dim=0, keepdim=True).float())

    return torch.stack(merged_masks, dim=0)


def build_scatter_pseudo_targets(
    images: torch.Tensor,
    gt_masks,
    out_sizes: Iterable[Tuple[int, int]],
    rho: float = 0.15,
    return_object_mask: bool = False,
):
    """Build continuous SAR scattering targets from high-pass response and masks.

    The pseudo-label is M * (rho + (1 - rho) * H), where H is a high-frequency
    response map estimated from the current normalized model input. A later data
    pipeline can replace this with raw SAR intensity without changing FSQM.
    """

    if images is None:
        raise ValueError("images must be provided to build FSQM scatter pseudo targets")

    images = images.float()
    batch_size, _, height, width = images.shape
    device = images.device

    image_gray = images.mean(dim=1, keepdim=True)
    image_gray = _per_image_minmax_norm(image_gray)
    low_freq = F.avg_pool2d(image_gray, kernel_size=7, stride=1, padding=3)
    high = (image_gray - low_freq).abs()
    # TODO: replace min-max normalization with percentile clipping if raw SAR
    # intensity with strong outliers is made available by the data pipeline.
    high = _per_image_minmax_norm(high)

    object_mask = _merge_instance_masks(gt_masks, batch_size, (height, width), device)
    pseudo = object_mask * (rho + (1.0 - rho) * high)

    pseudo_targets = [
        F.interpolate(pseudo, size=tuple(size), mode="bilinear", align_corners=False)
        for size in out_sizes
    ]

    if return_object_mask:
        return pseudo_targets, object_mask
    return pseudo_targets


def _zero_from_aux(scatter_maps: List[torch.Tensor], scatter_logits: List[torch.Tensor]) -> torch.Tensor:
    tensors = list(scatter_maps) + list(scatter_logits)
    if len(tensors) == 0:
        return torch.tensor(0.0)

    zero = tensors[0].sum() * 0.0
    for tensor in tensors[1:]:
        zero = zero + tensor.sum() * 0.0
    return zero


def fsqm_auxiliary_loss(
    aux_outputs: dict,
    images: Optional[torch.Tensor],
    targets,
    rho: float = 0.15,
    bg_weight: float = 0.2,
    aux_weight: float = 0.2,
):
    scatter_maps = aux_outputs.get("scatter_maps", [])
    scatter_logits = aux_outputs.get("scatter_logits", [])
    zero = _zero_from_aux(scatter_maps, scatter_logits)

    if images is None or targets is None or len(scatter_maps) == 0:
        return {
            "loss_fsqm_scatter": zero.detach(),
            "loss_fsqm_bg": zero.detach(),
            "loss_fsqm_aux": zero,
        }

    out_sizes = [scatter.shape[-2:] for scatter in scatter_maps]
    pseudo_targets, object_mask = build_scatter_pseudo_targets(
        images,
        targets,
        out_sizes,
        rho=rho,
        return_object_mask=True,
    )

    if object_mask.sum() == 0:
        return {
            "loss_fsqm_scatter": zero.detach(),
            "loss_fsqm_bg": zero.detach(),
            "loss_fsqm_aux": zero,
        }

    scatter_loss = zero
    bg_loss = zero

    for scatter, logit, pseudo in zip(scatter_maps, scatter_logits, pseudo_targets):
        pseudo = pseudo.to(device=scatter.device, dtype=scatter.dtype)
        scatter_loss = scatter_loss + F.smooth_l1_loss(scatter, pseudo, reduction="mean")

        background = pseudo <= 0
        if background.any():
            bg_logits = logit[background]
            bg_loss = bg_loss + F.binary_cross_entropy_with_logits(
                bg_logits,
                torch.zeros_like(bg_logits),
                reduction="mean",
            )

    total_aux = aux_weight * (scatter_loss + bg_weight * bg_loss)
    return {
        "loss_fsqm_scatter": scatter_loss.detach(),
        "loss_fsqm_bg": bg_loss.detach(),
        "loss_fsqm_aux": total_aux,
    }
