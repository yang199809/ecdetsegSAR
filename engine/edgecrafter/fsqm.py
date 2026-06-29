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
    queries with a bounded residual scale and query-level scattering confidence.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        query_dim: int,
        hidden_ratio: float = 0.25,
        use_conf_gate: bool = True,
        max_gamma: float = 0.05,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.query_dim = query_dim
        self.use_conf_gate = use_conf_gate
        self.max_gamma = max_gamma

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
        self.prior_norm = nn.LayerNorm(query_dim)
        self.gate_proj = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.Sigmoid(),
        )

        self.raw_gamma = nn.Parameter(torch.zeros(1))

    def forward(self, queries: torch.Tensor, feats: List[torch.Tensor], ref_points: torch.Tensor):
        if len(feats) != len(self.input_proj):
            raise ValueError(f"FSQM got {len(feats)} feature levels, expected {len(self.input_proj)}")

        if ref_points.size(-1) == 4:
            fsqm_ref_points = ref_points[..., :2].detach()
        elif ref_points.size(-1) == 2:
            fsqm_ref_points = ref_points.detach()
        else:
            raise ValueError(f"FSQM expects ref_points last dim 2 or 4, got {ref_points.size(-1)}")

        fsqm_ref_points = fsqm_ref_points.clamp(0.0, 1.0)
        grid = (fsqm_ref_points * 2.0 - 1.0).unsqueeze(2)

        scatter_logits = []
        scatter_maps = []
        sampled_scores = []
        sampled_feats = []

        for feat, proj, head in zip(feats, self.input_proj, self.map_heads):
            feat_proj = proj(feat)
            logit = head(feat_proj)
            scatter = torch.sigmoid(logit)
            scatter_dilated = F.max_pool2d(scatter, kernel_size=3, stride=1, padding=1)

            scatter_logits.append(logit)
            scatter_maps.append(scatter)

            score = F.grid_sample(
                scatter_dilated,
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
        scatter_conf = scores.max(dim=-1, keepdim=True).values
        if not self.use_conf_gate:
            scatter_conf = torch.ones_like(scatter_conf)

        weights = scores + 1e-3
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self.eps)

        prior_feat = sampled_feats[0].new_zeros(sampled_feats[0].shape)
        for i, feat_i in enumerate(sampled_feats):
            prior_feat = prior_feat + weights[..., i:i + 1] * feat_i

        prior = self.prior_proj(prior_feat)
        prior = self.prior_norm(prior)
        gate = self.gate_proj(prior)
        gamma = self.max_gamma * torch.tanh(self.raw_gamma)
        modulation = gamma * scatter_conf * gate * prior
        mod_queries = queries + modulation

        aux_outputs = {
            "scatter_maps": scatter_maps,
            "scatter_logits": scatter_logits,
            "scatter_conf": scatter_conf,
            "base_feat": feats[0].detach(),
        }
        return mod_queries, aux_outputs


def _per_image_minmax_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    flat = x.flatten(1)
    x_min = flat.amin(dim=1).view(-1, 1, 1, 1)
    x_max = flat.amax(dim=1).view(-1, 1, 1, 1)
    return (x - x_min) / (x_max - x_min + eps)


def _source_to_energy(source: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert an image or high-dimensional feature map to one energy channel.

    Raw SAR images [B, 1, H, W] keep their intensity, RGB/repeated images use
    channel mean, and feature maps with C > 3 use RMS/L2 energy to avoid
    cancellation between positive and negative feature responses.
    """
    if source.size(1) == 1:
        return source
    if source.size(1) <= 3:
        return source.mean(dim=1, keepdim=True)

    # RMS/L2 energy avoids channel cancellation for high-dimensional feature
    # maps and is more stable than max activation from a single outlier channel.
    # A future ablation can test source.abs().amax(dim=1, keepdim=True).
    return torch.sqrt(torch.mean(source ** 2, dim=1, keepdim=True) + eps)


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
    source: torch.Tensor,
    gt_masks,
    out_sizes: Iterable[Tuple[int, int]],
    rho: float = 0.4,
    return_object_mask: bool = False,
    return_stats: bool = False,
):
    """Build continuous SAR scattering targets from high-pass response and masks.

    The pseudo-label is M * (rho + (1 - rho) * H), where H is a high-frequency
    response map estimated from either the current normalized model input or a
    detached feature fallback when images are not available.
    """

    if source is None:
        raise ValueError("source must be provided to build FSQM scatter pseudo targets")

    source = source.detach().float()
    batch_size, channels, height, width = source.shape
    device = source.device

    source_energy = _source_to_energy(source)
    source_energy = _per_image_minmax_norm(source_energy)
    low_freq = F.avg_pool2d(source_energy, kernel_size=7, stride=1, padding=3)

    # ReLU high-pass emphasizes local strong scattering peaks without boosting
    # dark valleys. If it becomes too sparse, ablate with:
    # high = (source_energy - low_freq).abs()
    high = F.relu(source_energy - low_freq)
    # TODO: replace min-max normalization with percentile clipping if raw SAR
    # intensity with strong outliers is made available by the data pipeline.
    high = _per_image_minmax_norm(high)

    object_mask = _merge_instance_masks(gt_masks, batch_size, (height, width), device)
    pseudo_targets = []
    object_targets = []
    for size in out_sizes:
        size = tuple(size)
        object_target = F.interpolate(object_mask, size=size, mode="area").clamp_(0.0, 1.0)
        high_target = F.interpolate(high, size=size, mode="bilinear", align_corners=False)
        pseudo_target = object_target * (rho + (1.0 - rho) * high_target)
        pseudo_targets.append(pseudo_target)
        object_targets.append(object_target)

    if return_stats:
        stats = {
            "source_channels": float(channels),
            "source_energy_mean": _debug_value(source_energy.detach().float().mean()),
            "source_energy_max": _debug_value(source_energy.detach().float().max()),
            "high_mean": _debug_value(high.detach().float().mean()),
            "high_max": _debug_value(high.detach().float().max()),
        }
        if return_object_mask:
            return pseudo_targets, object_mask, object_targets, stats
        return pseudo_targets, stats

    if return_object_mask:
        return pseudo_targets, object_mask, object_targets
    return pseudo_targets


def _zero_from_aux(scatter_maps: List[torch.Tensor], scatter_logits: List[torch.Tensor]) -> torch.Tensor:
    tensors = list(scatter_maps) + list(scatter_logits)
    if len(tensors) == 0:
        return torch.tensor(0.0)

    zero = tensors[0].sum() * 0.0
    for tensor in tensors[1:]:
        zero = zero + tensor.sum() * 0.0
    return zero


def _dice_loss_from_probs(inputs: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2.0 * (inputs * targets).sum(dim=1)
    denominator = inputs.sum(dim=1) + targets.sum(dim=1)
    return (1.0 - (numerator + eps) / (denominator + eps)).mean()


def _debug_value(value: torch.Tensor) -> float:
    return float(value.detach().float().item())


def _debug_print(stats: dict):
    stats_str = " ".join(
        f"{key}={value}" if isinstance(value, str) else f"{key}={value:.6f}"
        for key, value in stats.items()
    )
    print(f"[FSQM-v1.1 debug] {stats_str}", flush=True)


def fsqm_auxiliary_loss(
    aux_outputs: dict,
    targets,
    rho: float = 0.4,
    obj_weight: float = 0.2,
    bg_weight: float = 0.2,
    aux_weight: float = 0.1,
    debug: bool = False,
    training: bool = True,
):
    scatter_maps = aux_outputs.get("scatter_maps", [])
    scatter_logits = aux_outputs.get("scatter_logits", [])
    zero = _zero_from_aux(scatter_maps, scatter_logits)

    if targets is None or len(scatter_maps) == 0:
        if debug:
            _debug_print({
                "pseudo_source_type": "none",
                "source_channels": 0.0,
                "source_energy_mean": 0.0,
                "source_energy_max": 0.0,
                "high_mean": 0.0,
                "high_max": 0.0,
                "object_mask_sum": 0.0,
                "pseudo_target_mean": 0.0,
                "pseudo_target_max": 0.0,
                "scatter_map_mean": 0.0,
                "scatter_map_max": 0.0,
                "loss_fsqm_scatter": 0.0,
                "loss_fsqm_obj": 0.0,
                "loss_fsqm_bg": 0.0,
                "loss_fsqm_aux": 0.0,
            })
        return {
            "loss_fsqm_scatter": zero.detach(),
            "loss_fsqm_obj": zero.detach(),
            "loss_fsqm_bg": zero.detach(),
            "loss_fsqm_aux": zero,
        }

    images = aux_outputs.get("images", None)
    base_feat = aux_outputs.get("base_feat", None)
    if images is not None:
        pseudo_source = images
        pseudo_source_type = "image"
    elif base_feat is not None:
        pseudo_source = base_feat
        pseudo_source_type = "feature"
    elif training or debug:
        raise RuntimeError(
            "FSQM auxiliary loss needs either aux_outputs['images'] or "
            "aux_outputs['base_feat'] to build pseudo targets."
        )
    else:
        return {
            "loss_fsqm_scatter": zero.detach(),
            "loss_fsqm_obj": zero.detach(),
            "loss_fsqm_bg": zero.detach(),
            "loss_fsqm_aux": zero,
        }

    out_sizes = [scatter.shape[-2:] for scatter in scatter_maps]
    pseudo_outputs = build_scatter_pseudo_targets(
        pseudo_source,
        targets,
        out_sizes,
        rho=rho,
        return_object_mask=True,
        return_stats=debug,
    )
    if debug:
        pseudo_targets, object_mask, object_targets, source_stats = pseudo_outputs
    else:
        pseudo_targets, object_mask, object_targets = pseudo_outputs
        source_stats = {}

    scatter_loss = zero
    obj_loss = zero
    bg_loss = zero

    if object_mask.sum() == 0:
        if debug:
            scatter_means = [scatter.detach().float().mean() for scatter in scatter_maps]
            scatter_maxes = [scatter.detach().float().max() for scatter in scatter_maps]
            pseudo_means = [pseudo.detach().float().mean() for pseudo in pseudo_targets]
            pseudo_maxes = [pseudo.detach().float().max() for pseudo in pseudo_targets]
            stats = {"pseudo_source_type": pseudo_source_type}
            stats.update(source_stats)
            stats.update({
                "object_mask_sum": 0.0,
                "pseudo_target_mean": _debug_value(torch.stack(pseudo_means).mean()),
                "pseudo_target_max": _debug_value(torch.stack(pseudo_maxes).max()),
                "scatter_map_mean": _debug_value(torch.stack(scatter_means).mean()),
                "scatter_map_max": _debug_value(torch.stack(scatter_maxes).max()),
                "loss_fsqm_scatter": 0.0,
                "loss_fsqm_obj": 0.0,
                "loss_fsqm_bg": 0.0,
                "loss_fsqm_aux": 0.0,
            })
            _debug_print(stats)
        return {
            "loss_fsqm_scatter": zero.detach(),
            "loss_fsqm_obj": zero.detach(),
            "loss_fsqm_bg": zero.detach(),
            "loss_fsqm_aux": zero,
        }

    for scatter, logit, pseudo, object_target in zip(
        scatter_maps,
        scatter_logits,
        pseudo_targets,
        object_targets,
    ):
        scatter = scatter.float()
        logit = logit.float()
        pseudo = pseudo.to(device=scatter.device, dtype=torch.float32)
        object_target = object_target.to(device=scatter.device, dtype=torch.float32)

        scatter_loss = scatter_loss + F.smooth_l1_loss(scatter, pseudo, reduction="mean")
        obj_loss = obj_loss + _dice_loss_from_probs(scatter, object_target)

        background = object_target < 1e-3
        if background.any():
            bg_logits = logit[background]
            bg_loss = bg_loss + F.binary_cross_entropy_with_logits(
                bg_logits,
                torch.zeros_like(bg_logits),
                reduction="mean",
            )

    total_aux = aux_weight * (scatter_loss + obj_weight * obj_loss + bg_weight * bg_loss)

    if debug:
        pseudo_means = [pseudo.detach().float().mean() for pseudo in pseudo_targets]
        pseudo_maxes = [pseudo.detach().float().max() for pseudo in pseudo_targets]
        scatter_means = [scatter.detach().float().mean() for scatter in scatter_maps]
        scatter_maxes = [scatter.detach().float().max() for scatter in scatter_maps]
        stats = {"pseudo_source_type": pseudo_source_type}
        stats.update(source_stats)
        stats.update({
            "object_mask_sum": _debug_value(object_mask.detach().float().sum()),
            "pseudo_target_mean": _debug_value(torch.stack(pseudo_means).mean()),
            "pseudo_target_max": _debug_value(torch.stack(pseudo_maxes).max()),
            "scatter_map_mean": _debug_value(torch.stack(scatter_means).mean()),
            "scatter_map_max": _debug_value(torch.stack(scatter_maxes).max()),
            "loss_fsqm_scatter": _debug_value(scatter_loss),
            "loss_fsqm_obj": _debug_value(obj_loss),
            "loss_fsqm_bg": _debug_value(bg_loss),
            "loss_fsqm_aux": _debug_value(total_aux),
        })
        _debug_print(stats)

    return {
        "loss_fsqm_scatter": scatter_loss.detach(),
        "loss_fsqm_obj": obj_loss.detach(),
        "loss_fsqm_bg": bg_loss.detach(),
        "loss_fsqm_aux": total_aux,
    }
