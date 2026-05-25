"""
Weak mask-derived geometry utilities for SAR instance segmentation.

These helpers derive orientation-like supervision from polygons or masks without
requiring human-provided angle labels.
"""

import torch
import faster_coco_eval.core.mask as mask_util


def polygons_to_mask(segmentations, height, width, device=None):
    """Convert COCO polygon/RLE segmentations to a boolean mask tensor."""
    if isinstance(segmentations, torch.Tensor):
        masks = segmentations.to(dtype=torch.bool, device=device)
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        return masks

    masks = []
    for segmentation in segmentations:
        if segmentation is None or (not isinstance(segmentation, dict) and len(segmentation) == 0):
            masks.append(torch.zeros((height, width), dtype=torch.uint8))
            continue
        if isinstance(segmentation, dict):
            rles = [segmentation]
        else:
            rles = mask_util.frPyObjects(segmentation, height, width)
        mask = mask_util.decode(rles)
        if mask.ndim < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8).any(dim=2)
        masks.append(mask)

    if not masks:
        return torch.zeros((0, height, width), dtype=torch.bool, device=device)
    return torch.stack(masks, dim=0).to(dtype=torch.bool, device=device)


def principal_axis_from_mask(mask, min_area=4, eps=1e-6):
    """Return center, axis, scale, anisotropy, and validity for one mask."""
    mask = torch.as_tensor(mask).bool()
    device = mask.device
    h, w = mask.shape[-2:]
    coords_yx = torch.nonzero(mask, as_tuple=False)

    if coords_yx.shape[0] < min_area:
        zero2 = torch.zeros(2, dtype=torch.float32, device=device)
        zero1 = torch.zeros(1, dtype=torch.float32, device=device)
        return zero2, zero2, zero2, zero1, zero1, zero1

    coords = coords_yx.to(dtype=torch.float32)
    coords_xy = torch.stack(
        ((coords[:, 1] + 0.5) / max(float(w), 1.0),
         (coords[:, 0] + 0.5) / max(float(h), 1.0)),
        dim=1,
    )

    center = coords_xy.mean(dim=0)
    centered = coords_xy - center
    denom = max(coords_xy.shape[0] - 1, 1)
    cov = centered.t().matmul(centered) / float(denom)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp_min(eps)

    axis = eigvecs[:, -1]
    axis = axis / axis.norm(p=2).clamp_min(eps)
    axis = torch.where(axis[:1] < 0, -axis, axis)
    ortho = torch.stack((-axis[1], axis[0]))

    major_proj = centered.matmul(axis)
    minor_proj = centered.matmul(ortho)
    scale = torch.stack((
        major_proj.max() - major_proj.min(),
        minor_proj.max() - minor_proj.min(),
    )).clamp(min=0.0, max=1.0)

    major = eigvals[-1].sqrt()
    minor = eigvals[0].sqrt()
    anisotropy = ((major - minor) / (major + minor + eps)).clamp(0.0, 1.0).view(1)
    valid = torch.ones(1, dtype=torch.float32, device=device)
    geo_weight = valid * anisotropy
    return center, scale, axis, anisotropy, valid, geo_weight


def anisotropy_from_mask(mask, min_area=4, eps=1e-6):
    """Return only the elongation score for one mask."""
    return principal_axis_from_mask(mask, min_area=min_area, eps=eps)[3]


def masks_to_weak_geometry(masks, min_area=4, eps=1e-6):
    """Derive weak geometry targets from an [N, H, W] mask tensor."""
    masks = torch.as_tensor(masks)
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)

    device = masks.device
    if masks.numel() == 0 or masks.shape[0] == 0:
        return {
            'gt_center': torch.zeros((0, 2), dtype=torch.float32, device=device),
            'gt_scale': torch.zeros((0, 2), dtype=torch.float32, device=device),
            'gt_axis': torch.zeros((0, 2), dtype=torch.float32, device=device),
            'gt_anisotropy': torch.zeros((0, 1), dtype=torch.float32, device=device),
            'gt_geo_valid': torch.zeros((0, 1), dtype=torch.float32, device=device),
            'gt_geo_weight': torch.zeros((0, 1), dtype=torch.float32, device=device),
        }

    centers, scales, axes, anisotropies, valids, weights = [], [], [], [], [], []
    for mask in masks:
        center, scale, axis, anisotropy, valid, weight = principal_axis_from_mask(
            mask, min_area=min_area, eps=eps)
        centers.append(center)
        scales.append(scale)
        axes.append(axis)
        anisotropies.append(anisotropy)
        valids.append(valid)
        weights.append(weight)

    return {
        'gt_center': torch.stack(centers, dim=0),
        'gt_scale': torch.stack(scales, dim=0),
        'gt_axis': torch.stack(axes, dim=0),
        'gt_anisotropy': torch.stack(anisotropies, dim=0),
        'gt_geo_valid': torch.stack(valids, dim=0),
        'gt_geo_weight': torch.stack(weights, dim=0),
    }
