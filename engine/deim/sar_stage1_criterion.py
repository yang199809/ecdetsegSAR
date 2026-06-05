"""
Criterion for Stage 1 SAR instance segmentation.
"""

import torch
import torch.nn.functional as F

from ..core import register
from .deim_criterion import DEIMCriterion


def _zero_loss(outputs):
    return outputs["pred_logits"].sum() * 0.0


def _point_sample(input_tensor, point_coords, mode: str):
    grid = point_coords.mul(2.0).sub(1.0).unsqueeze(2)
    sampled = F.grid_sample(input_tensor, grid, mode=mode, align_corners=False)
    return sampled.squeeze(3).squeeze(1)


def _dice_loss(inputs, targets, eps: float = 1.0):
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    return (1 - (numerator + eps) / (denominator + eps)).mean()


@register()
class SARStage1Criterion(DEIMCriterion):
    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        mask_point_sample_ratio: int = 16,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        mask_min_sampled_points: int = 256,
        mask_max_sampled_points: int = 1024,
        **kwargs,
    ):
        super().__init__(matcher=matcher, weight_dict=weight_dict, losses=losses, **kwargs)
        self.mask_point_sample_ratio = mask_point_sample_ratio
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_min_sampled_points = mask_min_sampled_points
        self.mask_max_sampled_points = mask_max_sampled_points

    def _validate_masks(self, targets):
        for batch_idx, target in enumerate(targets):
            if "masks" not in target:
                if len(target["labels"]) == 0:
                    continue
                raise ValueError(
                    f"Target {batch_idx} has labels but no masks. "
                    "Check that the COCO annotation contains segmentation and return_masks=True."
                )
            if len(target["labels"]) != len(target["masks"]):
                raise ValueError(
                    f"Target {batch_idx} label/mask count mismatch: "
                    f"{len(target['labels'])} labels vs {len(target['masks'])} masks. "
                    "A transform likely failed to synchronize masks."
                )

    def _num_sampled_points(self, pred_masks):
        _, _, height, width = pred_masks.shape
        points = max(1, (height * width) // max(1, self.mask_point_sample_ratio))
        points = max(points, self.mask_min_sampled_points)
        points = min(points, self.mask_max_sampled_points)
        return points

    def _sample_points(self, src_masks, num_points):
        num_masks = src_masks.shape[0]
        oversample_points = max(num_points, int(num_points * self.oversample_ratio))
        point_coords = torch.rand(num_masks, oversample_points, 2, device=src_masks.device)

        with torch.no_grad():
            point_logits = _point_sample(src_masks.unsqueeze(1), point_coords, mode="bilinear")
            uncertainty = -point_logits.abs()
            num_uncertain = int(num_points * self.importance_sample_ratio)
            num_uncertain = min(num_uncertain, num_points)

            if num_uncertain > 0:
                topk_idx = torch.topk(uncertainty, k=num_uncertain, dim=1).indices
                topk_coords = point_coords.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, 2))
            else:
                topk_coords = point_coords[:, :0]

            num_random = num_points - num_uncertain
            random_coords = torch.rand(num_masks, num_random, 2, device=src_masks.device)
            point_coords = torch.cat([topk_coords, random_coords], dim=1)

        return point_coords

    def loss_masks(self, outputs, targets, indices, num_boxes):
        if "pred_masks" not in outputs or outputs["pred_masks"] is None:
            return {}

        self._validate_masks(targets)
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            zero = _zero_loss(outputs)
            return {"loss_mask_bce": zero, "loss_mask_dice": zero}

        target_masks = []
        for target, (_, tgt_idx) in zip(targets, indices):
            if len(tgt_idx) > 0:
                target_masks.append(target["masks"][tgt_idx])

        if not target_masks:
            zero = _zero_loss(outputs)
            return {"loss_mask_bce": zero, "loss_mask_dice": zero}

        src_masks = outputs["pred_masks"][idx]
        target_masks = torch.cat(target_masks, dim=0).to(device=src_masks.device, dtype=src_masks.dtype)
        if target_masks.numel() == 0:
            zero = _zero_loss(outputs)
            return {"loss_mask_bce": zero, "loss_mask_dice": zero}

        num_points = self._num_sampled_points(outputs["pred_masks"])
        point_coords = self._sample_points(src_masks.detach(), num_points)

        src_points = _point_sample(src_masks.unsqueeze(1), point_coords, mode="bilinear")
        tgt_points = _point_sample(target_masks.unsqueeze(1), point_coords, mode="nearest")

        return {
            "loss_mask_bce": F.binary_cross_entropy_with_logits(src_points, tgt_points),
            "loss_mask_dice": _dice_loss(src_points, tgt_points),
        }

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        if loss == "masks":
            return self.loss_masks(outputs, targets, indices, num_boxes)
        return super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, epoch=0, **kwargs):
        losses = super().forward(outputs, targets, epoch=epoch, **kwargs)
        if "loss_ddf" in self.weight_dict and not any(
            key == "loss_ddf" or key.startswith("loss_ddf_") for key in losses
        ):
            losses["loss_ddf"] = _zero_loss(outputs)
        return losses
