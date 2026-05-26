"""
Stage-1 SAR instance segmentation criterion.

Detection losses and Hungarian matching stay identical to DEIMv2. Mask and weak
geometry losses are added only for final matched positive object queries.
"""

import torch
import torch.nn.functional as F

from ..core import register
from ..data.dataset.weak_geometry import masks_to_weak_geometry
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .deim_criterion import DEIMCriterion


@register()
class SARStage1Criterion(DEIMCriterion):
    CUSTOM_LOSSES = {'masks', 'weak_geometry'}

    def _zero_loss(self, outputs):
        return outputs['pred_logits'].sum() * 0.0

    def _num_matched(self, indices):
        return sum(src.numel() for src, _ in indices)

    def _ensure_geometry_targets(self, targets):
        for target in targets:
            if 'gt_center' not in target and 'masks' in target:
                target.update(masks_to_weak_geometry(target['masks']))

    def loss_masks(self, outputs, targets, indices, num_boxes):
        if 'pred_masks' not in outputs or self._num_matched(indices) == 0:
            zero = self._zero_loss(outputs)
            return {'loss_mask_bce': zero, 'loss_mask_dice': zero}

        if not all('masks' in target for target in targets):
            zero = self._zero_loss(outputs)
            return {'loss_mask_bce': zero, 'loss_mask_dice': zero}

        for batch_idx, target in enumerate(targets):
            num_labels = int(target['labels'].shape[0])
            num_masks = int(target['masks'].shape[0])
            if num_labels != num_masks:
                raise ValueError(
                    f"Mask/label count mismatch at batch index {batch_idx}: "
                    f"{num_labels} labels but {num_masks} masks. "
                    "Disable mask-unsafe augmentations or update them to transform masks."
                )

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs['pred_masks'][src_idx]
        target_masks = torch.cat([
            target['masks'][j] for target, (_, j) in zip(targets, indices)
        ], dim=0).to(device=src_masks.device, dtype=src_masks.dtype)

        target_masks = F.interpolate(
            target_masks[:, None],
            size=src_masks.shape[-2:],
            mode='nearest',
        )[:, 0]

        loss_bce = F.binary_cross_entropy_with_logits(
            src_masks, target_masks, reduction='none')
        loss_bce = loss_bce.flatten(1).mean(1).sum() / num_boxes

        src_probs = src_masks.sigmoid().flatten(1)
        target_flat = target_masks.flatten(1)
        numerator = 2 * (src_probs * target_flat).sum(1)
        denominator = src_probs.sum(1) + target_flat.sum(1)
        loss_dice = (1 - (numerator + 1) / (denominator + 1)).sum() / num_boxes

        return {'loss_mask_bce': loss_bce, 'loss_mask_dice': loss_dice}

    def loss_weak_geometry(self, outputs, targets, indices, num_boxes):
        self._ensure_geometry_targets(targets)
        geo_outputs = outputs.get('geo_outputs', {})
        required_preds = ('pred_center', 'pred_scale', 'pred_dir', 'pred_anisotropy')
        required_targets = ('gt_center', 'gt_scale', 'gt_axis', 'gt_anisotropy')

        if (self._num_matched(indices) == 0 or
                not all(key in geo_outputs for key in required_preds) or
                not all(all(key in target for key in required_targets) for target in targets)):
            zero = self._zero_loss(outputs)
            return {
                'loss_geo_center': zero,
                'loss_geo_scale': zero,
                'loss_geo_dir': zero,
                'loss_geo_ani': zero,
            }

        src_idx = self._get_src_permutation_idx(indices)
        pred_center = geo_outputs['pred_center'][src_idx]
        pred_scale = geo_outputs['pred_scale'][src_idx]
        pred_dir = geo_outputs['pred_dir'][src_idx]
        pred_anisotropy = geo_outputs['pred_anisotropy'][src_idx]

        def cat_target(key):
            return torch.cat([
                target[key][j] for target, (_, j) in zip(targets, indices)
            ], dim=0).to(device=pred_center.device, dtype=pred_center.dtype)

        gt_center = cat_target('gt_center')
        gt_scale = cat_target('gt_scale')
        gt_axis = F.normalize(cat_target('gt_axis'), p=2, dim=-1, eps=1e-6)
        gt_anisotropy = cat_target('gt_anisotropy')
        valid = cat_target('gt_geo_valid') if all('gt_geo_valid' in t for t in targets) else torch.ones_like(gt_anisotropy)
        geo_weight = cat_target('gt_geo_weight') if all('gt_geo_weight' in t for t in targets) else valid

        valid = valid.squeeze(-1)
        geo_weight = geo_weight.squeeze(-1)
        loss_center = (F.l1_loss(pred_center, gt_center, reduction='none').sum(-1) * valid).sum() / num_boxes
        loss_scale = (F.l1_loss(pred_scale, gt_scale, reduction='none').sum(-1) * geo_weight).sum() / num_boxes

        dir_dot = (pred_dir * gt_axis).sum(-1).clamp(-1.0, 1.0).abs()
        loss_dir = ((1.0 - dir_dot) * geo_weight).sum() / num_boxes
        loss_ani = (F.l1_loss(pred_anisotropy, gt_anisotropy, reduction='none').squeeze(-1) * geo_weight).sum() / num_boxes

        return {
            'loss_geo_center': loss_center,
            'loss_geo_scale': loss_scale,
            'loss_geo_dir': loss_dir,
            'loss_geo_ani': loss_ani,
        }

    def forward(self, outputs, targets, epoch=0, **kwargs):
        requested_losses = list(self.losses)
        det_losses = [loss for loss in requested_losses if loss not in self.CUSTOM_LOSSES]

        try:
            self.losses = det_losses
            losses = super().forward(outputs, targets, epoch=epoch, **kwargs)
        finally:
            self.losses = requested_losses

        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
        indices = self.matcher(
            outputs_without_aux, targets, epoch=epoch, step=kwargs.get('global_step', None))['indices']

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs['pred_logits'].device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        custom_losses = {}
        if 'masks' in requested_losses:
            custom_losses.update(self.loss_masks(outputs, targets, indices, num_boxes))
        if 'weak_geometry' in requested_losses:
            custom_losses.update(self.loss_weak_geometry(outputs, targets, indices, num_boxes))

        custom_losses = {
            key: value * self.weight_dict.get(key, 1.0)
            for key, value in custom_losses.items()
        }
        losses.update(custom_losses)
        return {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
