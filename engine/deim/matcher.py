"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from typing import Dict

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou

from ..core import register
import numpy as np


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0,
                change_matcher=False, iou_order_alpha=1.0, matcher_change_epoch=10000,
                mask_point_sample_ratio=16):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']
        self.cost_mask_bce = weight_dict.get('cost_mask_bce', 0.0)
        self.cost_mask_dice = weight_dict.get('cost_mask_dice', 0.0)
        self.mask_point_sample_ratio = mask_point_sample_ratio

        self.change_matcher = change_matcher
        self.iou_order_alpha = iou_order_alpha
        self.matcher_change_epoch = matcher_change_epoch
        if self.change_matcher:
            print(f"Using the new matching cost with iou_order_alpha = {iou_order_alpha} at epoch {matcher_change_epoch}")

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0 \
            or self.cost_mask_bce != 0 or self.cost_mask_dice != 0, "all costs cant be 0"

    def _mask_cost(self, outputs, targets):
        if (self.cost_mask_bce == 0 and self.cost_mask_dice == 0) or "pred_masks" not in outputs:
            return None

        target_masks = []
        for target in targets:
            if "masks" not in target:
                if len(target["labels"]) == 0:
                    continue
                raise ValueError("Matcher mask cost requested, but a target with labels has no masks.")
            if len(target["labels"]) != len(target["masks"]):
                raise ValueError(
                    f"Matcher target label/mask mismatch: {len(target['labels'])} labels vs "
                    f"{len(target['masks'])} masks."
                )
            if len(target["masks"]) > 0:
                target_masks.append(target["masks"])

        if not target_masks:
            return None

        pred_masks = outputs["pred_masks"].flatten(0, 1)
        tgt_masks = torch.cat(target_masks, dim=0).to(device=pred_masks.device, dtype=pred_masks.dtype)

        height, width = outputs["pred_masks"].shape[-2:]
        num_points = max(1, (height * width) // max(1, self.mask_point_sample_ratio))
        point_coords = torch.rand(1, num_points, 2, device=pred_masks.device)

        pred_points = F.grid_sample(
            pred_masks.unsqueeze(1),
            point_coords.expand(pred_masks.shape[0], -1, -1).mul(2).sub(1).unsqueeze(2),
            mode="bilinear",
            align_corners=False,
        ).squeeze(3).squeeze(1)
        tgt_points = F.grid_sample(
            tgt_masks.unsqueeze(1),
            point_coords.expand(tgt_masks.shape[0], -1, -1).mul(2).sub(1).unsqueeze(2),
            mode="nearest",
            align_corners=False,
        ).squeeze(3).squeeze(1)

        costs = []
        if self.cost_mask_bce != 0:
            bce = F.binary_cross_entropy_with_logits(
                pred_points[:, None, :].expand(-1, tgt_points.shape[0], -1),
                tgt_points[None, :, :].expand(pred_points.shape[0], -1, -1),
                reduction="none",
            ).mean(-1)
            costs.append(self.cost_mask_bce * bce)

        if self.cost_mask_dice != 0:
            pred_prob = pred_points.sigmoid()
            numerator = 2 * torch.einsum("np,tp->nt", pred_prob, tgt_points)
            denominator = pred_prob.sum(-1)[:, None] + tgt_points.sum(-1)[None, :]
            dice = 1 - (numerator + 1.0) / (denominator + 1.0)
            costs.append(self.cost_mask_dice * dice)

        return sum(costs) if costs else None

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False, epoch=0):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        sizes = [len(v["boxes"]) for v in targets]
        if tgt_bbox.numel() == 0:
            empty = [
                (torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64))
                for _ in range(bs)
            ]
            return {'indices': empty}

        if self.change_matcher and epoch >= self.matcher_change_epoch:
            # Compute the class_score
            class_score = out_prob[:, tgt_ids]  # shape = [batch_size * num_queries, gt num within a batch]

            # # Compute iou
            bbox_iou, _ = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = (-1) * (class_score * torch.pow(bbox_iou, self.iou_order_alpha))
        else:
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            if self.use_focal_loss:
                out_prob = out_prob[:, tgt_ids]
                neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class
            else:
                cost_class = -out_prob[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

            mask_cost = self._mask_cost(outputs, targets)
            if mask_cost is not None:
                C = C + mask_cost

        C = C.view(bs, num_queries, -1).cpu()

        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_pre]

        # Compute topk indices
        if return_topk:
            return {'indices_o2m': self.get_top_k_matches(C, sizes=sizes, k=return_topk, initial_indices=indices_pre)}

        return {'indices': indices} # , 'indices_o2m': C.min(-1)[1]}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        # C_original = C.clone()
        for i in range(k):
            indices_k = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] if i > 0 else initial_indices
            indices_list.append([
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices_k
            ])
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [(torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                        torch.cat([indices_list[i][j][1] for i in range(k)], dim=0)) for j in range(len(sizes))]
        # C.copy_(C_original)
        return indices_list
