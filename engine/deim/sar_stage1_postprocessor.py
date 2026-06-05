"""
Post-processing for SAR Stage 1 instance segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..core import register
from .postprocessor import mod


@register()
class SARInstancePostProcessor(nn.Module):
    __share__ = [
        "num_classes",
        "use_focal_loss",
        "num_top_queries",
        "remap_mscoco_category",
        "category_id_offset",
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        num_top_masks=100,
        mask_threshold=0.5,
        remap_mscoco_category=False,
        category_id_offset=0,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_top_masks = num_top_masks
        self.mask_threshold = mask_threshold
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.category_id_offset = int(category_id_offset)

    def _select(self, logits, boxes):
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, flat_index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = mod(flat_index, self.num_classes)
            query_index = flat_index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=query_index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            query_index = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0).repeat(scores.shape[0], 1)
            if scores.shape[1] > self.num_top_queries:
                scores, top_index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = labels.gather(dim=1, index=top_index)
                query_index = query_index.gather(dim=1, index=top_index)
                boxes = boxes.gather(dim=1, index=top_index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))

        return scores, labels, query_index, boxes

    def _map_labels(self, labels, device):
        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor(
                [mscoco_label2category[int(x.item())] for x in labels.flatten()],
                device=device,
            ).reshape(labels.shape)
        elif self.category_id_offset:
            labels = labels + self.category_id_offset
        return labels

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        scores, labels, query_index, boxes = self._select(logits, boxes)

        boxes = boxes * orig_target_sizes.repeat(1, 2).unsqueeze(1)
        labels = self._map_labels(labels, boxes.device)

        results = []
        pred_masks = outputs.get("pred_masks", None)
        for batch_idx, (lab, box, sco, orig_size) in enumerate(zip(labels, boxes, scores, orig_target_sizes)):
            result = {"labels": lab, "boxes": box, "scores": sco}

            if pred_masks is not None:
                keep = min(self.num_top_masks, query_index.shape[1])
                mask_query_index = query_index[batch_idx, :keep]
                masks = pred_masks[batch_idx].index_select(0, mask_query_index)
                out_w, out_h = int(orig_size[0].item()), int(orig_size[1].item())
                masks = F.interpolate(
                    masks.unsqueeze(1), size=(out_h, out_w), mode="bilinear", align_corners=False
                )
                masks = (masks.sigmoid() > self.mask_threshold).to(torch.uint8)
                result.update(
                    {
                        "masks": masks,
                        "mask_labels": lab[:keep],
                        "mask_scores": sco[:keep],
                    }
                )

            results.append(result)

        return results
