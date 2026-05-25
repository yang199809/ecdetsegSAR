"""
Post-processing for DEIMv2 SAR Stage-1 instance segmentation.
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
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category'
    ]

    def __init__(self,
                 num_classes=1,
                 use_focal_loss=True,
                 num_top_queries=300,
                 remap_mscoco_category=False,
                 category_id_offset=0,
                 return_geometry=False):
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.category_id_offset = category_id_offset
        self.return_geometry = return_geometry
        self.deploy_mode = False

    def _select_topk(self, logits, bbox_pred):
        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            topk = min(self.num_top_queries, scores.flatten(1).shape[1])
            scores, index = torch.topk(scores.flatten(1), topk, dim=-1)
            labels = mod(index, self.num_classes)
            query_index = index // self.num_classes
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, query_index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=query_index)
            else:
                query_index = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0).tile(scores.shape[0], 1)

        boxes = bbox_pred.gather(dim=1, index=query_index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
        return labels, boxes, scores, query_index

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        labels, boxes, scores, query_index = self._select_topk(logits, bbox_pred)

        if self.deploy_mode:
            return labels, boxes, scores

        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)
        elif self.category_id_offset:
            labels = labels + int(self.category_id_offset)

        pred_masks = outputs.get('pred_masks')
        selected_masks = None
        if pred_masks is not None:
            mask_index = query_index.unsqueeze(-1).unsqueeze(-1).repeat(
                1, 1, pred_masks.shape[-2], pred_masks.shape[-1])
            selected_masks = pred_masks.gather(dim=1, index=mask_index).sigmoid()

        geo_outputs = outputs.get('geo_outputs', {})
        results = []
        for batch_idx, (lab, box, sco) in enumerate(zip(labels, boxes, scores)):
            result = dict(labels=lab, boxes=box, scores=sco)
            if selected_masks is not None:
                width, height = orig_target_sizes[batch_idx].to(dtype=torch.int64).tolist()
                masks = F.interpolate(
                    selected_masks[batch_idx][:, None],
                    size=(int(height), int(width)),
                    mode='bilinear',
                    align_corners=False,
                )
                result['masks'] = masks

            if self.return_geometry and geo_outputs:
                result['geometry'] = {}
                for key, value in geo_outputs.items():
                    if torch.is_tensor(value) and value.ndim >= 3:
                        gather_index = query_index[batch_idx].unsqueeze(-1).repeat(1, value.shape[-1])
                        result['geometry'][key] = value[batch_idx].gather(dim=0, index=gather_index)

            results.append(result)

        return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
