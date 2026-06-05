"""
Stage 1 SAR instance segmentation extension for official DEIMv2.
"""

import torch
import torch.nn as nn

from ..core import register
from .deim import DEIM
from .deim_decoder import DEIMTransformer
from .denoising import get_contrastive_denoising_training_group
from .sar_segmentation_head import SARSegmentationHead


@register()
class DEIMv2_SAR_INS_STAGE1(DEIM):
    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__(backbone=backbone, encoder=encoder, decoder=decoder)


@register()
class SARStage1DEIMTransformer(DEIMTransformer):
    def __init__(
        self,
        mask_hidden_dim: int = 128,
        mask_output_stride: int = 4,
        mask_num_blocks: int = 4,
        use_multiscale_mask_fusion: bool = True,
        use_mask_aux_loss: bool = True,
        use_sparse_mask_train: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mask_head = SARSegmentationHead(
            hidden_dim=self.hidden_dim,
            mask_hidden_dim=mask_hidden_dim,
            mask_output_stride=mask_output_stride,
            mask_num_blocks=mask_num_blocks,
            use_multiscale_fusion=use_multiscale_mask_fusion,
        )
        self.use_mask_aux_loss = use_mask_aux_loss
        self.use_sparse_mask_train = use_sparse_mask_train
        self.mask_output_stride = mask_output_stride

    @staticmethod
    def _memory_to_feature_maps(memory: torch.Tensor, spatial_shapes):
        bsz, _, channels = memory.shape
        split_sizes = [h * w for h, w in spatial_shapes]
        levels = memory.split(split_sizes, dim=1)
        return [
            level.transpose(1, 2).reshape(bsz, channels, h, w).contiguous()
            for level, (h, w) in zip(levels, spatial_shapes)
        ]

    def _predict_masks(self, mask_features, query_layers: torch.Tensor):
        masks = [
            self.mask_head(mask_features[0], layer_queries, multiscale_features=mask_features, block_index=i)
            for i, layer_queries in enumerate(query_layers)
        ]
        return torch.stack(masks, dim=0)

    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)
        mask_features = self._memory_to_feature_maps(memory, spatial_shapes)

        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=1.0,
                )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits, out_queries = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta,
            return_intermediate_queries=True,
        )

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1)

            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta["dn_num_split"], dim=2)
            _, out_queries = torch.split(out_queries, dn_meta["dn_num_split"], dim=2)

        pred_masks_all = self._predict_masks(mask_features, out_queries)

        if self.training:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],
                "pred_masks": pred_masks_all[-1],
                "up": self.up,
                "reg_scale": self.reg_scale,
            }
        else:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_masks": pred_masks_all[-1],
            }

        if out["pred_logits"].shape[:2] != out["pred_masks"].shape[:2]:
            raise RuntimeError(
                "Stage 1 mask query count mismatch: "
                f"pred_logits={tuple(out['pred_logits'].shape)}, "
                f"pred_masks={tuple(out['pred_masks'].shape)}"
            )

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],
                out_logits[-1],
            )
            if self.use_mask_aux_loss:
                for aux, aux_masks in zip(out["aux_outputs"], pred_masks_all[:-1]):
                    aux["pred_masks"] = aux_masks

            out["enc_aux_outputs"] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}

            if dn_meta is not None:
                out["dn_outputs"] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_corners[-1],
                    dn_out_logits[-1],
                )
                out["dn_pre_outputs"] = {"pred_logits": dn_pre_logits, "pred_boxes": dn_pre_bboxes}
                out["dn_meta"] = dn_meta

        return out
