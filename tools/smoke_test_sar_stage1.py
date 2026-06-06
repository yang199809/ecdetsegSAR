import argparse
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import engine  # noqa: F401
from engine.core import YAMLConfig


def _device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def _make_fast(model, num_queries=50, num_denoising=10):
    if hasattr(model.encoder, "eval_spatial_size"):
        model.encoder.eval_spatial_size = None
    decoder = model.decoder
    decoder.eval_spatial_size = None
    decoder.num_queries = min(num_queries, decoder.num_queries)
    if hasattr(decoder, "num_denoising"):
        decoder.num_denoising = min(num_denoising, decoder.num_denoising)
    return model


def _dummy_targets(batch_size, image_size, device):
    targets = []
    for i in range(batch_size):
        masks = torch.zeros(1, image_size, image_size, device=device)
        x0, y0, x1, y1 = image_size // 4, image_size // 4, image_size // 2, image_size // 2
        masks[0, y0:y1, x0:x1] = 1
        box = torch.tensor(
            [[(x0 + x1) / 2 / image_size, (y0 + y1) / 2 / image_size,
              (x1 - x0) / image_size, (y1 - y0) / image_size]],
            dtype=torch.float32,
            device=device,
        )
        targets.append(
            {
                "boxes": box,
                "labels": torch.zeros(1, dtype=torch.long, device=device),
                "masks": masks,
                "orig_size": torch.tensor([image_size, image_size], device=device),
                "image_id": torch.tensor([i], device=device),
            }
        )
    return targets


def _assert_finite_losses(losses):
    required = ["loss_mal", "loss_bbox", "loss_giou", "loss_fgl", "loss_mask_bce", "loss_mask_dice"]
    for key in required:
        if key not in losses:
            raise AssertionError(f"Missing required loss: {key}")
        if not torch.isfinite(losses[key]):
            raise AssertionError(f"Loss is not finite: {key}={losses[key]}")
    if not any(k == "loss_ddf" or k.startswith("loss_ddf_") for k in losses):
        raise AssertionError("Missing loss_ddf or auxiliary loss_ddf_* from DEIMv2 local loss.")
    for key, value in losses.items():
        if torch.is_tensor(value) and not torch.isfinite(value):
            raise AssertionError(f"Non-finite loss: {key}={value}")


def _assert_forward_layers(model, device):
    decoder = model.decoder
    hidden_dim = decoder.hidden_dim
    num_layers = len(decoder.decoder.layers)
    batch_size, num_queries = 2, 7
    spatial = torch.randn(batch_size, hidden_dim, 10, 10, device=device)
    query_layers = torch.randn(num_layers, batch_size, num_queries, hidden_dim, device=device)

    with torch.no_grad():
        pred_masks_all = decoder.mask_head.forward_layers(spatial, query_layers)

    expected_hw = 20 if decoder.mask_head.mask_output_stride == 4 else 10
    expected = (num_layers, batch_size, num_queries, expected_hw, expected_hw)
    if tuple(pred_masks_all.shape) != expected:
        raise AssertionError(f"forward_layers shape mismatch: got {tuple(pred_masks_all.shape)}, expected {expected}")
    print(f"[mask-head] forward_layers pred_masks_all={tuple(pred_masks_all.shape)}")


def _assert_matcher_warmup(criterion):
    matcher = criterion.matcher
    if not hasattr(matcher, "_mask_cost_weight_scale"):
        raise AssertionError("Matcher is missing _mask_cost_weight_scale().")

    start = matcher.mask_cost_start_epoch
    warmup = matcher.mask_cost_warmup_epochs
    if warmup <= 0:
        if matcher._mask_cost_weight_scale(0) != 1.0:
            raise AssertionError("Matcher warmup disabled but scale is not 1.0.")
        print("[matcher] mask cost warmup disabled: scale=1.0")
        return

    if matcher._mask_cost_weight_scale(max(0, start - 1)) != 0.0:
        raise AssertionError("Matcher mask cost scale should be 0 before start epoch.")
    mid_scale = matcher._mask_cost_weight_scale(start)
    if not (0.0 < mid_scale <= 1.0):
        raise AssertionError(f"Matcher warmup scale should be in (0, 1], got {mid_scale}.")
    final_scale = matcher._mask_cost_weight_scale(start + warmup)
    if final_scale != 1.0:
        raise AssertionError(f"Matcher warmup scale should be 1 after warmup, got {final_scale}.")
    print(
        "[matcher] mask cost warmup scales: "
        f"pre={matcher._mask_cost_weight_scale(max(0, start - 1)):.3f}, "
        f"start={mid_scale:.3f}, post={final_scale:.3f}"
    )


def _inspect_category_id(ann_file):
    if not ann_file or not os.path.exists(ann_file):
        return None
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    annotations = data.get("annotations", [])
    if not annotations:
        return None
    category_ids = sorted({ann["category_id"] for ann in annotations if "category_id" in ann})
    has_segmentation = all("segmentation" in ann for ann in annotations)
    return {"category_ids": category_ids, "has_segmentation": has_segmentation}


def _maybe_real_dataloader_check(cfg, model, criterion, postprocessor, device):
    train_ann = cfg.yaml_cfg["train_dataloader"]["dataset"].get("ann_file")
    val_ann = cfg.yaml_cfg["val_dataloader"]["dataset"].get("ann_file")
    train_info = _inspect_category_id(train_ann)
    val_info = _inspect_category_id(val_ann)
    print(f"[real-data] train annotation: {train_ann}")
    print(f"[real-data] val annotation: {val_ann}")
    print(f"[real-data] train annotation info: {train_info}")
    print(f"[real-data] val annotation info: {val_info}")

    if not (train_ann and val_ann and os.path.exists(train_ann) and os.path.exists(val_ann)):
        print("[real-data] skipped: configured HRSID annotation files do not exist.")
        return

    model.train()
    samples, targets = next(iter(cfg.train_dataloader))
    samples = samples.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    with torch.no_grad():
        outputs = model(samples, targets=targets)
        losses = criterion(outputs, targets, epoch=0)
    _assert_finite_losses(losses)
    print(f"[real-data] train forward ok: pred_masks={tuple(outputs['pred_masks'].shape)}")

    model.eval()
    samples, targets = next(iter(cfg.val_dataloader))
    samples = samples.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    with torch.no_grad():
        outputs = model(samples)
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_sizes)
    if "masks" not in results[0]:
        raise AssertionError("Postprocessor did not return masks for real-data eval batch.")
    print(f"[real-data] eval forward ok: masks={tuple(results[0]['masks'].shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/our/deimv2_dinov3_s_HRSID_ins_stage1.yml")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--skip-real-data", action="store_true")
    args = parser.parse_args()

    device = _device(args.device)
    cfg = YAMLConfig(args.config)
    model = _make_fast(cfg.model).to(device)
    criterion = cfg.criterion.to(device)
    postprocessor = cfg.postprocessor.to(device)
    postprocessor.num_top_queries = min(postprocessor.num_top_queries, model.decoder.num_queries)
    if hasattr(postprocessor, "num_top_masks"):
        postprocessor.num_top_masks = min(postprocessor.num_top_masks, model.decoder.num_queries)

    samples = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    targets = _dummy_targets(args.batch_size, args.image_size, device)

    _assert_forward_layers(model, device)
    _assert_matcher_warmup(criterion)

    model.eval()
    with torch.no_grad():
        outputs = model(samples)
    for key in ["pred_logits", "pred_boxes", "pred_masks"]:
        if key not in outputs:
            raise AssertionError(f"Missing output key: {key}")
    if outputs["pred_logits"].shape[:2] != outputs["pred_boxes"].shape[:2]:
        raise AssertionError("pred_logits and pred_boxes query shapes differ.")
    if outputs["pred_logits"].shape[:2] != outputs["pred_masks"].shape[:2]:
        raise AssertionError("pred_logits and pred_masks query shapes differ.")
    print(f"[dummy-eval] pred_logits={tuple(outputs['pred_logits'].shape)}")
    print(f"[dummy-eval] pred_masks={tuple(outputs['pred_masks'].shape)}")

    orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    with torch.no_grad():
        results = postprocessor(outputs, orig_sizes)
    if "masks" not in results[0]:
        raise AssertionError("Postprocessor did not return masks.")
    if tuple(results[0]["masks"].shape[-2:]) != (args.image_size, args.image_size):
        raise AssertionError("Postprocessed mask size does not match original image size.")
    if not (len(results[0]["masks"]) == len(results[0]["mask_labels"]) == len(results[0]["mask_scores"])):
        raise AssertionError("masks/mask_labels/mask_scores length mismatch.")
    print(f"[postprocessor] masks={tuple(results[0]['masks'].shape)}")

    model.train()
    with torch.no_grad():
        train_outputs = model(samples, targets=targets)
        losses = criterion(train_outputs, targets, epoch=0)
    if "aux_outputs" in train_outputs:
        aux_with_masks = [aux for aux in train_outputs["aux_outputs"] if "pred_masks" in aux]
        if len(aux_with_masks) != len(train_outputs["aux_outputs"]):
            raise AssertionError("Auxiliary detection outputs are not fully aligned with mask outputs.")
        expected_layers = len(train_outputs["aux_outputs"]) + 1
        if expected_layers != len(model.decoder.decoder.layers):
            raise AssertionError(
                f"Unexpected decoder layer count: aux+final={expected_layers}, "
                f"decoder_layers={len(model.decoder.decoder.layers)}"
            )
    print("[dummy-train] loss keys:", ", ".join(sorted(losses.keys())))
    _assert_finite_losses(losses)
    print("[dummy-train] finite losses:", ", ".join(sorted(losses.keys())[:12]), "...")

    det_cfg = YAMLConfig("configs/our/deimv2_dinov3_s_HRSID.yml")
    det_model = _make_fast(det_cfg.model, num_queries=20, num_denoising=0).to(device).eval()
    with torch.no_grad():
        det_outputs = det_model(samples[:1])
    if "pred_masks" in det_outputs:
        raise AssertionError("Detection-only config unexpectedly returned pred_masks.")
    print("[detection-only] ok: no pred_masks")

    if not args.skip_real_data:
        _maybe_real_dataloader_check(cfg, model, criterion, postprocessor, device)

    print("[smoke] SAR Stage 1 smoke test passed.")


if __name__ == "__main__":
    main()
