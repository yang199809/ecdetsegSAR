import argparse
import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from engine.edgecrafter import DINOv3Adapter, HybridEncoder  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weights-path", default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--skip-encoder", action="store_true")
    args = parser.parse_args()

    backbone = DINOv3Adapter(
        name="dinov3_vits16",
        source="timm",
        timm_model_name="vit_small_patch16_dinov3.lvd1689m",
        hf_model_id="timm/vit_small_patch16_dinov3.lvd1689m",
        pretrained=args.pretrained,
        weights_path=args.weights_path,
        embed_dim=384,
        num_heads=6,
        proj_dim=256,
        out_indices=[5, 8, 11],
        patch_size=16,
        out_strides=[8, 16, 32],
        frozen_stages=-1,
        use_checkpoint=False,
        drop_path_rate=0.0,
    ).to(args.device).eval()

    expected = [(2, 256, 80, 80), (2, 256, 40, 40), (2, 256, 20, 20)]
    x = torch.randn(2, 3, 640, 640, device=args.device)
    with torch.no_grad():
        feats = backbone(x)

    for i, feat in enumerate(feats):
        print(f"feat_s{[8, 16, 32][i]}: {tuple(feat.shape)}")
        assert tuple(feat.shape) == expected[i], f"Expected {expected[i]}, got {tuple(feat.shape)}"

    if not args.skip_encoder:
        encoder = HybridEncoder(
            in_channels=[256, 256, 256],
            feat_strides=[8, 16, 32],
            depth_mult=1,
            expansion=0.75,
            hidden_dim=256,
            dim_feedforward=1024,
            eval_spatial_size=[640, 640],
            fuse_op="sum",
        ).to(args.device).eval()
        with torch.no_grad():
            encoded = encoder(feats)
        for i, feat in enumerate(encoded):
            print(f"encoded_s{[8, 16, 32][i]}: {tuple(feat.shape)}")


if __name__ == "__main__":
    main()
