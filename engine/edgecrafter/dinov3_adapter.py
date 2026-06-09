"""
DINOv3 adapter for ECDet.

This module keeps the ECDet neck/decoder contract unchanged by converting
DINOv3/ViT-S patch tokens into three 256-channel feature maps at strides
8, 16, and 32. The official DINOv3 builder API may vary by weight source;
the builder below is intentionally defensive and falls back to the local
ViT-S/16 implementation for shape tests when an external DINOv3 package is
not available.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register
from .ecvit import PatchEmbed, VisionTransformer
from .hybrid_encoder import ConvNormLayer_fuse

__all__ = ["DINOv3Adapter"]


def _strip_known_prefixes(key: str) -> str:
    prefixes = ("module.", "model.", "backbone.", "teacher.", "student.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _is_tensor_state_dict(value: Any) -> bool:
    return isinstance(value, dict) and any(torch.is_tensor(v) for v in value.values())


def _unwrap_checkpoint(ckpt: Any) -> Dict[str, torch.Tensor]:
    if _is_tensor_state_dict(ckpt):
        return ckpt

    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    for key in ("model", "state_dict", "teacher", "student", "backbone"):
        value = ckpt.get(key)
        if _is_tensor_state_dict(value):
            return value
        if isinstance(value, dict):
            try:
                return _unwrap_checkpoint(value)
            except TypeError:
                pass

    nested = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
    if nested:
        return nested
    raise TypeError("Could not find a tensor state_dict in checkpoint.")


@register()
class DINOv3Adapter(nn.Module):
    def __init__(
        self,
        name="dinov3_vits16",
        weights_path=None,
        pretrained=True,
        embed_dim=384,
        proj_dim=256,
        out_indices=[5, 8, 11],
        patch_size=16,
        out_strides=[8, 16, 32],
        frozen_stages=-1,
        use_checkpoint=False,
        drop_path_rate=0.0,
        skip_load_backbone=False,
        **kwargs,
    ):
        super().__init__()
        if patch_size != 16:
            raise NotImplementedError("DINOv3Adapter currently expects a ViT-S/16-style patch_size=16.")
        if list(out_strides) != [8, 16, 32]:
            raise NotImplementedError("DINOv3Adapter currently outputs strides [8, 16, 32].")

        self.name = name
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.out_indices = list(out_indices)
        self.patch_size = patch_size
        self.out_strides = list(out_strides)
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint

        self.backbone = self._build_backbone(drop_path_rate=drop_path_rate, **kwargs)
        self.projector = nn.ModuleList([
            ConvNormLayer_fuse(embed_dim, proj_dim, kernel_size=1, stride=1, act="silu")
            for _ in self.out_strides
        ])

        self._apply_checkpoint_flag()
        self._freeze_stages()

        if pretrained and not skip_load_backbone:
            self._load_weights(weights_path)

    def _build_backbone(self, drop_path_rate=0.0, **kwargs):
        # Prefer a local or installed DINOv3 builder if one is available. The
        # exact official API can differ across releases, so we keep this small
        # and fall back to the repo-local ViT-S/16 implementation.
        try:
            import dinov3  # type: ignore  # noqa: F401
            from dinov3.hub.backbones import dinov3_vits16  # type: ignore

            if self.name == "dinov3_vits16":
                return dinov3_vits16(pretrained=False, weights=None, **kwargs)
        except Exception as exc:
            print(f"[DINOv3Adapter] External DINOv3 builder is unavailable: {exc}")

        print(
            "[DINOv3Adapter] Using the local ViT-S/16-compatible fallback. "
            "For official DINOv3 weights, install the matching DINOv3 code "
            "or adapt _build_backbone to that weight source."
        )
        return VisionTransformer(
            embed_dim=self.embed_dim,
            num_heads=6,
            return_layers=self.out_indices,
            patch_size=self.patch_size,
            embed_layer=PatchEmbed,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )

    def _apply_checkpoint_flag(self):
        if not self.use_checkpoint:
            return
        for module in self.backbone.modules():
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = True

    def _freeze_stages(self):
        if self.frozen_stages < 0:
            return
        if hasattr(self.backbone, "patch_embed"):
            self.backbone.patch_embed.eval()
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad = False
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is not None:
            for block in list(blocks)[: self.frozen_stages]:
                block.eval()
                for p in block.parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        return self

    def _load_weights(self, weights_path):
        if not weights_path:
            print("[DINOv3Adapter] pretrained=True but weights_path is empty; using initialized weights.")
            return

        path = Path(weights_path)
        if not path.exists():
            print(f"[DINOv3Adapter] weights_path not found: {path}; using initialized weights.")
            return

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")

        raw_state = _unwrap_checkpoint(ckpt)
        state = {}
        ignored_prefixes = ("head.", "fc.", "classifier.")
        for key, value in raw_state.items():
            clean_key = _strip_known_prefixes(key)
            if clean_key.startswith(ignored_prefixes):
                continue
            state[clean_key] = value

        incompatible = self.backbone.load_state_dict(state, strict=False)
        print(
            "[DINOv3Adapter] Loaded weights with strict=False: "
            f"missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
        if incompatible.missing_keys:
            print(f"[DINOv3Adapter] First missing keys: {incompatible.missing_keys[:8]}")
        if incompatible.unexpected_keys:
            print(f"[DINOv3Adapter] First unexpected keys: {incompatible.unexpected_keys[:8]}")

    def _forward_intermediate(self, x):
        if hasattr(self.backbone, "get_intermediate_layers"):
            try:
                return self.backbone.get_intermediate_layers(
                    x, n=self.out_indices, reshape=False, return_class_token=False
                )
            except TypeError:
                try:
                    return self.backbone.get_intermediate_layers(x, n=self.out_indices, reshape=False)
                except TypeError:
                    return self.backbone.get_intermediate_layers(x, n=len(self.out_indices))

        if hasattr(self.backbone, "forward_features"):
            out = self.backbone.forward_features(x)
            if isinstance(out, dict):
                for key in ("x_norm_patchtokens", "patch_tokens", "tokens"):
                    if key in out:
                        return [out[key]]
            return out

        return self.backbone(x)

    @staticmethod
    def _as_feature_list(outputs: Any) -> List[torch.Tensor]:
        if isinstance(outputs, torch.Tensor):
            return [outputs]
        if isinstance(outputs, dict):
            for key in ("x_norm_patchtokens", "patch_tokens", "tokens"):
                if key in outputs:
                    return [outputs[key]]
            return [v for v in outputs.values() if torch.is_tensor(v)]
        if isinstance(outputs, Iterable):
            feats = []
            for item in outputs:
                if isinstance(item, (tuple, list)):
                    item = item[0]
                if isinstance(item, dict):
                    feats.extend(DINOv3Adapter._as_feature_list(item))
                elif torch.is_tensor(item):
                    feats.append(item)
            return feats
        raise TypeError(f"Unsupported DINOv3 output type: {type(outputs)}")

    def _tokens_to_map(self, tokens: torch.Tensor, h_patch: int, w_patch: int) -> torch.Tensor:
        if tokens.dim() == 4:
            return tokens
        if tokens.dim() != 3:
            raise ValueError(f"Expected patch tokens [B, N, C], got shape {tuple(tokens.shape)}")

        b, n, c = tokens.shape
        patch_tokens = h_patch * w_patch
        if n < patch_tokens:
            raise ValueError(
                f"Not enough patch tokens to reshape: got {n}, need {patch_tokens} "
                f"for grid {h_patch}x{w_patch}."
            )
        if n > patch_tokens:
            # DINO-style outputs often prepend cls/register tokens. Keep the
            # final spatial patch tokens and drop any non-spatial prefix.
            tokens = tokens[:, -patch_tokens:, :]

        return tokens.transpose(1, 2).contiguous().view(b, c, h_patch, w_patch)

    def forward(self, x):
        h_patch = x.shape[2] // self.patch_size
        w_patch = x.shape[3] // self.patch_size

        outputs = self._forward_intermediate(x)
        token_features = self._as_feature_list(outputs)
        if not token_features:
            raise RuntimeError("DINOv3Adapter did not receive any tensor features from the backbone.")

        maps = [self._tokens_to_map(feat, h_patch, w_patch) for feat in token_features]
        fused = torch.mean(torch.stack(maps), dim=0) if len(maps) > 1 else maps[0]

        proj_feats = []
        for stride, projector in zip(self.out_strides, self.projector):
            scale = self.patch_size / stride
            size = (int(h_patch * scale), int(w_patch * scale))
            feat = F.interpolate(fused, size=size, mode="bilinear", align_corners=False)
            proj_feats.append(projector(feat))

        return proj_feats
