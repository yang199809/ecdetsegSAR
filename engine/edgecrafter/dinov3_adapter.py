"""
DINOv3 adapter for ECDet.

This module keeps the ECDet neck/decoder contract unchanged by converting
Hugging Face DINOv3/ViT-S patch tokens into three 256-channel feature maps
at strides 8, 16, and 32.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register
from .hybrid_encoder import ConvNormLayer_fuse

__all__ = ["DINOv3Adapter"]


def _candidate_keys(key: str) -> List[str]:
    prefixes = ("module.", "backbone.", "teacher.", "student.", "model.")
    candidates = [key]
    changed = True
    current = key
    while changed:
        changed = False
        for prefix in prefixes:
            if current.startswith(prefix):
                current = current[len(prefix):]
                candidates.append(current)
                changed = True
    return candidates


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
        source="huggingface",
        hf_model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        hf_cache_dir=None,
        local_files_only=False,
        trust_remote_code=False,
        weights_path=None,
        pretrained=True,
        embed_dim=384,
        num_heads=6,
        num_register_tokens=4,
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
        self.source = source
        self.hf_model_id = hf_model_id
        self.hf_cache_dir = hf_cache_dir
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens
        self.proj_dim = proj_dim
        self.out_indices = list(out_indices)
        self.patch_size = patch_size
        self.out_strides = list(out_strides)
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self._weights_loaded_by_builder = False

        self.backbone = self._build_hf_backbone(
            pretrained=pretrained and not skip_load_backbone,
            drop_path_rate=drop_path_rate,
            weights_path=weights_path,
            **kwargs,
        )
        self._sync_backbone_dims()
        self.projector = nn.ModuleList([
            ConvNormLayer_fuse(self.embed_dim, proj_dim, kernel_size=1, stride=1, act="silu")
            for _ in self.out_strides
        ])

        self._apply_checkpoint_flag()
        self._freeze_stages()

        if pretrained and not skip_load_backbone and not self._weights_loaded_by_builder:
            self._load_weights(weights_path)

    def _build_hf_backbone(self, pretrained=True, drop_path_rate=0.0, weights_path=None, **kwargs):
        if self.source not in ("huggingface", "hf"):
            raise ValueError(
                "DINOv3Adapter now builds the backbone only from Hugging Face/Transformers. "
                "Please set source: huggingface."
            )

        try:
            from transformers import AutoModel
        except Exception as exc:
            raise RuntimeError(
                "DINOv3Adapter(source='huggingface') requires transformers with DINOv3 support "
                "(transformers>=4.56.0 is recommended)."
            ) from exc

        path = Path(weights_path) if weights_path else None

        if path is not None and path.is_dir():
            try:
                model = AutoModel.from_pretrained(
                    str(path),
                    cache_dir=self.hf_cache_dir,
                    local_files_only=self.local_files_only,
                    trust_remote_code=self.trust_remote_code,
                    output_hidden_states=True,
                    **kwargs,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load local Hugging Face DINOv3 snapshot. "
                    "The directory should contain config.json and model weights downloaded "
                    f"from {self.hf_model_id}. weights_path={weights_path}, "
                    f"local_files_only={self.local_files_only}."
                ) from exc
            self._weights_loaded_by_builder = True
            print(f"[DINOv3Adapter] Loaded Hugging Face DINOv3 backbone from local directory: {path}")
            return model

        if path is not None and path.is_file():
            print(f"[DINOv3Adapter] Building Hugging Face DINOv3 backbone and loading local weight file: {path}")
            return self._build_hf_model_from_config(drop_path_rate=drop_path_rate, **kwargs)

        if pretrained and weights_path:
            raise RuntimeError(
                f"DINOv3Adapter pretrained=True but weights_path does not exist: {weights_path}. "
                "Download the Hugging Face snapshot first, or point weights_path to a local .pth/.bin/.safetensors file."
            )

        if pretrained:
            try:
                model = AutoModel.from_pretrained(
                    self.hf_model_id,
                    cache_dir=self.hf_cache_dir,
                    local_files_only=self.local_files_only,
                    trust_remote_code=self.trust_remote_code,
                    output_hidden_states=True,
                    **kwargs,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to download/load DINOv3 from Hugging Face. Check network/access permissions, "
                    f"hf_model_id={self.hf_model_id}, local_files_only={self.local_files_only}."
                ) from exc
            self._weights_loaded_by_builder = True
            print(f"[DINOv3Adapter] Downloaded/loaded Hugging Face DINOv3 backbone: {self.hf_model_id}")
            return model

        return self._build_hf_model_from_config(drop_path_rate=drop_path_rate, **kwargs)

    def _build_hf_model_from_config(self, drop_path_rate=0.0, **kwargs):
        try:
            from transformers import DINOv3ViTConfig, DINOv3ViTModel
        except Exception as exc:
            raise RuntimeError(
                "Random DINOv3 construction requires transformers>=4.56.0 with "
                "DINOv3ViTConfig/DINOv3ViTModel."
            ) from exc

        config = DINOv3ViTConfig(
            image_size=kwargs.pop("image_size", 640),
            patch_size=self.patch_size,
            hidden_size=self.embed_dim,
            num_hidden_layers=max(self.out_indices) + 1,
            num_attention_heads=self.num_heads,
            num_register_tokens=self.num_register_tokens,
            drop_path_rate=drop_path_rate,
            output_hidden_states=True,
        )
        return DINOv3ViTModel(config)

    def _sync_backbone_dims(self):
        config = getattr(self.backbone, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is not None and hidden_size != self.embed_dim:
            print(f"[DINOv3Adapter] Using backbone hidden_size={hidden_size} instead of embed_dim={self.embed_dim}.")
            self.embed_dim = hidden_size

        patch_size = getattr(config, "patch_size", None)
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        if patch_size is not None and patch_size != self.patch_size:
            print(f"[DINOv3Adapter] Using backbone patch_size={patch_size} instead of patch_size={self.patch_size}.")
            self.patch_size = patch_size

    def _apply_checkpoint_flag(self):
        if not self.use_checkpoint:
            return
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
            return
        for module in self.backbone.modules():
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = True

    def _freeze_stages(self):
        if self.frozen_stages < 0:
            return
        if hasattr(self.backbone, "embeddings"):
            self.backbone.embeddings.eval()
            for p in self.backbone.embeddings.parameters():
                p.requires_grad = False
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
        encoder = getattr(self.backbone, "encoder", None)
        hf_encoder = getattr(self.backbone, "model", None)
        layers = (
            getattr(encoder, "layer", None)
            or getattr(encoder, "layers", None)
            or getattr(hf_encoder, "layer", None)
            or getattr(hf_encoder, "layers", None)
        )
        if layers is not None:
            for layer in list(layers)[: self.frozen_stages]:
                layer.eval()
                for p in layer.parameters():
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
            raise RuntimeError(f"DINOv3Adapter weights_path not found: {path}")

        if path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except Exception as exc:
                raise RuntimeError("Loading .safetensors weights requires the safetensors package.") from exc
            ckpt = load_file(str(path), device="cpu")
        else:
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                ckpt = torch.load(path, map_location="cpu")

        raw_state = _unwrap_checkpoint(ckpt)
        state = {}
        target_keys = set(self.backbone.state_dict().keys())
        ignored_prefixes = ("head.", "fc.", "classifier.")
        for key, value in raw_state.items():
            candidates = _candidate_keys(key)
            if any(candidate.startswith(ignored_prefixes) for candidate in candidates):
                continue
            clean_key = next((candidate for candidate in candidates if candidate in target_keys), candidates[-1])
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
        if self.source in ("huggingface", "hf"):
            outputs = self.backbone(pixel_values=x, output_hidden_states=True, return_dict=True)
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is not None:
                offset = 1 if len(hidden_states) > max(self.out_indices) + 1 else 0
                return [hidden_states[i + offset] for i in self.out_indices]
            return [outputs.last_hidden_state]

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
