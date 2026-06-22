"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
"""

import torch.nn as nn

from ..core import register

__all__ = ['ECDet', 'ECSeg']


class _ECBase(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder']

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward_features(self, x):
        fsem_aux = None
        x = self.backbone(x)

        if isinstance(x, tuple) and len(x) == 2:
            x, fsem_aux = x

        x = self.encoder(x)
        return x, fsem_aux

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


@register()
class ECDet(_ECBase):

    def forward(self, x, targets=None):
        x, fsem_aux = self.forward_features(x)
        outputs = self.decoder(x, targets)

        if self.training and fsem_aux is not None and isinstance(outputs, dict):
            outputs["fsem_aux"] = fsem_aux

        return outputs


@register()
class ECSeg(_ECBase):

    def forward(self, x, targets=None):
        x, fsem_aux = self.forward_features(x)
        spatial_feat = x[0]
        outputs = self.decoder(x, targets, spatial_feat)

        if self.training and fsem_aux is not None and isinstance(outputs, dict):
            outputs["fsem_aux"] = fsem_aux

        return outputs
