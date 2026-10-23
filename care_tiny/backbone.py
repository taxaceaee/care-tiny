"""ImageNet-pretrained timm backbone returning the stride 4/8/16/32 features and the stride-2 stem activation."""

import timm
import torch.nn as nn
from timm.layers import freeze_batch_norm_2d


class Backbone(nn.Module):
    """timm feature extractor (HGNetv2 family). After each forward, ``self.s2`` holds the stride-2 stem activation
    (HGNetv2 ``stem.stem1``), which feeds the shared high-resolution map of the detector."""

    def __init__(self, name, pretrained=True, freeze_bn=True, freeze_stem=False, levels=4):
        super().__init__()
        self.body = timm.create_model(name, pretrained=pretrained, features_only=True,
                                      out_indices=tuple(range(levels)))
        self.channels = list(self.body.feature_info.channels())
        reductions = list(self.body.feature_info.reduction())
        if reductions != [4, 8, 16, 32][:levels]:
            raise ValueError(f"{name}: expected reductions {[4, 8, 16, 32][:levels]}, got {reductions}")
        stem1 = getattr(getattr(self.body, "stem", None), "stem1", None)
        if stem1 is None:
            raise ValueError(f"{name}: no stem.stem1 module for the stride-2 hook")
        self.s2 = None
        self.s2_channels = stem1.conv.out_channels
        stem1.register_forward_hook(self._keep_s2)
        if freeze_bn:
            # handles timm BatchNormAct2d (keeps the fused activation), as D-FINE freezes the HGNetv2 norms
            self.body = freeze_batch_norm_2d(self.body)
        if freeze_stem:
            for n, p in self.body.named_parameters():
                if n.startswith("stem"):
                    p.requires_grad_(False)

    def _keep_s2(self, module, inp, out):
        self.s2 = out

    def forward(self, x):
        return self.body(x)
