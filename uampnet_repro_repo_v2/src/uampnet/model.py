"""
UAMP-Net model implementation.

The implementation keeps the three manuscript-level branches explicit:
- global feature branch;
- activation-guided local mining branch;
- Bayesian prototype evidence branch.

This file is intentionally verbose because the repository is intended as a readable
reproducibility bundle rather than a minimal model definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = object
    F = None


class ActivationResponseMap(nn.Module):
    """
    Class-specific activation response map.

    Given spatial feature tensor F_i in R^{B x D x H x W}, the module produces
    A_i^c(u,v) = sigmoid(sum_r alpha_cr F_i^r(u,v) + b_c).

    During training, the map for the ground-truth class can be selected.
    During inference, top-L candidate maps can be selected.
    """

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.class_projection = nn.Conv2d(feature_dim, num_classes, kernel_size=1, bias=True)

    def forward(self, feature_map: "torch.Tensor") -> "torch.Tensor":
        return torch.sigmoid(self.class_projection(feature_map))

    def select_maps(
        self,
        activation_maps: "torch.Tensor",
        labels: Optional["torch.Tensor"] = None,
        logits: Optional["torch.Tensor"] = None,
        top_l: int = 1,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        b, c, h, w = activation_maps.shape
        if labels is not None:
            index = labels.view(-1, 1, 1, 1).expand(-1, 1, h, w)
            selected = activation_maps.gather(1, index).squeeze(1)
            return selected, labels
        if logits is None:
            score = activation_maps.flatten(2).mean(-1)
        else:
            score = logits
        classes = score.topk(k=min(top_l, c), dim=1).indices
        index = classes[:, :1].view(-1, 1, 1, 1).expand(-1, 1, h, w)
        selected = activation_maps.gather(1, index).squeeze(1)
        return selected, classes[:, 0]


class LocalDescriptorExtractor(nn.Module):
    """
    Extract activation-weighted local descriptors from high-response regions.

    The descriptor z_ij^c is L2 normalized after multiplying by activation strength.
    For efficiency, this implementation uses top-k activation locations per image.
    """

    def __init__(self, feature_dim: int, descriptor_dim: int, pool_kernel: int = 3, max_regions: int = 36):
        super().__init__()
        self.feature_dim = feature_dim
        self.descriptor_dim = descriptor_dim
        self.max_regions = max_regions
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=1, padding=pool_kernel // 2)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, descriptor_dim),
            nn.LayerNorm(descriptor_dim),
            nn.GELU(),
            nn.Linear(descriptor_dim, descriptor_dim),
        )

    def forward(
        self,
        feature_map: "torch.Tensor",
        activation_map: "torch.Tensor",
        quantile: float = 0.85,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        b, d, h, w = feature_map.shape
        pooled = self.pool(feature_map)
        flat_a = activation_map.flatten(1)
        thresholds = torch.quantile(flat_a.detach(), q=quantile, dim=1, keepdim=True)
        mask = flat_a >= thresholds
        k = min(self.max_regions, h * w)
        vals, inds = flat_a.masked_fill(~mask, -1.0).topk(k=k, dim=1)
        flat_f = pooled.flatten(2).transpose(1, 2)
        gather_index = inds.unsqueeze(-1).expand(-1, -1, d)
        descriptors = flat_f.gather(1, gather_index)
        weights = torch.clamp(vals, min=0.0).unsqueeze(-1)
        descriptors = descriptors * weights
        descriptors = F.normalize(self.proj(descriptors), dim=-1, eps=1e-6)
        return descriptors, vals


class GlobalPrototypeFusion(nn.Module):
    """
    Fuse global feature with reliability-weighted prototype evidence.

    The prototype vector is first projected to the same hidden space. Global and
    local evidence are then concatenated with an elementwise interaction term.
    """

    def __init__(self, global_dim: int, proto_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.global_proj = nn.Linear(global_dim, hidden_dim)
        self.proto_proj = nn.Linear(proto_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, global_feature: "torch.Tensor", proto_feature: "torch.Tensor"):
        g = self.global_proj(global_feature)
        p = self.proto_proj(proto_feature)
        x = torch.cat([g, p, g * p], dim=-1)
        fused = self.fusion(x)
        logits = self.classifier(fused)
        return logits, fused


class LightweightBackbone(nn.Module):
    """
    Fallback CNN backbone for smoke tests. Real runs can swap this with timm
    ConvNeXt-Tiny, Swin-Tiny, ResNet-50, or EfficientNet-B3.
    """

    def __init__(self, out_dim: int = 384):
        super().__init__()
        widths = [32, 64, 128, out_dim]
        layers = []
        in_ch = 3
        for w in widths:
            layers += [
                nn.Conv2d(in_ch, w, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(w),
                nn.GELU(),
                nn.Conv2d(w, w, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(w),
                nn.GELU(),
                nn.MaxPool2d(2),
            ]
            in_ch = w
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class UAMPNet(nn.Module):
    """
    UAMP-Net wrapper.

    The model can run without a fitted prototype memory; in that case it uses
    a zero prototype vector and behaves as a global/activation baseline. Once a
    prototype memory is attached, local descriptors are matched against prototypes,
    uncertainty suppresses unreliable matches, and the fused representation is used
    for final classification.
    """

    def __init__(
        self,
        num_classes: int,
        descriptor_dim: int = 256,
        backbone_name: str = "lightweight",
        pretrained: bool = False,
        top_l_candidates: int = 3,
        arm_quantile: float = 0.85,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.descriptor_dim = descriptor_dim
        self.top_l_candidates = top_l_candidates
        self.arm_quantile = arm_quantile
        self.backbone_name = backbone_name

        self.backbone, feature_dim = self._build_backbone(backbone_name, pretrained)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.arm = ActivationResponseMap(feature_dim, num_classes)
        self.local_extractor = LocalDescriptorExtractor(feature_dim, descriptor_dim)
        self.fusion_head = GlobalPrototypeFusion(feature_dim, descriptor_dim, 384, num_classes)

        self.register_buffer("prototype_centers", torch.empty(0, descriptor_dim))
        self.register_buffer("prototype_classes", torch.empty(0, dtype=torch.long))
        self.register_buffer("prototype_reliability", torch.empty(0))
        self.register_buffer("prototype_uncertainty", torch.empty(0))
        self.prototype_ready = False

    def _build_backbone(self, backbone_name: str, pretrained: bool):
        if backbone_name == "lightweight" or backbone_name is None:
            bb = LightweightBackbone(384)
            return bb, bb.out_dim
        try:
            import timm
            model = timm.create_model(backbone_name, pretrained=pretrained, features_only=True, out_indices=(-1,))
            feature_info = model.feature_info[-1]
            return model, int(feature_info["num_chs"])
        except Exception:
            bb = LightweightBackbone(384)
            return bb, bb.out_dim

    def attach_prototypes(self, centers, classes, reliability, uncertainty):
        device = next(self.parameters()).device
        self.prototype_centers = torch.as_tensor(centers, dtype=torch.float32, device=device)
        self.prototype_classes = torch.as_tensor(classes, dtype=torch.long, device=device)
        self.prototype_reliability = torch.as_tensor(reliability, dtype=torch.float32, device=device)
        self.prototype_uncertainty = torch.as_tensor(uncertainty, dtype=torch.float32, device=device)
        self.prototype_ready = self.prototype_centers.numel() > 0

    def extract_features(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        out = self.backbone(x)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        global_feature = self.global_pool(out).flatten(1)
        return out, global_feature

    def _match_prototypes(self, descriptors: "torch.Tensor", pred_classes: "torch.Tensor") -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        b, m, d = descriptors.shape
        if not self.prototype_ready:
            return descriptors.mean(dim=1), {
                "matched_index": torch.full((b,), -1, device=descriptors.device, dtype=torch.long),
                "match_similarity": torch.zeros(b, device=descriptors.device),
                "match_reliability": torch.zeros(b, device=descriptors.device),
                "match_uncertainty": torch.ones(b, device=descriptors.device),
            }

        centers = F.normalize(self.prototype_centers, dim=-1)
        desc = F.normalize(descriptors, dim=-1)
        sim = torch.einsum("bmd,kd->bmk", desc, centers)
        class_mask = self.prototype_classes.view(1, 1, -1) == pred_classes.view(-1, 1, 1)
        sim = sim.masked_fill(~class_mask, -1e4)
        best_sim, best_idx = sim.max(dim=-1)
        reliability = self.prototype_reliability[best_idx].clamp(0, 1)
        uncertainty = self.prototype_uncertainty[best_idx].clamp(0, 1)
        weights = torch.softmax(best_sim * (0.5 + reliability) * (1.0 - 0.5 * uncertainty), dim=1)
        proto_feature = torch.sum(desc * weights.unsqueeze(-1), dim=1)
        dominant = weights.argmax(dim=1)
        matched_index = best_idx.gather(1, dominant.view(-1, 1)).squeeze(1)
        aux = {
            "matched_index": matched_index,
            "match_similarity": best_sim.gather(1, dominant.view(-1, 1)).squeeze(1),
            "match_reliability": self.prototype_reliability[matched_index],
            "match_uncertainty": self.prototype_uncertainty[matched_index],
        }
        return proto_feature, aux

    def forward(self, x: "torch.Tensor", labels: Optional["torch.Tensor"] = None):
        feature_map, global_feature = self.extract_features(x)
        activation_maps = self.arm(feature_map)
        if labels is not None:
            selected_map, selected_class = self.arm.select_maps(activation_maps, labels=labels)
            logits_initial = None
        else:
            # provisional class from global branch via average ARM response
            arm_logits = activation_maps.flatten(2).mean(-1)
            selected_map, selected_class = self.arm.select_maps(activation_maps, logits=arm_logits, top_l=self.top_l_candidates)
            logits_initial = arm_logits
        descriptors, region_scores = self.local_extractor(feature_map, selected_map, quantile=self.arm_quantile)
        proto_feature, aux = self._match_prototypes(descriptors, selected_class)
        logits, fused = self.fusion_head(global_feature, proto_feature)
        return {
            "logits": logits,
            "initial_logits": logits_initial,
            "feature_map": feature_map,
            "global_feature": global_feature,
            "activation_maps": activation_maps,
            "selected_activation_map": selected_map,
            "selected_class": selected_class,
            "local_descriptors": descriptors,
            "region_scores": region_scores,
            "fused_feature": fused,
            **aux,
        }
