import math

import torch
import torch.nn.functional as F
from torch import nn


class GaussianBindingModule(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        anchor_dim: int = 10,
        anchor_context_dim: int = 0,
        gate_init_bias: float = -2.2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.anchor_dim = anchor_dim
        self.anchor_context_dim = anchor_context_dim
        self.gate_init_bias = gate_init_bias

        input_dim = anchor_dim + 4 * feature_dim + 2 + anchor_context_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.gate_th2rgb_head = nn.Linear(hidden_dim, feature_dim)
        self.gate_rgb2th_head = nn.Linear(hidden_dim, feature_dim)
        self.delta_th2rgb_head = nn.Linear(hidden_dim, feature_dim)
        self.delta_rgb2th_head = nn.Linear(hidden_dim, feature_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        for head in (
            self.gate_th2rgb_head,
            self.gate_rgb2th_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, self.gate_init_bias)

        for head in (
            self.delta_th2rgb_head,
            self.delta_rgb2th_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _build_fused_state(self, anchor_state, rgb_flat, thermal_flat, anchor_context=None):
        feature_diff = rgb_flat - thermal_flat
        feature_abs_diff = feature_diff.abs()
        feature_product = rgb_flat * thermal_flat
        discrepancy_norm = feature_diff.norm(dim=1, keepdim=True) / math.sqrt(max(self.feature_dim, 1))
        cosine_dissimilarity = 1.0 - F.cosine_similarity(rgb_flat, thermal_flat, dim=1, eps=1e-6).unsqueeze(1)
        if self.anchor_context_dim > 0:
            if anchor_context is None:
                anchor_context = anchor_state.new_zeros((anchor_state.shape[0], self.anchor_context_dim))
            else:
                anchor_context = anchor_context.reshape(anchor_state.shape[0], -1)
                if anchor_context.shape[1] != self.anchor_context_dim:
                    raise ValueError(
                        f"Expected anchor_context_dim={self.anchor_context_dim}, got {anchor_context.shape[1]}."
                    )
        else:
            anchor_context = anchor_state.new_zeros((anchor_state.shape[0], 0))
        return torch.cat(
            (
                anchor_state,
                rgb_flat,
                thermal_flat,
                feature_abs_diff,
                feature_product,
                discrepancy_norm,
                cosine_dissimilarity,
                anchor_context,
            ),
            dim=1,
        )

    def forward(
        self,
        anchor_state,
        rgb_features,
        thermal_features,
        thermal_context_features=None,
        anchor_context=None,
    ):
        rgb_flat = rgb_features.reshape(rgb_features.shape[0], -1)
        if thermal_context_features is None:
            thermal_context_features = thermal_features
        thermal_context_flat = thermal_context_features.reshape(thermal_context_features.shape[0], -1)
        fused_state = self._build_fused_state(
            anchor_state,
            rgb_flat,
            thermal_context_flat,
            anchor_context=anchor_context,
        )
        hidden = self.backbone(fused_state)

        gate_th2rgb = torch.sigmoid(self.gate_th2rgb_head(hidden)).view_as(rgb_features)
        gate_rgb2th = torch.sigmoid(self.gate_rgb2th_head(hidden)).view_as(thermal_features)
        delta_th2rgb = torch.tanh(self.delta_th2rgb_head(hidden)).view_as(rgb_features)
        delta_rgb2th = torch.tanh(self.delta_rgb2th_head(hidden)).view_as(thermal_features)

        updated_rgb_features = rgb_features + gate_th2rgb * delta_th2rgb
        updated_thermal_features = thermal_features + gate_rgb2th * delta_rgb2th

        return {
            "gate_th2rgb": gate_th2rgb,
            "gate_rgb2th": gate_rgb2th,
            "delta_th2rgb": delta_th2rgb,
            "delta_rgb2th": delta_rgb2th,
            "updated_rgb_features": updated_rgb_features,
            "updated_thermal_features": updated_thermal_features,
        }
