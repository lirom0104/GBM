#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr



import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.gbm import GaussianBindingModule

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(
        self,
        sh_degree : int,
        use_gbm : bool = False,
        use_thermal_residual_geometry : bool = False,
        gbm_hidden_dim : int = 32,
        gbm_gate_init_bias : float = -2.2,
        gbm_thermal_grayscale_context : bool = True,
        gbm_rgb_luma_transfer_only : bool = True,
    ):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._thermal_dc = torch.empty(0)
        self._thermal_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity_base = torch.empty(0)
        self._opacity_bias_rgb = torch.empty(0)
        self._opacity_bias_th = torch.empty(0)
        self._delta_xyz_th = torch.empty(0)
        self._delta_scaling_th = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.anchor_lifecycle = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.thermal_residual_xyz_scheduler_args = None
        self.anchor_multimodal_stats = {}
        self.anchor_stats_ema = 0.95
        self.save_anchor_stats_enabled = False
        self.joint_lifecycle_enabled = False
        self.joint_lifecycle_warmup_iters = 0
        self.joint_split_rgb_weight = 1.0
        self.joint_split_th_weight = 1.0
        self.joint_split_gbm_boost = 0.04
        self.joint_split_thgeo_boost = 0.04
        self.joint_prune_visibility_thresh = 0.05
        self.joint_prune_contribution_thresh = 0.25
        self.joint_prune_residual_thresh = 0.25
        self.joint_prune_gbm_veto_thresh = 1.25
        self.joint_prune_thgeo_veto_thresh = 1.25
        self.joint_split_score_threshold = 1.2
        self.joint_split_max_extra_ratio = 0.15
        self.joint_lifecycle_max_point_ratio = 1.8
        self.joint_lifecycle_start_point_count = 0
        self.joint_lifecycle_budget_reference_count = 0
        # Fresh densified anchors should earn their lifecycle stats from recent evidence.
        self.new_anchor_stats_inherit_scale = 0.0
        self.last_joint_lifecycle_selection_diagnostics = {
            "joint_split_trigger_count": 0.0,
            "split_suppressed_by_budget_count": 0.0,
        }
        self.last_joint_lifecycle_diagnostics = {}
        self.use_gbm = use_gbm
        self.use_thermal_residual_geometry = use_thermal_residual_geometry
        self.gbm_hidden_dim = gbm_hidden_dim
        self.gbm_gate_init_bias = gbm_gate_init_bias
        self.gbm_thermal_grayscale_context = gbm_thermal_grayscale_context
        self.gbm_rgb_luma_transfer_only = gbm_rgb_luma_transfer_only
        self.gbm_gate_target_std = 0.1
        self.gbm = None
        self.setup_functions()
        self._configure_gbm_module()

    def capture(self):
        return {
            "version": 3,
            "active_sh_degree": self.active_sh_degree,
            "xyz": self._xyz,
            "features_dc": self._features_dc,
            "features_rest": self._features_rest,
            "thermal_dc": self._thermal_dc,
            "thermal_rest": self._thermal_rest,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity_base": self._opacity_base,
            "opacity_bias_rgb": self._opacity_bias_rgb,
            "opacity_bias_th": self._opacity_bias_th,
            "delta_xyz_th": self._delta_xyz_th,
            "delta_scaling_th": self._delta_scaling_th,
            "max_radii2D": self.max_radii2D,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "denom": self.denom,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "spatial_lr_scale": self.spatial_lr_scale,
            "use_gbm": self.use_gbm,
            "use_thermal_residual_geometry": self.use_thermal_residual_geometry,
            "gbm_hidden_dim": self.gbm_hidden_dim,
            "gbm_gate_init_bias": self.gbm_gate_init_bias,
            "gbm_thermal_grayscale_context": self.gbm_thermal_grayscale_context,
            "gbm_rgb_luma_transfer_only": self.gbm_rgb_luma_transfer_only,
            "gbm_gate_target_std": self.gbm_gate_target_std,
            "gbm_state": self.gbm.state_dict() if self.gbm is not None else None,
            "anchor_multimodal_stats": self.get_anchor_multimodal_stats(),
            "anchor_stats_ema": self.anchor_stats_ema,
            "save_anchor_stats_enabled": self.save_anchor_stats_enabled,
        }
    
    def restore(self, model_args, training_args):
        legacy_checkpoint = not isinstance(model_args, dict)
        anchor_multimodal_stats = None
        if legacy_checkpoint:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._thermal_dc,
                self._thermal_rest,
                self._scaling,
                self._rotation,
                opacity_legacy,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args
            self._opacity_base = opacity_legacy
            self._opacity_bias_rgb = self._make_zero_parameter_like(self._opacity_base)
            self._opacity_bias_th = self._make_zero_parameter_like(self._opacity_base)
            self._delta_xyz_th = self._make_zero_parameter_like(self._xyz)
            self._delta_scaling_th = self._make_zero_parameter_like(self._scaling)
            anchor_multimodal_stats = None
        else:
            self.active_sh_degree = model_args["active_sh_degree"]
            self._xyz = model_args["xyz"]
            self._features_dc = model_args["features_dc"]
            self._features_rest = model_args["features_rest"]
            self._thermal_dc = model_args["thermal_dc"]
            self._thermal_rest = model_args["thermal_rest"]
            self._scaling = model_args["scaling"]
            self._rotation = model_args["rotation"]
            self._opacity_base = model_args["opacity_base"] if "opacity_base" in model_args else model_args["opacity"]
            self._opacity_bias_rgb = (
                model_args["opacity_bias_rgb"] if "opacity_bias_rgb" in model_args else self._make_zero_parameter_like(self._opacity_base)
            )
            self._opacity_bias_th = (
                model_args["opacity_bias_th"] if "opacity_bias_th" in model_args else self._make_zero_parameter_like(self._opacity_base)
            )
            self._delta_xyz_th = (
                model_args["delta_xyz_th"] if "delta_xyz_th" in model_args else self._make_zero_parameter_like(self._xyz)
            )
            self._delta_scaling_th = (
                model_args["delta_scaling_th"] if "delta_scaling_th" in model_args else self._make_zero_parameter_like(self._scaling)
            )
            self.max_radii2D = model_args["max_radii2D"]
            xyz_gradient_accum = model_args["xyz_gradient_accum"]
            denom = model_args["denom"]
            opt_dict = model_args.get("optimizer_state")
            self.spatial_lr_scale = model_args["spatial_lr_scale"]
            self.use_gbm = model_args.get("use_gbm", self.use_gbm)
            self.use_thermal_residual_geometry = model_args.get(
                "use_thermal_residual_geometry", self.use_thermal_residual_geometry
            )
            self.gbm_hidden_dim = model_args.get("gbm_hidden_dim", self.gbm_hidden_dim)
            self.gbm_gate_init_bias = model_args.get("gbm_gate_init_bias", self.gbm_gate_init_bias)
            self.gbm_thermal_grayscale_context = model_args.get(
                "gbm_thermal_grayscale_context", self.gbm_thermal_grayscale_context
            )
            self.gbm_rgb_luma_transfer_only = model_args.get(
                "gbm_rgb_luma_transfer_only", self.gbm_rgb_luma_transfer_only
            )
            self.gbm_gate_target_std = model_args.get("gbm_gate_target_std", self.gbm_gate_target_std)
            self.anchor_stats_ema = model_args.get("anchor_stats_ema", self.anchor_stats_ema)
            self.save_anchor_stats_enabled = model_args.get("save_anchor_stats_enabled", self.save_anchor_stats_enabled)
            anchor_multimodal_stats = model_args.get("anchor_multimodal_stats")

        self._configure_gbm_module()
        self._refresh_optional_parameter_grad_flags()
        self.training_setup(training_args)
        gbm_state = model_args.get("gbm_state") if isinstance(model_args, dict) else None
        self._load_gbm_state_compat(gbm_state)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.anchor_lifecycle = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._restore_anchor_multimodal_stats(anchor_multimodal_stats, self.get_xyz.shape[0])
        if opt_dict is None:
            return
        if legacy_checkpoint:
            print("[Warning] Loaded a legacy checkpoint; optimizer state is reinitialized for the refactored GaussianModel.")
            return
        try:
            self.optimizer.load_state_dict(opt_dict)
        except ValueError:
            print("[Warning] Optimizer state is incompatible with the refactored GaussianModel and was reinitialized.")

    def _make_zero_parameter_like(self, reference):
        return nn.Parameter(torch.zeros_like(reference).requires_grad_(True))

    def _gbm_feature_dim(self):
        return 3 * ((self.max_sh_degree + 1) ** 2)

    def _gbm_anchor_context_dim(self):
        return 4

    def _configure_gbm_module(self):
        if not self.use_gbm:
            self.gbm = None
            return

        expected_feature_dim = self._gbm_feature_dim()
        expected_anchor_context_dim = self._gbm_anchor_context_dim()
        if (
            self.gbm is None
            or self.gbm.feature_dim != expected_feature_dim
            or self.gbm.hidden_dim != self.gbm_hidden_dim
            or getattr(self.gbm, "anchor_context_dim", 0) != expected_anchor_context_dim
            or self.gbm.gate_init_bias != self.gbm_gate_init_bias
        ):
            self.gbm = GaussianBindingModule(
                feature_dim=expected_feature_dim,
                hidden_dim=self.gbm_hidden_dim,
                anchor_context_dim=expected_anchor_context_dim,
                gate_init_bias=self.gbm_gate_init_bias,
            ).cuda()

    def _load_gbm_state_compat(self, gbm_state):
        if gbm_state is None or self.gbm is None:
            return

        current_state = self.gbm.state_dict()
        adapted_state = {}
        incompatible_keys = []

        for key, current_value in current_state.items():
            loaded_value = gbm_state.get(key)
            if loaded_value is None:
                adapted_state[key] = current_value
                incompatible_keys.append(key)
                continue

            if loaded_value.shape == current_value.shape:
                adapted_state[key] = loaded_value
                continue

            if key == "backbone.0.weight" and loaded_value.shape[0] == current_value.shape[0]:
                adapted_weight = current_value.new_zeros(current_value.shape)
                overlap = min(loaded_value.shape[1], current_value.shape[1])
                adapted_weight[:, :overlap] = loaded_value[:, :overlap]
                adapted_state[key] = adapted_weight
                continue

            adapted_state[key] = current_value
            incompatible_keys.append(key)

        self.gbm.load_state_dict(adapted_state, strict=False)
        if incompatible_keys:
            print(
                "[Warning] GBM checkpoint partially reused; incompatible keys were reinitialized: {}".format(
                    ", ".join(sorted(incompatible_keys))
                )
            )

    def _legacy_shared_mode(self):
        return (not self.use_gbm) and (not self.use_thermal_residual_geometry)

    def _anchor_stat_names(self):
        return (
            "visibility_rgb",
            "visibility_th",
            "contribution_rgb",
            "contribution_th",
            "residual_rgb",
            "residual_th",
            "gbm_usage_th2rgb",
            "gbm_usage_rgb2th",
            "thermal_geometry_usage",
        )

    def _anchor_stats_device(self):
        if self._xyz.numel() > 0:
            return self._xyz.device
        return torch.device("cuda")

    def _empty_anchor_multimodal_stats(self, num_anchors=0, device=None):
        device = self._anchor_stats_device() if device is None else device
        return {
            stat_name: torch.zeros((num_anchors, 1), device=device)
            for stat_name in self._anchor_stat_names()
        }

    def _initialize_anchor_multimodal_stats(self, num_anchors, device=None):
        self.anchor_multimodal_stats = self._empty_anchor_multimodal_stats(num_anchors, device=device)

    def _anchor_multimodal_stats_size(self):
        if not self.anchor_multimodal_stats:
            return 0
        first_stat = next(iter(self.anchor_multimodal_stats.values()), None)
        if first_stat is None:
            return 0
        return first_stat.shape[0]

    def _resize_anchor_multimodal_stats(self, num_anchors, device=None):
        device = self._anchor_stats_device() if device is None else device
        resized_stats = self._empty_anchor_multimodal_stats(num_anchors, device=device)
        current_size = self._anchor_multimodal_stats_size()
        if current_size == 0:
            self.anchor_multimodal_stats = resized_stats
            return

        overlap = min(current_size, num_anchors)
        for stat_name in self._anchor_stat_names():
            stat_tensor = self.anchor_multimodal_stats.get(stat_name)
            if stat_tensor is None:
                continue
            stat_tensor = torch.as_tensor(stat_tensor, device=device)
            if stat_tensor.ndim == 1:
                stat_tensor = stat_tensor.unsqueeze(1)
            resized_stats[stat_name][:overlap] = stat_tensor[:overlap].detach().clone()
        self.anchor_multimodal_stats = resized_stats

    def _ensure_anchor_multimodal_stats(self, num_anchors=None, device=None):
        if num_anchors is None:
            num_anchors = self.get_xyz.shape[0]
        device = self._anchor_stats_device() if device is None else device
        if not self.anchor_multimodal_stats:
            self._initialize_anchor_multimodal_stats(num_anchors, device=device)
            return

        expected_names = set(self._anchor_stat_names())
        current_names = set(self.anchor_multimodal_stats.keys())
        shape_matches = all(
            self.anchor_multimodal_stats[stat_name].shape[0] == num_anchors
            for stat_name in current_names
        )
        if current_names != expected_names or not shape_matches:
            self._resize_anchor_multimodal_stats(num_anchors, device=device)
            current_names = set(self.anchor_multimodal_stats.keys())

        for stat_name in expected_names - current_names:
            self.anchor_multimodal_stats[stat_name] = torch.zeros((num_anchors, 1), device=device)
        for stat_name in expected_names:
            self.anchor_multimodal_stats[stat_name] = self.anchor_multimodal_stats[stat_name].to(device)

    def _restore_anchor_multimodal_stats(self, stats_dict, num_anchors):
        device = self._anchor_stats_device()
        self._initialize_anchor_multimodal_stats(num_anchors, device=device)
        if not isinstance(stats_dict, dict):
            return

        for stat_name in self._anchor_stat_names():
            stat_value = stats_dict.get(stat_name)
            if stat_value is None:
                continue
            stat_tensor = torch.as_tensor(stat_value, device=device)
            if stat_tensor.ndim == 1:
                stat_tensor = stat_tensor.unsqueeze(1)
            if stat_tensor.shape[0] != num_anchors:
                continue
            self.anchor_multimodal_stats[stat_name] = stat_tensor.detach().clone()

    def _prune_anchor_multimodal_stats(self, valid_points_mask):
        self._ensure_anchor_multimodal_stats(valid_points_mask.shape[0])
        for stat_name in self._anchor_stat_names():
            self.anchor_multimodal_stats[stat_name] = self.anchor_multimodal_stats[stat_name][valid_points_mask].detach().clone()

    def _select_anchor_multimodal_stats(self, selected_pts_mask, repeat=1, inherit_scale=1.0):
        self._ensure_anchor_multimodal_stats(selected_pts_mask.shape[0])
        device = self._anchor_stats_device()
        num_selected = int(selected_pts_mask.sum().item()) * repeat
        if num_selected == 0:
            return self._empty_anchor_multimodal_stats(0, device=device)
        if inherit_scale <= 0.0:
            return self._empty_anchor_multimodal_stats(num_selected, device=device)

        selected_stats = {}
        scale = float(inherit_scale)
        for stat_name in self._anchor_stat_names():
            selected = self.anchor_multimodal_stats[stat_name][selected_pts_mask]
            if repeat != 1:
                selected = selected.repeat(repeat, 1)
            selected_stats[stat_name] = (selected * scale).detach().clone()
        return selected_stats

    def _extend_anchor_multimodal_stats(self, extension_stats, num_new_anchors, previous_num_anchors=None):
        device = self._anchor_stats_device()
        if previous_num_anchors is None:
            previous_num_anchors = max(self.get_xyz.shape[0] - num_new_anchors, 0)
        self._ensure_anchor_multimodal_stats(previous_num_anchors, device=device)
        if extension_stats is None:
            extension_stats = self._empty_anchor_multimodal_stats(num_new_anchors, device=device)

        for stat_name in self._anchor_stat_names():
            extension_value = extension_stats.get(stat_name)
            if extension_value is None:
                extension_tensor = torch.zeros((num_new_anchors, 1), device=device)
            else:
                extension_tensor = torch.as_tensor(extension_value, device=device)
                if extension_tensor.ndim == 1:
                    extension_tensor = extension_tensor.unsqueeze(1)
            self.anchor_multimodal_stats[stat_name] = torch.cat(
                (self.anchor_multimodal_stats[stat_name], extension_tensor.detach().clone()),
                dim=0,
            )

    def _prepare_anchor_stat_observation(self, values, num_anchors, visibility_mask=None, default_value=0.0):
        device = self._anchor_stats_device()
        if values is None:
            observed = torch.full((num_anchors,), float(default_value), device=device)
        else:
            observed_values = torch.as_tensor(values, device=device).detach().reshape(-1)
            observed_values = torch.nan_to_num(
                observed_values,
                nan=float(default_value),
                posinf=float(default_value),
                neginf=float(default_value),
            )
            if observed_values.numel() == 1 and num_anchors != 1:
                observed_values = observed_values.expand(num_anchors)
            observed = torch.full(
                (num_anchors,),
                float(default_value),
                device=device,
                dtype=observed_values.dtype,
            )
            overlap = min(num_anchors, observed_values.shape[0])
            if overlap > 0:
                observed[:overlap] = observed_values[:overlap]

        if visibility_mask is not None:
            visibility_mask = visibility_mask.detach().reshape(-1).bool()
            if visibility_mask.numel() != num_anchors:
                raise ValueError("Anchor stat observation expects visibility masks with one value per anchor.")
            observed = observed.clone()
            observed[~visibility_mask] = float(default_value)

        return observed.reshape(-1, 1)

    def _ema_update_anchor_stat(self, stat_name, observed_values, ema):
        self._ensure_anchor_multimodal_stats()
        target = self.anchor_multimodal_stats[stat_name]
        if observed_values.shape != target.shape:
            raise ValueError("Anchor stat EMA update expects full per-anchor observations.")
        target.mul_(ema).add_(observed_values * (1.0 - ema))

    def _refresh_optional_parameter_grad_flags(self):
        bias_requires_grad = True
        delta_requires_grad = self.use_thermal_residual_geometry
        if isinstance(self._opacity_bias_rgb, nn.Parameter):
            self._opacity_bias_rgb.requires_grad_(bias_requires_grad)
        if isinstance(self._opacity_bias_th, nn.Parameter):
            self._opacity_bias_th.requires_grad_(bias_requires_grad)
        if isinstance(self._delta_xyz_th, nn.Parameter):
            self._delta_xyz_th.requires_grad_(delta_requires_grad)
        if isinstance(self._delta_scaling_th, nn.Parameter):
            self._delta_scaling_th.requires_grad_(delta_requires_grad)

    def _parameter_from_tensor(self, tensor, requires_grad):
        return nn.Parameter(tensor.detach().clone(), requires_grad=requires_grad)

    def _masked_parameter(self, parameter, mask):
        return self._parameter_from_tensor(parameter[mask], parameter.requires_grad)

    def _concatenated_parameter(self, parameter, extension_tensor):
        return self._parameter_from_tensor(torch.cat((parameter, extension_tensor), dim=0), parameter.requires_grad)

    def _gbm_regularization_zero(self):
        if self._xyz.numel() > 0:
            return self._xyz.new_zeros(())
        return torch.tensor(0.0, device="cuda")

    def _gbm_summary_stats(self, values):
        if values.numel() == 0:
            zero = self._gbm_regularization_zero()
            return zero, zero, zero
        return values.mean(), values.std(unbiased=False), values.max()

    def _gbm_anchorwise_mean(self, tensor):
        return tensor.flatten(start_dim=1).mean(dim=1)

    def _gbm_anchorwise_norm(self, tensor):
        return tensor.flatten(start_dim=1).norm(dim=1)

    def _gbm_luma_replicated_features(self, tensor):
        if tensor is None or tensor.numel() == 0 or tensor.shape[-1] != 3:
            return tensor
        luma_weights = tensor.new_tensor((0.299, 0.587, 0.114)).view(1, 1, 3)
        luma = (tensor * luma_weights).sum(dim=-1, keepdim=True)
        return luma.expand_as(tensor)

    def _gbm_chunked_gated_l1(self, gate_tensor, delta_tensor, chunk_size=16384):
        if gate_tensor.numel() == 0:
            return self._gbm_regularization_zero()

        total = self._gbm_regularization_zero()
        num_anchors = gate_tensor.shape[0]
        for chunk_start in range(0, num_anchors, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_anchors)
            gate_chunk = gate_tensor[chunk_start:chunk_end]
            delta_chunk = delta_tensor[chunk_start:chunk_end]
            total = total + torch.sum(gate_chunk * delta_chunk.abs())
        return total / float(gate_tensor.numel())

    def _get_anchor_summary(self):
        return torch.cat((self.get_xyz, self.get_scaling, self.get_rotation), dim=1)

    def _build_render_params(self, means3D, scaling, features, opacity, scaling_modifier=1.0):
        return {
            "means3D": means3D,
            "xyz": means3D,
            "features": features,
            "shs": features,
            "opacity": opacity,
            "scales": scaling,
            "scaling": scaling,
            "rotations": self.get_rotation,
            "rotation": self.get_rotation,
            "cov3D_precomp": self.get_covariance(scaling_modifier=scaling_modifier, scaling=scaling, rotation=self._rotation),
        }

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_rgb_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features(self):
        return self.get_rgb_features
    
    @property
    def get_thermal_features(self):  
        thermal_dc = self._thermal_dc
        thermal_rest = self._thermal_rest
        return torch.cat((thermal_dc, thermal_rest), dim=1)
    
    @property
    def get_opacity_base(self):
        return self.opacity_activation(self._opacity_base)
    
    @property
    def get_rgb_opacity(self):
        return self.opacity_activation(self._opacity_base + self._opacity_bias_rgb)

    @property
    def get_thermal_opacity(self):
        return self.opacity_activation(self._opacity_base + self._opacity_bias_th)

    @property
    def get_opacity(self):
        return torch.maximum(self.get_rgb_opacity, self.get_thermal_opacity)

    @property
    def get_thermal_xyz(self):
        if not self.use_thermal_residual_geometry:
            return self.get_xyz
        return self._xyz + self._delta_xyz_th

    @property
    def get_thermal_scaling(self):
        if not self.use_thermal_residual_geometry:
            return self.get_scaling
        return self.scaling_activation(self._scaling + self._delta_scaling_th)

    def get_covariance(self, scaling_modifier = 1, scaling = None, rotation = None):
        scaling = self.get_scaling if scaling is None else scaling
        rotation = self._rotation if rotation is None else rotation
        return self.covariance_activation(scaling, scaling_modifier, rotation)

    def get_gbm_feature_bindings(self):
        rgb_features = self.get_rgb_features
        thermal_features = self.get_thermal_features
        if not self.use_gbm or self.gbm is None:
            zero = self._gbm_regularization_zero()
            return {
                "gate_th2rgb": None,
                "gate_rgb2th": None,
                "delta_th2rgb": None,
                "delta_rgb2th": None,
                "updated_rgb_features": rgb_features,
                "updated_thermal_features": thermal_features,
                "gate_th2rgb_anchor": rgb_features.new_zeros((rgb_features.shape[0],)),
                "gate_rgb2th_anchor": thermal_features.new_zeros((thermal_features.shape[0],)),
                "stability_reg": zero,
                "gate_sparsity_reg": zero,
                "gate_collapse_reg": zero,
                "gate_overlap_reg": zero,
                "gate_th2rgb_mean": zero,
                "gate_th2rgb_std": zero,
                "gate_th2rgb_max": zero,
                "gate_rgb2th_mean": zero,
                "gate_rgb2th_std": zero,
                "gate_rgb2th_max": zero,
                "delta_th2rgb_mag_mean": zero,
                "delta_rgb2th_mag_mean": zero,
            }

        thermal_context_features = thermal_features
        if self.gbm_thermal_grayscale_context:
            thermal_context_features = self._gbm_luma_replicated_features(thermal_features)
        gbm_anchor_context = self._get_gbm_anchor_context()

        gbm_outputs = self.gbm(
            anchor_state=self._get_anchor_summary(),
            rgb_features=rgb_features,
            thermal_features=thermal_features,
            thermal_context_features=thermal_context_features,
            anchor_context=gbm_anchor_context,
        )

        if self.gbm_rgb_luma_transfer_only:
            gbm_outputs["gate_th2rgb"] = self._gbm_luma_replicated_features(gbm_outputs["gate_th2rgb"])
            gbm_outputs["delta_th2rgb"] = self._gbm_luma_replicated_features(gbm_outputs["delta_th2rgb"])
            gbm_outputs["updated_rgb_features"] = (
                rgb_features + gbm_outputs["gate_th2rgb"] * gbm_outputs["delta_th2rgb"]
            )

        gate_th2rgb_anchor = self._gbm_anchorwise_mean(gbm_outputs["gate_th2rgb"])
        gate_rgb2th_anchor = self._gbm_anchorwise_mean(gbm_outputs["gate_rgb2th"])
        delta_th2rgb_mag = self._gbm_anchorwise_norm(gbm_outputs["delta_th2rgb"])
        delta_rgb2th_mag = self._gbm_anchorwise_norm(gbm_outputs["delta_rgb2th"])

        gate_th2rgb_mean, gate_th2rgb_std, gate_th2rgb_max = self._gbm_summary_stats(gate_th2rgb_anchor)
        gate_rgb2th_mean, gate_rgb2th_std, gate_rgb2th_max = self._gbm_summary_stats(gate_rgb2th_anchor)

        gbm_outputs["stability_reg"] = (
            self._gbm_chunked_gated_l1(gbm_outputs["gate_th2rgb"], gbm_outputs["delta_th2rgb"])
            + self._gbm_chunked_gated_l1(gbm_outputs["gate_rgb2th"], gbm_outputs["delta_rgb2th"])
        )
        gbm_outputs["gate_sparsity_reg"] = gbm_outputs["gate_th2rgb"].mean() + gbm_outputs["gate_rgb2th"].mean()
        gbm_outputs["gate_collapse_reg"] = (
            torch.relu(gate_th2rgb_std.new_tensor(self.gbm_gate_target_std) - gate_th2rgb_std)
            + torch.relu(gate_rgb2th_std.new_tensor(self.gbm_gate_target_std) - gate_rgb2th_std)
        )
        gbm_outputs["gate_overlap_reg"] = (gate_th2rgb_anchor * gate_rgb2th_anchor).mean()
        gbm_outputs["gate_th2rgb_anchor"] = gate_th2rgb_anchor
        gbm_outputs["gate_rgb2th_anchor"] = gate_rgb2th_anchor
        gbm_outputs["gate_th2rgb_mean"] = gate_th2rgb_mean
        gbm_outputs["gate_th2rgb_std"] = gate_th2rgb_std
        gbm_outputs["gate_th2rgb_max"] = gate_th2rgb_max
        gbm_outputs["gate_rgb2th_mean"] = gate_rgb2th_mean
        gbm_outputs["gate_rgb2th_std"] = gate_rgb2th_std
        gbm_outputs["gate_rgb2th_max"] = gate_rgb2th_max
        gbm_outputs["delta_th2rgb_mag_mean"] = delta_th2rgb_mag.mean()
        gbm_outputs["delta_rgb2th_mag_mean"] = delta_rgb2th_mag.mean()
        return gbm_outputs

    def get_rgb_render_params(self, scaling_modifier=1.0, feature_bindings=None):
        if feature_bindings is None:
            feature_bindings = self.get_gbm_feature_bindings()
        return self._build_render_params(
            means3D=self.get_xyz,
            scaling=self.get_scaling,
            features=feature_bindings["updated_rgb_features"],
            opacity=self.get_rgb_opacity,
            scaling_modifier=scaling_modifier,
        )

    def get_thermal_render_params(self, scaling_modifier=1.0, feature_bindings=None):
        if feature_bindings is None:
            feature_bindings = self.get_gbm_feature_bindings()
        return self._build_render_params(
            means3D=self.get_thermal_xyz,
            scaling=self.get_thermal_scaling,
            features=feature_bindings["updated_thermal_features"],
            opacity=self.get_thermal_opacity,
            scaling_modifier=scaling_modifier,
        )

    def get_thermal_residual_l1(self):
        if not self.use_thermal_residual_geometry:
            return self.get_xyz.new_zeros(())
        return self._delta_xyz_th.abs().mean() + self._delta_scaling_th.abs().mean()

    def get_anchor_thermal_geometry_usage(self):
        if not self.use_thermal_residual_geometry or self.get_xyz.shape[0] == 0:
            return self.get_xyz.new_zeros((self.get_xyz.shape[0],))
        return self._delta_xyz_th.norm(dim=1) + self._delta_scaling_th.norm(dim=1)

    def _get_gbm_anchor_context(self):
        num_anchors = self.get_xyz.shape[0]
        device = self._anchor_stats_device()
        context_dim = self._gbm_anchor_context_dim()
        if num_anchors == 0:
            return torch.zeros((0, context_dim), device=device)

        self._ensure_anchor_multimodal_stats(num_anchors)
        visibility_rgb = torch.clamp(self._anchor_stat_vector("visibility_rgb"), 0.0, 1.0)
        visibility_th = torch.clamp(self._anchor_stat_vector("visibility_th"), 0.0, 1.0)
        contribution_rgb = self._normalize_anchor_stat(self._anchor_stat_vector("contribution_rgb"))
        contribution_th = self._normalize_anchor_stat(self._anchor_stat_vector("contribution_th"))
        return torch.stack(
            (visibility_rgb, visibility_th, contribution_rgb, contribution_th),
            dim=1,
        )

    def get_anchor_multimodal_stats(self):
        self._ensure_anchor_multimodal_stats()
        return {
            stat_name: stat_tensor.detach().clone()
            for stat_name, stat_tensor in self.anchor_multimodal_stats.items()
        }

    def get_anchor_multimodal_stats_summary(self):
        self._ensure_anchor_multimodal_stats()
        summary = {}
        for stat_name in self._anchor_stat_names():
            stat_tensor = self.anchor_multimodal_stats[stat_name].detach().flatten()
            if stat_tensor.numel() == 0:
                summary[f"{stat_name}_mean"] = 0.0
                summary[f"{stat_name}_std"] = 0.0
            else:
                summary[f"{stat_name}_mean"] = stat_tensor.mean().item()
                summary[f"{stat_name}_std"] = stat_tensor.std(unbiased=False).item()
        return summary

    def update_anchor_multimodal_stats(
        self,
        rgb_visibility_filter,
        thermal_visibility_filter,
        rgb_contribution_proxy,
        thermal_contribution_proxy,
        rgb_residual_proxy,
        thermal_residual_proxy,
        gbm_usage_th2rgb=None,
        gbm_usage_rgb2th=None,
        thermal_geometry_usage=None,
        ema=None,
    ):
        self._ensure_anchor_multimodal_stats()
        ema = self.anchor_stats_ema if ema is None else ema
        ema = max(0.0, min(float(ema), 0.9999))
        num_anchors = self.get_xyz.shape[0]

        rgb_visibility_filter = rgb_visibility_filter.detach().reshape(-1).bool()
        thermal_visibility_filter = thermal_visibility_filter.detach().reshape(-1).bool()
        if rgb_visibility_filter.numel() != num_anchors or thermal_visibility_filter.numel() != num_anchors:
            raise ValueError("Anchor stats update expects per-anchor rgb/thermal visibility masks.")

        self._ema_update_anchor_stat(
            "visibility_rgb",
            self._prepare_anchor_stat_observation(rgb_visibility_filter.float(), num_anchors),
            ema,
        )
        self._ema_update_anchor_stat(
            "visibility_th",
            self._prepare_anchor_stat_observation(thermal_visibility_filter.float(), num_anchors),
            ema,
        )
        self._ema_update_anchor_stat(
            "contribution_rgb",
            self._prepare_anchor_stat_observation(
                rgb_contribution_proxy,
                num_anchors,
                visibility_mask=rgb_visibility_filter,
            ),
            ema,
        )
        self._ema_update_anchor_stat(
            "contribution_th",
            self._prepare_anchor_stat_observation(
                thermal_contribution_proxy,
                num_anchors,
                visibility_mask=thermal_visibility_filter,
            ),
            ema,
        )
        self._ema_update_anchor_stat(
            "residual_rgb",
            self._prepare_anchor_stat_observation(
                rgb_residual_proxy,
                num_anchors,
                visibility_mask=rgb_visibility_filter,
            ),
            ema,
        )
        self._ema_update_anchor_stat(
            "residual_th",
            self._prepare_anchor_stat_observation(
                thermal_residual_proxy,
                num_anchors,
                visibility_mask=thermal_visibility_filter,
            ),
            ema,
        )
        self._ema_update_anchor_stat(
            "gbm_usage_th2rgb",
            self._prepare_anchor_stat_observation(
                gbm_usage_th2rgb,
                num_anchors,
                visibility_mask=rgb_visibility_filter,
            ),
            ema,
        )
        self._ema_update_anchor_stat(
            "gbm_usage_rgb2th",
            self._prepare_anchor_stat_observation(
                gbm_usage_rgb2th,
                num_anchors,
                visibility_mask=thermal_visibility_filter,
            ),
            ema,
        )
        self._ema_update_anchor_stat(
            "thermal_geometry_usage",
            self._prepare_anchor_stat_observation(
                thermal_geometry_usage,
                num_anchors,
                visibility_mask=thermal_visibility_filter,
            ),
            ema,
        )
        return self.get_anchor_multimodal_stats_summary()

    def _joint_lifecycle_ready(self, iteration=None):
        if not self.joint_lifecycle_enabled:
            return False
        return iteration is None or iteration >= self.joint_lifecycle_warmup_iters

    def _joint_budget_state(self, iteration=None):
        current_point_count = int(self.get_xyz.shape[0])
        if current_point_count > 0 and self.joint_lifecycle_start_point_count <= 0:
            self.joint_lifecycle_start_point_count = current_point_count

        if self._joint_lifecycle_ready(iteration) and self.joint_lifecycle_budget_reference_count <= 0 and current_point_count > 0:
            self.joint_lifecycle_budget_reference_count = current_point_count

        reference_point_count = self.joint_lifecycle_budget_reference_count
        if reference_point_count <= 0:
            reference_point_count = self.joint_lifecycle_start_point_count
        if reference_point_count <= 0:
            reference_point_count = max(1, current_point_count)

        reference_point_count = int(reference_point_count)
        growth_ratio = (
            current_point_count / float(reference_point_count)
            if current_point_count > 0
            else 0.0
        )
        max_points = max(reference_point_count, int(np.ceil(reference_point_count * self.joint_lifecycle_max_point_ratio)))
        remaining_point_budget = max(0, max_points - current_point_count)
        return {
            "current_point_count": current_point_count,
            "reference_point_count": reference_point_count,
            "point_growth_ratio": growth_ratio,
            "max_points": max_points,
            "remaining_point_budget": remaining_point_budget,
            "budget_exceeded": current_point_count >= max_points,
        }

    def _joint_lifecycle_zero_scores(self, iteration=None):
        num_anchors = self.get_xyz.shape[0]
        device = self._anchor_stats_device()
        zero_scores = torch.zeros((num_anchors,), device=device)
        false_mask = torch.zeros((num_anchors,), dtype=torch.bool, device=device)
        true_mask = torch.ones((num_anchors,), dtype=torch.bool, device=device)
        budget_state = self._joint_budget_state(iteration)
        diagnostics = {
            "joint_lifecycle_enabled": 0.0,
            "joint_split_score_mean": 0.0,
            "joint_split_score_std": 0.0,
            "joint_split_score_max": 0.0,
            "current_point_count": float(budget_state["current_point_count"]),
            "point_growth_ratio_vs_baseline_or_start": budget_state["point_growth_ratio"],
            "joint_split_trigger_count": 0.0,
            "split_suppressed_by_budget_count": 0.0,
            "joint_prune_candidate_ratio": 0.0,
            "joint_prune_veto_ratio": 0.0,
            "prune_by_both_modalities_count": 0.0,
            "split_triggered_by_rgb_count": 0.0,
            "split_triggered_by_th_count": 0.0,
            "split_boosted_by_gbm_count": 0.0,
            "split_boosted_by_thgeo_count": 0.0,
        }
        self.last_joint_lifecycle_diagnostics = diagnostics
        return {
            "joint_split_score": zero_scores,
            "joint_prune_mask": true_mask,
            "joint_prune_candidate_mask": false_mask,
            "joint_prune_veto_mask": false_mask,
            "split_rgb_score": zero_scores,
            "split_th_score": zero_scores,
            "gbm_boost": zero_scores,
            "thgeo_boost": zero_scores,
            "diagnostics": diagnostics,
        }

    def _anchor_stat_vector(self, stat_name):
        num_anchors = self.get_xyz.shape[0]
        device = self._anchor_stats_device()
        stat_tensor = self.anchor_multimodal_stats.get(stat_name)
        if stat_tensor is None:
            return torch.zeros((num_anchors,), device=device)
        return torch.nan_to_num(
            torch.as_tensor(stat_tensor, device=device).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _normalize_anchor_stat(self, values, clamp_max=2.0):
        values = torch.nan_to_num(values.detach(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        if values.numel() == 0:
            return values
        positive_values = values[values > 0]
        if positive_values.numel() == 0:
            return torch.zeros_like(values)
        scale = positive_values.mean()
        if positive_values.numel() > 1:
            scale = scale + positive_values.std(unbiased=False)
        scale = scale.clamp_min(1e-6)
        return torch.clamp(values / scale, 0.0, clamp_max)

    def _pad_per_anchor_signal(self, values, fill_value=0.0):
        num_anchors = self.get_xyz.shape[0]
        device = self._anchor_stats_device()
        if values is None:
            return torch.full((num_anchors,), fill_value, device=device)
        values = torch.nan_to_num(values.detach().reshape(-1), nan=fill_value, posinf=fill_value, neginf=fill_value)
        if values.shape[0] == num_anchors:
            return values.to(device)
        padded = torch.full((num_anchors,), fill_value, device=device, dtype=values.dtype)
        overlap = min(num_anchors, values.shape[0])
        if overlap > 0:
            padded[:overlap] = values[:overlap].to(device)
        return padded

    def get_joint_lifecycle_scores(self, iteration=None):
        num_anchors = self.get_xyz.shape[0]
        if num_anchors == 0 or (not self._joint_lifecycle_ready(iteration)):
            return self._joint_lifecycle_zero_scores(iteration=iteration)

        self._ensure_anchor_multimodal_stats(num_anchors)
        budget_state = self._joint_budget_state(iteration)

        visibility_rgb = torch.clamp(self._anchor_stat_vector("visibility_rgb"), 0.0, 1.0)
        visibility_th = torch.clamp(self._anchor_stat_vector("visibility_th"), 0.0, 1.0)
        contribution_rgb = self._normalize_anchor_stat(self._anchor_stat_vector("contribution_rgb"))
        contribution_th = self._normalize_anchor_stat(self._anchor_stat_vector("contribution_th"))
        residual_rgb = self._normalize_anchor_stat(self._anchor_stat_vector("residual_rgb"))
        residual_th = self._normalize_anchor_stat(self._anchor_stat_vector("residual_th"))
        gbm_usage_th2rgb = self._normalize_anchor_stat(self._anchor_stat_vector("gbm_usage_th2rgb"))
        gbm_usage_rgb2th = self._normalize_anchor_stat(self._anchor_stat_vector("gbm_usage_rgb2th"))
        thermal_geometry_usage = self._normalize_anchor_stat(self._anchor_stat_vector("thermal_geometry_usage"))

        split_rgb = self.joint_split_rgb_weight * visibility_rgb * contribution_rgb * (1.0 + 0.5 * residual_rgb)
        split_th = self.joint_split_th_weight * visibility_th * contribution_th * (1.0 + 0.5 * residual_th)
        base_split = torch.maximum(split_rgb, split_th)

        gbm_usage_proxy = torch.maximum(gbm_usage_th2rgb, gbm_usage_rgb2th)
        gbm_boost = self.joint_split_gbm_boost * gbm_usage_proxy
        thgeo_boost = self.joint_split_thgeo_boost * thermal_geometry_usage
        joint_split_score = base_split + gbm_boost + thgeo_boost

        low_rgb = (
            (visibility_rgb < self.joint_prune_visibility_thresh)
            & (contribution_rgb < self.joint_prune_contribution_thresh)
            & (residual_rgb < self.joint_prune_residual_thresh)
        )
        low_th = (
            (visibility_th < self.joint_prune_visibility_thresh)
            & (contribution_th < self.joint_prune_contribution_thresh)
            & (residual_th < self.joint_prune_residual_thresh)
        )
        joint_prune_candidate_mask = low_rgb & low_th
        joint_prune_veto_mask = (
            (gbm_usage_proxy > self.joint_prune_gbm_veto_thresh)
            | (thermal_geometry_usage > self.joint_prune_thgeo_veto_thresh)
        )
        joint_prune_mask = joint_prune_candidate_mask & (~joint_prune_veto_mask)

        split_threshold = self.joint_split_score_threshold
        base_trigger = base_split >= split_threshold
        gbm_trigger = (base_split + gbm_boost) >= split_threshold
        final_trigger = joint_split_score >= split_threshold

        split_triggered_by_rgb = (split_rgb >= split_threshold) & (split_rgb >= split_th)
        split_triggered_by_th = (split_th >= split_threshold) & (split_th > split_rgb)
        split_boosted_by_gbm = (~base_trigger) & gbm_trigger & (gbm_boost > 0)
        split_boosted_by_thgeo = (~gbm_trigger) & final_trigger & (thgeo_boost > 0)

        candidate_count = int(joint_prune_candidate_mask.sum().item())
        veto_ratio = (
            joint_prune_veto_mask[joint_prune_candidate_mask].float().mean().item()
            if candidate_count > 0
            else 0.0
        )
        diagnostics = {
            "joint_lifecycle_enabled": 1.0,
            "joint_split_score_mean": joint_split_score.mean().item(),
            "joint_split_score_std": joint_split_score.std(unbiased=False).item(),
            "joint_split_score_max": joint_split_score.max().item(),
            "current_point_count": float(budget_state["current_point_count"]),
            "point_growth_ratio_vs_baseline_or_start": budget_state["point_growth_ratio"],
            "joint_split_trigger_count": self.last_joint_lifecycle_selection_diagnostics["joint_split_trigger_count"],
            "split_suppressed_by_budget_count": self.last_joint_lifecycle_selection_diagnostics[
                "split_suppressed_by_budget_count"
            ],
            "joint_prune_candidate_ratio": joint_prune_candidate_mask.float().mean().item(),
            "joint_prune_veto_ratio": veto_ratio,
            "prune_by_both_modalities_count": float(candidate_count),
            "split_triggered_by_rgb_count": float(split_triggered_by_rgb.sum().item()),
            "split_triggered_by_th_count": float(split_triggered_by_th.sum().item()),
            "split_boosted_by_gbm_count": float(split_boosted_by_gbm.sum().item()),
            "split_boosted_by_thgeo_count": float(split_boosted_by_thgeo.sum().item()),
        }
        self.last_joint_lifecycle_diagnostics = diagnostics
        return {
            "joint_split_score": joint_split_score,
            "joint_prune_mask": joint_prune_mask,
            "joint_prune_candidate_mask": joint_prune_candidate_mask,
            "joint_prune_veto_mask": joint_prune_veto_mask,
            "split_rgb_score": split_rgb,
            "split_th_score": split_th,
            "gbm_boost": gbm_boost,
            "thgeo_boost": thgeo_boost,
            "diagnostics": diagnostics,
        }

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        thermal_features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()  
        thermal_features[:, :, :] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._thermal_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._thermal_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity_base = nn.Parameter(opacities.requires_grad_(True))
        self._opacity_bias_rgb = nn.Parameter(torch.zeros_like(opacities).requires_grad_(True))
        self._opacity_bias_th = nn.Parameter(torch.zeros_like(opacities).requires_grad_(True))
        self._delta_xyz_th = nn.Parameter(torch.zeros_like(fused_point_cloud).requires_grad_(True))
        self._delta_scaling_th = nn.Parameter(torch.zeros_like(scales).requires_grad_(True))
        self._refresh_optional_parameter_grad_flags()
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.anchor_lifecycle = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._initialize_anchor_multimodal_stats(self.get_xyz.shape[0])

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.anchor_lifecycle = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.gbm_gate_target_std = getattr(training_args, "gbm_gate_target_std", self.gbm_gate_target_std)
        self.anchor_stats_ema = getattr(training_args, "anchor_stats_ema", self.anchor_stats_ema)
        self.save_anchor_stats_enabled = getattr(training_args, "save_anchor_stats", self.save_anchor_stats_enabled)
        self.joint_lifecycle_enabled = getattr(training_args, "joint_lifecycle", self.joint_lifecycle_enabled)
        self.joint_lifecycle_warmup_iters = max(
            getattr(training_args, "anchor_stats_warmup_iters", 0),
            getattr(training_args, "joint_lifecycle_warmup_iters", self.joint_lifecycle_warmup_iters),
        )
        self.joint_split_rgb_weight = getattr(training_args, "joint_split_rgb_weight", self.joint_split_rgb_weight)
        self.joint_split_th_weight = getattr(training_args, "joint_split_th_weight", self.joint_split_th_weight)
        self.joint_split_gbm_boost = getattr(training_args, "joint_split_gbm_boost", self.joint_split_gbm_boost)
        self.joint_split_thgeo_boost = getattr(
            training_args, "joint_split_thgeo_boost", self.joint_split_thgeo_boost
        )
        self.joint_split_score_threshold = getattr(
            training_args, "joint_split_score_threshold", self.joint_split_score_threshold
        )
        self.joint_split_max_extra_ratio = max(
            0.0,
            getattr(training_args, "joint_split_max_extra_ratio", self.joint_split_max_extra_ratio),
        )
        self.joint_lifecycle_max_point_ratio = max(
            1.0,
            getattr(training_args, "joint_lifecycle_max_point_ratio", self.joint_lifecycle_max_point_ratio),
        )
        self.joint_prune_visibility_thresh = getattr(
            training_args, "joint_prune_visibility_thresh", self.joint_prune_visibility_thresh
        )
        self.joint_prune_contribution_thresh = getattr(
            training_args, "joint_prune_contribution_thresh", self.joint_prune_contribution_thresh
        )
        self.joint_prune_residual_thresh = getattr(
            training_args, "joint_prune_residual_thresh", self.joint_prune_residual_thresh
        )
        self.joint_prune_gbm_veto_thresh = getattr(
            training_args, "joint_prune_gbm_veto_thresh", self.joint_prune_gbm_veto_thresh
        )
        self.joint_prune_thgeo_veto_thresh = getattr(
            training_args, "joint_prune_thgeo_veto_thresh", self.joint_prune_thgeo_veto_thresh
        )
        self.joint_lifecycle_start_point_count = int(self.get_xyz.shape[0]) if self.get_xyz.shape[0] > 0 else 0
        self.joint_lifecycle_budget_reference_count = 0
        self.last_joint_lifecycle_selection_diagnostics = {
            "joint_split_trigger_count": 0.0,
            "split_suppressed_by_budget_count": 0.0,
        }
        self._configure_gbm_module()
        self._refresh_optional_parameter_grad_flags()
        self.thermal_residual_xyz_scheduler_args = None
        self._ensure_anchor_multimodal_stats(self.get_xyz.shape[0])

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz", "per_anchor": True},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc", "per_anchor": True},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest", "per_anchor": True},
            {'params': [self._thermal_dc], 'lr': training_args.thermal_feature_lr, "name": "thermal_dc", "per_anchor": True},
            {'params': [self._thermal_rest], 'lr': training_args.thermal_feature_lr / 20.0, "name": "t_rest", "per_anchor": True},
            {'params': [self._opacity_base], 'lr': training_args.opacity_lr, "name": "opacity_base", "per_anchor": True},
            {'params': [self._opacity_bias_rgb], 'lr': training_args.opacity_lr, "name": "opacity_bias_rgb", "per_anchor": True},
            {'params': [self._opacity_bias_th], 'lr': training_args.opacity_lr, "name": "opacity_bias_th", "per_anchor": True},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling", "per_anchor": True},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation", "per_anchor": True},
        ]
        if self.use_gbm:
            gbm_lr_scale = getattr(training_args, "gbm_lr_scale", 0.1)
            l.append(
                {
                    'params': list(self.gbm.parameters()),
                    'lr': training_args.feature_lr * gbm_lr_scale,
                    "name": "gbm",
                    "per_anchor": False,
                }
            )
        if self.use_thermal_residual_geometry:
            thermal_residual_lr_scale = getattr(training_args, "thermal_residual_lr_scale", 0.1)
            delta_xyz_lr_init = training_args.position_lr_init * self.spatial_lr_scale * thermal_residual_lr_scale
            delta_xyz_lr_final = training_args.position_lr_final * self.spatial_lr_scale * thermal_residual_lr_scale
            l.extend([
                {'params': [self._delta_xyz_th], 'lr': delta_xyz_lr_init, "name": "delta_xyz_th", "per_anchor": True},
                {'params': [self._delta_scaling_th], 'lr': training_args.scaling_lr * thermal_residual_lr_scale, "name": "delta_scaling_th", "per_anchor": True},
            ])
            self.thermal_residual_xyz_scheduler_args = get_expon_lr_func(
                lr_init=delta_xyz_lr_init,
                lr_final=delta_xyz_lr_final,
                lr_delay_mult=training_args.position_lr_delay_mult,
                max_steps=training_args.position_lr_max_steps,
            )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        xyz_lr = self.xyz_scheduler_args(iteration)
        thermal_residual_xyz_lr = (
            self.thermal_residual_xyz_scheduler_args(iteration)
            if self.thermal_residual_xyz_scheduler_args is not None
            else None
        )
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = xyz_lr
            elif param_group["name"] == "delta_xyz_th" and thermal_residual_xyz_lr is not None:
                param_group['lr'] = thermal_residual_xyz_lr
        return xyz_lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._thermal_dc.shape[1]*self._thermal_dc.shape[2]):
            l.append('t_dc_{}'.format(i))
        for i in range(self._thermal_rest.shape[1]*self._thermal_rest.shape[2]):
            l.append('t_rest_{}'.format(i))
        l.append('opacity')
        l.append('opacity_base')
        l.append('opacity_bias_rgb')
        l.append('opacity_bias_th')
        for i in range(self._delta_xyz_th.shape[1]):
            l.append('delta_xyz_th_{}'.format(i))
        for i in range(self._delta_scaling_th.shape[1]):
            l.append('delta_scaling_th_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        thermal_dc = self._thermal_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        t_rest = self._thermal_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        compatibility_opacity = self._opacity_base.detach().cpu().numpy()
        opacity_base = self._opacity_base.detach().cpu().numpy()
        opacity_bias_rgb = self._opacity_bias_rgb.detach().cpu().numpy()
        opacity_bias_th = self._opacity_bias_th.detach().cpu().numpy()
        delta_xyz_th = self._delta_xyz_th.detach().cpu().numpy()
        delta_scaling_th = self._delta_scaling_th.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (
                xyz,
                normals,
                f_dc,
                f_rest,
                thermal_dc,
                t_rest,
                compatibility_opacity,
                opacity_base,
                opacity_bias_rgb,
                opacity_bias_th,
                delta_xyz_th,
                delta_scaling_th,
                scale,
                rotation,
            ),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def reset_opacity(self):
        base_opacity_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(base_opacity_new, "opacity_base")
        self._opacity_base = optimizable_tensors["opacity_base"]
        bias_rgb_new = torch.zeros_like(self._opacity_bias_rgb)
        optimizable_tensors = self.replace_tensor_to_optimizer(bias_rgb_new, "opacity_bias_rgb")
        if "opacity_bias_rgb" in optimizable_tensors:
            self._opacity_bias_rgb = optimizable_tensors["opacity_bias_rgb"]
        bias_th_new = torch.zeros_like(self._opacity_bias_th)
        optimizable_tensors = self.replace_tensor_to_optimizer(bias_th_new, "opacity_bias_th")
        if "opacity_bias_th" in optimizable_tensors:
            self._opacity_bias_th = optimizable_tensors["opacity_bias_th"]
        

    def load_ply(self, path):
        plydata = PlyData.read(path)
        property_names = {p.name for p in plydata.elements[0].properties}

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacity_base = (
            np.asarray(plydata.elements[0]["opacity_base"])[..., np.newaxis]
            if "opacity_base" in property_names
            else np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        )
        opacity_bias_rgb = (
            np.asarray(plydata.elements[0]["opacity_bias_rgb"])[..., np.newaxis]
            if "opacity_bias_rgb" in property_names
            else np.zeros((xyz.shape[0], 1))
        )
        opacity_bias_th = (
            np.asarray(plydata.elements[0]["opacity_bias_th"])[..., np.newaxis]
            if "opacity_bias_th" in property_names
            else np.zeros((xyz.shape[0], 1))
        )


        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        thermal_dc = np.zeros((xyz.shape[0], 3, 1))
        thermal_dc[:, 0, 0] = np.asarray(plydata.elements[0]["t_dc_0"])
        thermal_dc[:, 1, 0] = np.asarray(plydata.elements[0]["t_dc_1"])
        thermal_dc[:, 2, 0] = np.asarray(plydata.elements[0]["t_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

        extra_t_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("t_rest_")]
        extra_t_names = sorted(extra_t_names, key = lambda x: int(x.split('_')[-1]))

        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))

        assert len(extra_t_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        thermal_extra = np.zeros((xyz.shape[0], len(extra_t_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        for idx, attr_name in enumerate(extra_t_names):
            thermal_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        thermal_extra = thermal_extra.reshape((thermal_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))           ##

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        delta_xyz_th = np.zeros((xyz.shape[0], 3))
        for idx in range(delta_xyz_th.shape[1]):
            attr_name = "delta_xyz_th_{}".format(idx)
            if attr_name in property_names:
                delta_xyz_th[:, idx] = np.asarray(plydata.elements[0][attr_name])

        delta_scaling_th = np.zeros((xyz.shape[0], 3))
        for idx in range(delta_scaling_th.shape[1]):
            attr_name = "delta_scaling_th_{}".format(idx)
            if attr_name in property_names:
                delta_scaling_th[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._thermal_dc = nn.Parameter(torch.tensor(thermal_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._thermal_rest = nn.Parameter(torch.tensor(thermal_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity_base = nn.Parameter(torch.tensor(opacity_base, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity_bias_rgb = nn.Parameter(torch.tensor(opacity_bias_rgb, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity_bias_th = nn.Parameter(torch.tensor(opacity_bias_th, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._delta_xyz_th = nn.Parameter(torch.tensor(delta_xyz_th, dtype=torch.float, device="cuda").requires_grad_(True))
        self._delta_scaling_th = nn.Parameter(torch.tensor(delta_scaling_th, dtype=torch.float, device="cuda").requires_grad_(True))
        self._refresh_optional_parameter_grad_flags()
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.anchor_lifecycle = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._initialize_anchor_multimodal_stats(self.get_xyz.shape[0])

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                requires_grad = group["params"][0].requires_grad
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.detach().clone(), requires_grad=requires_grad)
                    self.optimizer.state[group['params'][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(tensor.detach().clone(), requires_grad=requires_grad)

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if not group.get("per_anchor", True):
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            requires_grad = group["params"][0].requires_grad
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].detach().clone(), requires_grad=requires_grad)
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].detach().clone(), requires_grad=requires_grad)
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._thermal_dc = optimizable_tensors["thermal_dc"]
        self._thermal_rest = optimizable_tensors["t_rest"]
        self._opacity_base = optimizable_tensors["opacity_base"]
        self._opacity_bias_rgb = optimizable_tensors.get("opacity_bias_rgb", self._masked_parameter(self._opacity_bias_rgb, valid_points_mask))
        self._opacity_bias_th = optimizable_tensors.get("opacity_bias_th", self._masked_parameter(self._opacity_bias_th, valid_points_mask))
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._delta_xyz_th = optimizable_tensors.get("delta_xyz_th", self._masked_parameter(self._delta_xyz_th, valid_points_mask))
        self._delta_scaling_th = optimizable_tensors.get("delta_scaling_th", self._masked_parameter(self._delta_scaling_th, valid_points_mask))

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.anchor_lifecycle = self.anchor_lifecycle[valid_points_mask]
        self._prune_anchor_multimodal_stats(valid_points_mask)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if not group.get("per_anchor", True):
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            requires_grad = group["params"][0].requires_grad
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).detach().clone(),
                    requires_grad=requires_grad,
                )
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).detach().clone(),
                    requires_grad=requires_grad,
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_thermal_dc,
        new_thermal_rest,
        new_opacity_base,
        new_opacity_bias_rgb,
        new_opacity_bias_th,
        new_scaling,
        new_rotation,
        new_delta_xyz_th,
        new_delta_scaling_th,
        new_anchor_multimodal_stats=None,
    ):
        previous_num_anchors = self.get_xyz.shape[0]
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "thermal_dc": new_thermal_dc,
        "t_rest": new_thermal_rest,
        "opacity_base": new_opacity_base,
        "opacity_bias_rgb": new_opacity_bias_rgb,
        "opacity_bias_th": new_opacity_bias_th,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "delta_xyz_th": new_delta_xyz_th,
        "delta_scaling_th": new_delta_scaling_th}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._thermal_dc = optimizable_tensors["thermal_dc"]
        self._thermal_rest = optimizable_tensors["t_rest"]
        self._opacity_base = optimizable_tensors["opacity_base"]
        self._opacity_bias_rgb = optimizable_tensors.get(
            "opacity_bias_rgb", self._concatenated_parameter(self._opacity_bias_rgb, new_opacity_bias_rgb)
        )
        self._opacity_bias_th = optimizable_tensors.get(
            "opacity_bias_th", self._concatenated_parameter(self._opacity_bias_th, new_opacity_bias_th)
        )
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._delta_xyz_th = optimizable_tensors.get(
            "delta_xyz_th", self._concatenated_parameter(self._delta_xyz_th, new_delta_xyz_th)
        )
        self._delta_scaling_th = optimizable_tensors.get(
            "delta_scaling_th", self._concatenated_parameter(self._delta_scaling_th, new_delta_scaling_th)
        )

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.anchor_lifecycle = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._extend_anchor_multimodal_stats(
            new_anchor_multimodal_stats,
            new_xyz.shape[0],
            previous_num_anchors=previous_num_anchors,
        )

    def _baseline_densify_selection_mask(self, grads, grad_threshold, scene_extent, prefer_large_scales):
        n_init_points = self.get_xyz.shape[0]
        grad_values = grads.detach().reshape(-1)
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grad_values.shape[0]] = grad_values
        selected_pts_mask = padded_grad >= grad_threshold

        scale_mask = torch.max(self.get_scaling, dim=1).values
        if prefer_large_scales:
            scale_mask = scale_mask > self.percent_dense * scene_extent
        else:
            scale_mask = scale_mask <= self.percent_dense * scene_extent
        return selected_pts_mask & scale_mask

    def _select_topk_mask(self, candidate_mask, scores, k):
        if k <= 0:
            return torch.zeros_like(candidate_mask)

        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
        if candidate_indices.numel() == 0:
            return torch.zeros_like(candidate_mask)
        if candidate_indices.numel() <= k:
            return candidate_mask

        candidate_scores = scores[candidate_indices]
        topk_indices = torch.topk(candidate_scores, k=k, largest=True, sorted=False).indices
        selected_mask = torch.zeros_like(candidate_mask)
        selected_mask[candidate_indices[topk_indices]] = True
        return selected_mask

    def _joint_densify_selection_mask(
        self,
        grads,
        grad_threshold,
        scene_extent,
        prefer_large_scales,
        joint_split_score=None,
        iteration=None,
    ):
        baseline_mask = self._baseline_densify_selection_mask(
            grads=grads,
            grad_threshold=grad_threshold,
            scene_extent=scene_extent,
            prefer_large_scales=prefer_large_scales,
        )
        selection_diagnostics = {
            "joint_split_trigger_count": 0.0,
            "split_suppressed_by_budget_count": 0.0,
        }

        if (
            (not prefer_large_scales)
            or joint_split_score is None
            or (not self._joint_lifecycle_ready(iteration))
            or self.joint_split_max_extra_ratio <= 0.0
        ):
            return baseline_mask, selection_diagnostics

        baseline_count = int(baseline_mask.sum().item())
        if baseline_count <= 0:
            return baseline_mask, selection_diagnostics

        padded_joint_score = self._pad_per_anchor_signal(joint_split_score, fill_value=0.0)
        scale_mask = torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        joint_extra_candidates = (~baseline_mask) & scale_mask & (padded_joint_score >= self.joint_split_score_threshold)
        joint_extra_candidate_count = int(joint_extra_candidates.sum().item())
        if joint_extra_candidate_count <= 0:
            return baseline_mask, selection_diagnostics

        max_extra_candidates = max(1, int(np.ceil(baseline_count * self.joint_split_max_extra_ratio)))
        capped_extra_candidates = min(joint_extra_candidate_count, max_extra_candidates)

        budget_state = self._joint_budget_state(iteration)
        allowed_extra_candidates = min(capped_extra_candidates, budget_state["remaining_point_budget"])
        selection_diagnostics["split_suppressed_by_budget_count"] = float(
            max(0, capped_extra_candidates - allowed_extra_candidates)
        )
        if allowed_extra_candidates <= 0:
            return baseline_mask, selection_diagnostics

        selected_joint_extras = self._select_topk_mask(
            joint_extra_candidates,
            padded_joint_score,
            allowed_extra_candidates,
        )
        selection_diagnostics["joint_split_trigger_count"] = float(selected_joint_extras.sum().item())
        return baseline_mask | selected_joint_extras, selection_diagnostics

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, joint_split_score=None, iteration=None):
        selected_pts_mask, selection_diagnostics = self._joint_densify_selection_mask(
            grads=grads,
            grad_threshold=grad_threshold,
            scene_extent=scene_extent,
            prefer_large_scales=True,
            joint_split_score=joint_split_score,
            iteration=iteration,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_thermal_dc = self._thermal_dc[selected_pts_mask].repeat(N,1,1)
        new_thermal_rest = self._thermal_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity_base = self._opacity_base[selected_pts_mask].repeat(N,1)
        new_opacity_bias_rgb = self._opacity_bias_rgb[selected_pts_mask].repeat(N,1)
        new_opacity_bias_th = self._opacity_bias_th[selected_pts_mask].repeat(N,1)
        new_delta_xyz_th = self._delta_xyz_th[selected_pts_mask].repeat(N,1)
        new_delta_scaling_th = self._delta_scaling_th[selected_pts_mask].repeat(N,1)
        new_anchor_multimodal_stats = self._select_anchor_multimodal_stats(
            selected_pts_mask,
            repeat=N,
            inherit_scale=self.new_anchor_stats_inherit_scale,
        )

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_thermal_dc,
            new_thermal_rest,
            new_opacity_base,
            new_opacity_bias_rgb,
            new_opacity_bias_th,
            new_scaling,
            new_rotation,
            new_delta_xyz_th,
            new_delta_scaling_th,
            new_anchor_multimodal_stats=new_anchor_multimodal_stats,
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        return selection_diagnostics

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = self._baseline_densify_selection_mask(
            grads=grads,
            grad_threshold=grad_threshold,
            scene_extent=scene_extent,
            prefer_large_scales=False,
        )
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_thermal_dc = self._thermal_dc[selected_pts_mask]
        new_thermal_rest = self._thermal_rest[selected_pts_mask]
        new_opacity_base = self._opacity_base[selected_pts_mask]
        new_opacity_bias_rgb = self._opacity_bias_rgb[selected_pts_mask]
        new_opacity_bias_th = self._opacity_bias_th[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_delta_xyz_th = self._delta_xyz_th[selected_pts_mask]
        new_delta_scaling_th = self._delta_scaling_th[selected_pts_mask]
        new_anchor_multimodal_stats = self._select_anchor_multimodal_stats(
            selected_pts_mask,
            inherit_scale=self.new_anchor_stats_inherit_scale,
        )

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_thermal_dc,
            new_thermal_rest,
            new_opacity_base,
            new_opacity_bias_rgb,
            new_opacity_bias_th,
            new_scaling,
            new_rotation,
            new_delta_xyz_th,
            new_delta_scaling_th,
            new_anchor_multimodal_stats=new_anchor_multimodal_stats,
        )

    def _build_prune_mask(self, min_opacity, extent, max_screen_size, iteration=None):
        joint_post_scores = self.get_joint_lifecycle_scores(iteration=iteration)
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        prune_mask = prune_mask & joint_post_scores["joint_prune_mask"]
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        return prune_mask

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        joint_pre_scores = self.get_joint_lifecycle_scores(iteration=iteration)
        self.last_joint_lifecycle_selection_diagnostics = {
            "joint_split_trigger_count": 0.0,
            "split_suppressed_by_budget_count": 0.0,
        }
        self.densify_and_clone(grads, max_grad, extent)
        self.last_joint_lifecycle_selection_diagnostics = self.densify_and_split(
            grads,
            max_grad,
            extent,
            joint_split_score=joint_pre_scores["joint_split_score"],
            iteration=iteration,
        )

        prune_mask = self._build_prune_mask(
            min_opacity=min_opacity,
            extent=extent,
            max_screen_size=max_screen_size,
            iteration=iteration,
        )
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def late_prune_only(self, min_opacity, extent, max_screen_size, iteration=None):
        prune_mask = self._build_prune_mask(
            min_opacity=min_opacity,
            extent=extent,
            max_screen_size=max_screen_size,
            iteration=iteration,
        )
        removed_count = int(prune_mask.sum().item())
        self.prune_points(prune_mask)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        torch.cuda.empty_cache()
        return removed_count

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def save_feature_modules(self, path):
        if not self.use_gbm or self.gbm is None:
            return

        torch.save(
            {
                "use_gbm": self.use_gbm,
                "gbm_hidden_dim": self.gbm_hidden_dim,
                "gbm_gate_init_bias": self.gbm_gate_init_bias,
                "gbm_thermal_grayscale_context": self.gbm_thermal_grayscale_context,
                "gbm_rgb_luma_transfer_only": self.gbm_rgb_luma_transfer_only,
                "gbm_state": self.gbm.state_dict(),
            },
            os.path.join(path, "feature_modules.pth"),
        )

    def load_feature_modules(self, path):
        feature_module_path = os.path.join(path, "feature_modules.pth")
        if not self.use_gbm or not os.path.exists(feature_module_path):
            return

        module_state = torch.load(feature_module_path, map_location="cuda")
        self.gbm_hidden_dim = module_state.get("gbm_hidden_dim", self.gbm_hidden_dim)
        self.gbm_gate_init_bias = module_state.get("gbm_gate_init_bias", self.gbm_gate_init_bias)
        self.gbm_thermal_grayscale_context = module_state.get(
            "gbm_thermal_grayscale_context", self.gbm_thermal_grayscale_context
        )
        self.gbm_rgb_luma_transfer_only = module_state.get(
            "gbm_rgb_luma_transfer_only", self.gbm_rgb_luma_transfer_only
        )
        self._configure_gbm_module()
        gbm_state = module_state.get("gbm_state")
        self._load_gbm_state_compat(gbm_state)

    def save_anchor_stats(self, path):
        if not self.save_anchor_stats_enabled:
            return

        mkdir_p(path)
        torch.save(
            {
                "anchor_stats_ema": self.anchor_stats_ema,
                "anchor_multimodal_stats": {
                    stat_name: stat_tensor.detach().cpu()
                    for stat_name, stat_tensor in self.get_anchor_multimodal_stats().items()
                },
                "summary": self.get_anchor_multimodal_stats_summary(),
            },
            os.path.join(path, "anchor_stats.pth"),
        )

    def load_anchor_stats(self, path):
        anchor_stats_path = os.path.join(path, "anchor_stats.pth")
        if not os.path.exists(anchor_stats_path):
            self._ensure_anchor_multimodal_stats(self.get_xyz.shape[0])
            return

        stats_state = torch.load(anchor_stats_path, map_location="cuda")
        self.anchor_stats_ema = stats_state.get("anchor_stats_ema", self.anchor_stats_ema)
        self._restore_anchor_multimodal_stats(
            stats_state.get("anchor_multimodal_stats"),
            self.get_xyz.shape[0],
        )
