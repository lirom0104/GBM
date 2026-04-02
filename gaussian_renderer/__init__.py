#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def _resolve_head_appearance(render_params, viewpoint_camera, pc, pipe, override_precomp):
    if override_precomp is not None:
        return None, override_precomp

    if pipe.convert_SHs_python:
        shs_view = render_params["features"].transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = render_params["means3D"] - viewpoint_camera.camera_center.repeat(render_params["features"].shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        return None, torch.clamp_min(sh2rgb + 0.5, 0.0)

    return render_params["features"], None

def _render_head(
    rasterizer,
    render_params,
    means2D,
    opacity,
    viewpoint_camera,
    pc,
    pipe,
    override_precomp=None,
):
    head_shs, head_precomp = _resolve_head_appearance(
        render_params=render_params,
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        override_precomp=override_precomp,
    )

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = render_params["cov3D_precomp"]
    else:
        scales = render_params["scales"]
        rotations = render_params["rotations"]

    rendered_thermal, rendered_color, radii = rasterizer(
        means3D=render_params["means3D"],
        means2D=means2D,
        thermal_shs=head_shs,
        color_shs=head_shs,
        thermals_precomp=head_precomp,
        colors_precomp=head_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    return {
        "rendered_thermal": rendered_thermal,
        "rendered_color": rendered_color,
        "radii": radii,
        "visibility_filter": radii > 0,
        "viewspace_points": means2D,
    }

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, override_thermal = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    feature_bindings = pc.get_gbm_feature_bindings()
    rgb_render_params = pc.get_rgb_render_params(
        scaling_modifier=scaling_modifier,
        feature_bindings=feature_bindings,
    )
    thermal_render_params = pc.get_thermal_render_params(
        scaling_modifier=scaling_modifier,
        feature_bindings=feature_bindings,
    )

    rgb_head = _render_head(
        rasterizer=rasterizer,
        render_params=rgb_render_params,
        means2D=screenspace_points,
        opacity=rgb_render_params["opacity"],
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        override_precomp=override_color,
    )

    thermal_screenspace_points = torch.zeros_like(
        thermal_render_params["means3D"],
        dtype=thermal_render_params["means3D"].dtype,
        device="cuda",
    )
    thermal_head = _render_head(
        rasterizer=rasterizer,
        render_params=thermal_render_params,
        means2D=thermal_screenspace_points,
        opacity=thermal_render_params["opacity"],
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        override_precomp=override_thermal,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
            "render": rgb_head["rendered_color"],
            "render_thermal": thermal_head["rendered_thermal"],
            "render_color": rgb_head["rendered_color"],
            "viewspace_points": rgb_head["viewspace_points"],
            "visibility_filter": rgb_head["visibility_filter"],
            "radii": rgb_head["radii"],
            "rgb_viewspace_points": rgb_head["viewspace_points"],
            "rgb_visibility_filter": rgb_head["visibility_filter"],
            "rgb_radii": rgb_head["radii"],
            "thermal_viewspace_points": thermal_head["viewspace_points"],
            "thermal_visibility_filter": thermal_head["visibility_filter"],
            "thermal_radii": thermal_head["radii"],
            "gbm_gate_th2rgb_anchor": feature_bindings["gate_th2rgb_anchor"],
            "gbm_gate_rgb2th_anchor": feature_bindings["gate_rgb2th_anchor"],
            "gbm_stability_reg": feature_bindings["stability_reg"],
            "gbm_gate_sparsity_reg": feature_bindings["gate_sparsity_reg"],
            "gbm_gate_collapse_reg": feature_bindings["gate_collapse_reg"],
            "gbm_gate_overlap_reg": feature_bindings["gate_overlap_reg"],
            "gbm_gate_th2rgb_mean": feature_bindings["gate_th2rgb_mean"],
            "gbm_gate_th2rgb_std": feature_bindings["gate_th2rgb_std"],
            "gbm_gate_th2rgb_max": feature_bindings["gate_th2rgb_max"],
            "gbm_gate_rgb2th_mean": feature_bindings["gate_rgb2th_mean"],
            "gbm_gate_rgb2th_std": feature_bindings["gate_rgb2th_std"],
            "gbm_gate_rgb2th_max": feature_bindings["gate_rgb2th_max"],
            "gbm_delta_th2rgb_mag_mean": feature_bindings["delta_th2rgb_mag_mean"],
            "gbm_delta_rgb2th_mag_mean": feature_bindings["delta_rgb2th_mag_mean"],
    }
