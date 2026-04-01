import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel


def _project_points(points, viewpoint_camera):
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    points_h = torch.cat((points, ones), dim=1)
    clip = points_h @ viewpoint_camera.full_proj_transform
    w = clip[:, 3]
    ndc = clip[:, :3] / (w.unsqueeze(1) + 1e-8)

    # Match the rasterizer's ndc2Pix convention to avoid vertically flipping
    # diagnostic heatmaps relative to the rendered/image coordinate system.
    x = ((ndc[:, 0] + 1.0) * viewpoint_camera.image_width - 1.0) * 0.5
    y = ((ndc[:, 1] + 1.0) * viewpoint_camera.image_height - 1.0) * 0.5

    valid = (
        torch.isfinite(x)
        & torch.isfinite(y)
        & torch.isfinite(ndc[:, 2])
        & (w > 0)
        & (ndc[:, 0] >= -1.0)
        & (ndc[:, 0] <= 1.0)
        & (ndc[:, 1] >= -1.0)
        & (ndc[:, 1] <= 1.0)
        & (ndc[:, 2] >= 0.0)
    )
    return x, y, valid


def _gaussian_kernel(kernel_size=11, sigma=2.5, device="cuda"):
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d[None, None]


def _scatter_average_heatmap(x, y, values, height, width, kernel):
    if values.numel() == 0:
        return torch.zeros((height, width), dtype=torch.float32, device=kernel.device)

    x_idx = x.round().long().clamp(0, width - 1)
    y_idx = y.round().long().clamp(0, height - 1)
    flat_idx = y_idx * width + x_idx

    sums = torch.zeros(height * width, dtype=torch.float32, device=values.device)
    counts = torch.zeros(height * width, dtype=torch.float32, device=values.device)
    sums.index_add_(0, flat_idx, values.float())
    counts.index_add_(0, flat_idx, torch.ones_like(values, dtype=torch.float32))

    sums = sums.view(1, 1, height, width)
    counts = counts.view(1, 1, height, width)
    blurred_sums = F.conv2d(sums, kernel, padding=kernel.shape[-1] // 2)
    blurred_counts = F.conv2d(counts, kernel, padding=kernel.shape[-1] // 2)
    heatmap = blurred_sums / (blurred_counts + 1e-8)
    return heatmap[0, 0]


def _normalize_heatmap(heatmap):
    finite_mask = torch.isfinite(heatmap)
    if not finite_mask.any():
        return torch.zeros_like(heatmap)

    finite_values = heatmap[finite_mask]
    min_value = finite_values.min()
    max_value = finite_values.max()
    if torch.isclose(max_value, min_value):
        return torch.zeros_like(heatmap)
    return ((heatmap - min_value) / (max_value - min_value)).clamp(0.0, 1.0)


def _colorize_heatmap(heatmap):
    try:
        import matplotlib.cm as cm

        colored = cm.get_cmap("turbo")(heatmap.detach().cpu().numpy())[..., :3]
        return torch.from_numpy(colored).permute(2, 0, 1).float()
    except Exception:
        red = torch.clamp(1.5 * heatmap - 0.5, 0.0, 1.0)
        green = torch.clamp(1.5 - torch.abs(4.0 * heatmap - 2.0), 0.0, 1.0)
        blue = torch.clamp(1.0 - 1.5 * heatmap, 0.0, 1.0)
        return torch.stack((red, green, blue), dim=0).cpu()


def _save_tensor_image(image_tensor, path):
    image = image_tensor.detach().cpu().clamp(0.0, 1.0)
    image_np = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(image_np).save(path)


def _save_visualization(base_image, heatmap, output_stem):
    normalized = _normalize_heatmap(heatmap)
    colored = _colorize_heatmap(normalized)
    overlay = (0.55 * base_image[:3].detach().cpu() + 0.45 * colored).clamp(0.0, 1.0)
    grayscale = normalized.unsqueeze(0).repeat(3, 1, 1).cpu()

    _save_tensor_image(grayscale, str(output_stem) + "_gray.png")
    _save_tensor_image(colored, str(output_stem) + "_color.png")
    _save_tensor_image(overlay, str(output_stem) + "_overlay.png")


def _feature_bindings_to_scalars(gaussians, feature_bindings):
    num_points = gaussians.get_xyz.shape[0]
    zeros = torch.zeros(num_points, dtype=torch.float32, device=gaussians.get_xyz.device)

    if feature_bindings["gate_th2rgb"] is None:
        gate_th2rgb = zeros
        gate_rgb2th = zeros
        delta_th2rgb_mag = zeros
        delta_rgb2th_mag = zeros
    else:
        gate_th2rgb = feature_bindings["gate_th2rgb"].reshape(num_points, -1).mean(dim=1)
        gate_rgb2th = feature_bindings["gate_rgb2th"].reshape(num_points, -1).mean(dim=1)
        delta_th2rgb_mag = feature_bindings["delta_th2rgb"].reshape(num_points, -1).norm(dim=1)
        delta_rgb2th_mag = feature_bindings["delta_rgb2th"].reshape(num_points, -1).norm(dim=1)

    residual_xyz_mag = gaussians._delta_xyz_th.norm(dim=1)
    residual_scaling_mag = gaussians._delta_scaling_th.norm(dim=1)
    residual_geometry_mag = residual_xyz_mag + residual_scaling_mag

    return {
        "gate_th2rgb": gate_th2rgb,
        "gate_rgb2th": gate_rgb2th,
        "delta_th2rgb_mag": delta_th2rgb_mag,
        "delta_rgb2th_mag": delta_rgb2th_mag,
        "residual_xyz_mag": residual_xyz_mag,
        "residual_scaling_mag": residual_scaling_mag,
        "residual_geometry_mag": residual_geometry_mag,
    }


def _compute_usage_frequency(views, gaussians, pipeline, background):
    usage_counts = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.float32, device=gaussians.get_xyz.device)
    if len(views) == 0:
        return usage_counts

    for view in tqdm(views, desc="Accumulating usage frequency"):
        render_pkg = render(view, gaussians, pipeline, background)
        usage_counts += render_pkg["visibility_filter"].float()

    return usage_counts / float(len(views))


def _metric_summary(values):
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std().item()) if values.numel() > 1 else 0.0,
    }


def _select_views(scene, split, max_views):
    split_views = []
    if split in ("train", "all"):
        split_views.append(("train", scene.getTrainCameras()))
    if split in ("test", "all"):
        split_views.append(("test", scene.getTestCameras()))

    selected = []
    for split_name, views in split_views:
        if max_views > 0:
            views = views[:max_views]
        selected.append((split_name, views))
    return selected


def analyze_split(split_name, views, gaussians, pipeline, background, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    kernel = _gaussian_kernel(device=gaussians.get_xyz.device)
    feature_bindings = gaussians.get_gbm_feature_bindings()
    point_metrics = _feature_bindings_to_scalars(gaussians, feature_bindings)
    point_metrics["usage_frequency"] = _compute_usage_frequency(views, gaussians, pipeline, background)

    torch.save(
        {
            "shared_xyz": gaussians.get_xyz.detach().cpu(),
            "thermal_xyz": gaussians.get_thermal_xyz.detach().cpu(),
            "metrics": {name: tensor.detach().cpu() for name, tensor in point_metrics.items()},
        },
        output_dir / "per_point_metrics.pt",
    )

    summary = {
        "split": split_name,
        "num_views": len(views),
        "num_points": int(gaussians.get_xyz.shape[0]),
        "use_gbm": bool(gaussians.use_gbm),
        "use_thermal_residual_geometry": bool(gaussians.use_thermal_residual_geometry),
        "metrics": {name: _metric_summary(values) for name, values in point_metrics.items()},
    }
    with open(output_dir / "summary.json", "w") as summary_file:
        json.dump(summary, summary_file, indent=2)

    shared_points = gaussians.get_xyz
    thermal_points = gaussians.get_thermal_xyz

    for view in tqdm(views, desc=f"Saving {split_name} diagnostics"):
        view_dir = output_dir / view.image_name
        view_dir.mkdir(parents=True, exist_ok=True)
        base_image = view.original_image[:3].detach().cpu()

        shared_x, shared_y, shared_valid = _project_points(shared_points, view)
        thermal_x, thermal_y, thermal_valid = _project_points(thermal_points, view)

        metric_specs = [
            ("gate_th2rgb", shared_x, shared_y, shared_valid),
            ("gate_rgb2th", shared_x, shared_y, shared_valid),
            ("delta_th2rgb_mag", shared_x, shared_y, shared_valid),
            ("delta_rgb2th_mag", shared_x, shared_y, shared_valid),
            ("usage_frequency", shared_x, shared_y, shared_valid),
            ("residual_geometry_mag", thermal_x, thermal_y, thermal_valid),
        ]

        for metric_name, x, y, valid in metric_specs:
            heatmap = _scatter_average_heatmap(
                x=x[valid],
                y=y[valid],
                values=point_metrics[metric_name][valid],
                height=view.image_height,
                width=view.image_width,
                kernel=kernel,
            )
            _save_visualization(base_image, heatmap, view_dir / metric_name)


def analyze_scene(dataset, iteration, pipeline, split, max_views, output_dir):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.sh_degree,
            use_gbm=getattr(dataset, "use_gbm", False),
            use_thermal_residual_geometry=getattr(dataset, "use_thermal_residual_geometry", False),
            gbm_hidden_dim=getattr(dataset, "gbm_hidden_dim", 32),
            gbm_gate_init_bias=getattr(dataset, "gbm_gate_init_bias", -2.2),
            gbm_thermal_grayscale_context=getattr(dataset, "gbm_thermal_grayscale_context", True),
            gbm_rgb_luma_transfer_only=getattr(dataset, "gbm_rgb_luma_transfer_only", True),
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        analysis_root = Path(output_dir) if output_dir else Path(dataset.model_path) / "analysis" / "gbm_diagnostics"
        analysis_root = analysis_root / f"ours_{scene.loaded_iter}"

        for split_name, views in _select_views(scene, split, max_views):
            analyze_split(
                split_name=split_name,
                views=views,
                gaussians=gaussians,
                pipeline=pipeline,
                background=background,
                output_dir=analysis_root / split_name,
            )


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="GBM diagnostic visualization")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--max_views", default=0, type=int)
    parser.add_argument("--output_dir", default="", type=str)
    args = get_combined_args(parser)

    analyze_scene(
        dataset=model.extract(args),
        iteration=args.iteration,
        pipeline=pipeline.extract(args),
        split=args.split,
        max_views=args.max_views,
        output_dir=args.output_dir,
    )
