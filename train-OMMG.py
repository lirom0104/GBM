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

import os
import torch
from random import randint
from torchvision.utils import save_image
from utils.loss_utils import l1_loss, ssim, smoothness_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips

import time
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def _prefix_metrics(prefix, metrics):
    return {f"{prefix}/{metric_name}": metric_value for metric_name, metric_value in metrics.items()}

def _build_anchor_contribution_proxy(radii, opacity):
    return radii.detach().clamp_min(0.0) * opacity.detach().reshape(-1)

def _build_pairing_runtime_metrics(use_paired_views, pairing_summary, attempts, hits, sampling_fallbacks):
    hit_rate = (hits / attempts) if attempts > 0 else 0.0
    return {
        "paired_view_enabled": float(bool(use_paired_views)),
        "total_cameras_rgb": float(pairing_summary.get("total_cameras_rgb", 0)),
        "total_cameras_thermal": float(pairing_summary.get("total_cameras_thermal", 0)),
        "paired_camera_count": float(pairing_summary.get("paired_camera_count", 0)),
        "paired_sampling_hit_rate": hit_rate if use_paired_views else 0.0,
        "paired_sampling_fallback_count": float(sampling_fallbacks),
        "paired_camera_fallback_count": float(pairing_summary.get("fallback_count", 0)),
    }

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(
        dataset.sh_degree,
        use_gbm=getattr(dataset, "use_gbm", False),
        use_thermal_residual_geometry=getattr(dataset, "use_thermal_residual_geometry", False),
        gbm_hidden_dim=getattr(dataset, "gbm_hidden_dim", 32),
        gbm_gate_init_bias=getattr(dataset, "gbm_gate_init_bias", -2.2),
        gbm_thermal_grayscale_context=getattr(dataset, "gbm_thermal_grayscale_context", True),
        gbm_rgb_luma_transfer_only=getattr(dataset, "gbm_rgb_luma_transfer_only", True),
    )
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    use_paired_views = getattr(dataset, "use_paired_views", False)
    pairing_summary = scene.getPairingSummary("train")
    if use_paired_views and pairing_summary.get("paired_camera_count", 0) == 0:
        print("[Warning] --use_paired_views enabled but no paired cameras were discovered. Falling back to the legacy train-camera pool.")
    print(
        "[Pairing:train] enabled={} rgb={} thermal={} paired={}".format(
            use_paired_views,
            pairing_summary.get("total_cameras_rgb", 0),
            pairing_summary.get("total_cameras_thermal", 0),
            pairing_summary.get("paired_camera_count", 0),
        )
    )

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    paired_sampling_attempts = 0
    paired_sampling_hits = 0
    paired_sampling_fallbacks = 0

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_thermal = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render_thermal"]

                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    net_thermal_bytes = memoryview((torch.clamp(net_thermal, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send([net_image_bytes,net_thermal_bytes], dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        late_prune_phase_active = (
            getattr(opt, "late_prune_only", False)
            and iteration >= max(opt.densify_until_iter, getattr(opt, "late_prune_only_from_iter", opt.densify_until_iter))
            and iteration <= getattr(opt, "late_prune_only_until_iter", opt.iterations)
        )

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainSamplingCameras(paired_only=use_paired_views).copy()
            
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        paired_sampling_attempts += 1
        if use_paired_views and getattr(viewpoint_cam, "has_paired_view", False):
            paired_sampling_hits += 1
        elif use_paired_views:
            paired_sampling_fallbacks += 1
        #print("viewpoint_cam is:",viewpoint_cam)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, thermal, viewspace_point_tensor, visibility_filter, radii = render_pkg["render_color"], render_pkg["render_thermal"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]



        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_thermal = viewpoint_cam.original_thermal.cuda()
        
        smoothloss_thermal = smoothness_loss(thermal)

        Ll1 = l1_loss(image, gt_image)
        loss_color = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        Ll1_thermal = l1_loss(thermal, gt_thermal)
        loss_thermal = (1.0 - opt.lambda_dssim) * Ll1_thermal + opt.lambda_dssim * (1.0 - ssim(thermal, gt_thermal)) + 0.6 * smoothloss_thermal
        thermal_residual_reg = Ll1.new_zeros(())
        gbm_stability_reg = Ll1.new_zeros(())
        gate_sparsity_reg = Ll1.new_zeros(())
        gate_collapse_reg = Ll1.new_zeros(())
        gate_overlap_reg = Ll1.new_zeros(())
        gbm_log_metrics = {}
        if gaussians.use_thermal_residual_geometry:
            thermal_residual_reg = opt.thermal_residual_l1_weight * gaussians.get_thermal_residual_l1()
        if gaussians.use_gbm:
            gbm_stability_reg = render_pkg["gbm_stability_reg"]
            gate_sparsity_reg = render_pkg["gbm_gate_sparsity_reg"]
            gate_collapse_reg = render_pkg["gbm_gate_collapse_reg"]
            gate_overlap_reg = render_pkg["gbm_gate_overlap_reg"]
            gbm_log_metrics = {
                "gate_th2rgb_mean": render_pkg["gbm_gate_th2rgb_mean"].detach().item(),
                "gate_th2rgb_std": render_pkg["gbm_gate_th2rgb_std"].detach().item(),
                "gate_th2rgb_max": render_pkg["gbm_gate_th2rgb_max"].detach().item(),
                "gate_rgb2th_mean": render_pkg["gbm_gate_rgb2th_mean"].detach().item(),
                "gate_rgb2th_std": render_pkg["gbm_gate_rgb2th_std"].detach().item(),
                "gate_rgb2th_max": render_pkg["gbm_gate_rgb2th_max"].detach().item(),
                "delta_th2rgb_mag_mean": render_pkg["gbm_delta_th2rgb_mag_mean"].detach().item(),
                "delta_rgb2th_mag_mean": render_pkg["gbm_delta_rgb2th_mag_mean"].detach().item(),
                "gbm_stability_reg": gbm_stability_reg.detach().item(),
                "gate_sparsity_reg": gate_sparsity_reg.detach().item(),
                "gate_collapse_reg": gate_collapse_reg.detach().item(),
                "gate_overlap_reg": gate_overlap_reg.detach().item(),
            }
        loss= (
            (loss_color + loss_thermal) * 0.5
            + thermal_residual_reg
            + opt.gbm_stability_weight * gbm_stability_reg
            + opt.gbm_gate_sparsity_weight * gate_sparsity_reg
            + opt.gbm_gate_collapse_weight * gate_collapse_reg
            + opt.gbm_gate_overlap_weight * gate_overlap_reg
        )

        # print("loss:",loss)
        torch.cuda.synchronize()
        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            pairing_runtime_metrics = _build_pairing_runtime_metrics(
                use_paired_views=use_paired_views,
                pairing_summary=pairing_summary,
                attempts=paired_sampling_attempts,
                hits=paired_sampling_hits,
                sampling_fallbacks=paired_sampling_fallbacks,
            )
            if iteration >= getattr(opt, "anchor_stats_warmup_iters", 0):
                anchor_stats_log_metrics = gaussians.update_anchor_multimodal_stats(
                    rgb_visibility_filter=render_pkg["rgb_visibility_filter"],
                    thermal_visibility_filter=render_pkg["thermal_visibility_filter"],
                    rgb_contribution_proxy=_build_anchor_contribution_proxy(render_pkg["rgb_radii"], gaussians.get_rgb_opacity),
                    thermal_contribution_proxy=_build_anchor_contribution_proxy(render_pkg["thermal_radii"], gaussians.get_thermal_opacity),
                    rgb_residual_proxy=Ll1.detach(),
                    thermal_residual_proxy=Ll1_thermal.detach(),
                    gbm_usage_th2rgb=render_pkg["gbm_gate_th2rgb_anchor"],
                    gbm_usage_rgb2th=render_pkg["gbm_gate_rgb2th_anchor"],
                    thermal_geometry_usage=gaussians.get_anchor_thermal_geometry_usage(),
                    ema=opt.anchor_stats_ema,
                )
            else:
                anchor_stats_log_metrics = gaussians.get_anchor_multimodal_stats_summary()

            joint_lifecycle_scores = gaussians.get_joint_lifecycle_scores(iteration=iteration)
            joint_lifecycle_log_metrics = joint_lifecycle_scores["diagnostics"]

            train_metric_scalars = {}
            train_metric_scalars.update(_prefix_metrics("pairing", pairing_runtime_metrics))
            train_metric_scalars.update(_prefix_metrics("anchor_stats", anchor_stats_log_metrics))
            train_metric_scalars.update(_prefix_metrics("joint_lifecycle", joint_lifecycle_log_metrics))
            if gbm_log_metrics:
                train_metric_scalars.update(_prefix_metrics("gbm", gbm_log_metrics))

            # Log and save
            if gaussians.use_gbm and (iteration == first_iter or iteration % 100 == 0):
                print(
                    "\n[ITER {}] GBM gate_th2rgb mean/std/max {:.6f} {:.6f} {:.6f} | "
                    "gate_rgb2th mean/std/max {:.6f} {:.6f} {:.6f}".format(
                        iteration,
                        gbm_log_metrics["gate_th2rgb_mean"],
                        gbm_log_metrics["gate_th2rgb_std"],
                        gbm_log_metrics["gate_th2rgb_max"],
                        gbm_log_metrics["gate_rgb2th_mean"],
                        gbm_log_metrics["gate_rgb2th_std"],
                        gbm_log_metrics["gate_rgb2th_max"],
                    )
                )
                print(
                    "[ITER {}] GBM delta_th2rgb_mag_mean {:.6f} delta_rgb2th_mag_mean {:.6f} | "
                    "stability {:.6f} sparsity {:.6f} collapse {:.6f} overlap {:.6f}".format(
                        iteration,
                        gbm_log_metrics["delta_th2rgb_mag_mean"],
                        gbm_log_metrics["delta_rgb2th_mag_mean"],
                        gbm_log_metrics["gbm_stability_reg"],
                        gbm_log_metrics["gate_sparsity_reg"],
                        gbm_log_metrics["gate_collapse_reg"],
                        gbm_log_metrics["gate_overlap_reg"],
                    )
                )
            if iteration == first_iter or iteration % 100 == 0:
                print(
                    "[ITER {}] Pairing enabled={} paired={} hit_rate {:.6f} sample_fallbacks {} dataset_fallbacks {}".format(
                        iteration,
                        use_paired_views,
                        int(pairing_runtime_metrics["paired_camera_count"]),
                        pairing_runtime_metrics["paired_sampling_hit_rate"],
                        int(pairing_runtime_metrics["paired_sampling_fallback_count"]),
                        int(pairing_runtime_metrics["paired_camera_fallback_count"]),
                    )
                )
                print(
                    "[ITER {}] Anchor stats vis_rgb {:.6f}/{:.6f} vis_th {:.6f}/{:.6f} | "
                    "contrib_rgb {:.6f}/{:.6f} contrib_th {:.6f}/{:.6f}".format(
                        iteration,
                        anchor_stats_log_metrics["visibility_rgb_mean"],
                        anchor_stats_log_metrics["visibility_rgb_std"],
                        anchor_stats_log_metrics["visibility_th_mean"],
                        anchor_stats_log_metrics["visibility_th_std"],
                        anchor_stats_log_metrics["contribution_rgb_mean"],
                        anchor_stats_log_metrics["contribution_rgb_std"],
                        anchor_stats_log_metrics["contribution_th_mean"],
                        anchor_stats_log_metrics["contribution_th_std"],
                    )
                )
                print(
                    "[ITER {}] Anchor stats residual_rgb {:.6f}/{:.6f} residual_th {:.6f}/{:.6f} | "
                    "gbm_usage_th2rgb {:.6f}/{:.6f} gbm_usage_rgb2th {:.6f}/{:.6f} | "
                    "thermal_geom {:.6f}/{:.6f}".format(
                        iteration,
                        anchor_stats_log_metrics["residual_rgb_mean"],
                        anchor_stats_log_metrics["residual_rgb_std"],
                        anchor_stats_log_metrics["residual_th_mean"],
                        anchor_stats_log_metrics["residual_th_std"],
                        anchor_stats_log_metrics["gbm_usage_th2rgb_mean"],
                        anchor_stats_log_metrics["gbm_usage_th2rgb_std"],
                        anchor_stats_log_metrics["gbm_usage_rgb2th_mean"],
                        anchor_stats_log_metrics["gbm_usage_rgb2th_std"],
                        anchor_stats_log_metrics["thermal_geometry_usage_mean"],
                        anchor_stats_log_metrics["thermal_geometry_usage_std"],
                    )
                )
                print(
                    "[ITER {}] Joint lifecycle enabled={} points {} growth {:.6f} | "
                    "split {:.6f}/{:.6f}/{:.6f}".format(
                        iteration,
                        int(joint_lifecycle_log_metrics["joint_lifecycle_enabled"]),
                        int(joint_lifecycle_log_metrics["current_point_count"]),
                        joint_lifecycle_log_metrics["point_growth_ratio_vs_baseline_or_start"],
                        joint_lifecycle_log_metrics["joint_split_score_mean"],
                        joint_lifecycle_log_metrics["joint_split_score_std"],
                        joint_lifecycle_log_metrics["joint_split_score_max"],
                    )
                )
                print(
                    "[ITER {}] Joint lifecycle extra_split {} | suppressed_by_budget {} | "
                    "prune_candidate_ratio {:.6f} veto_ratio {:.6f}".format(
                        iteration,
                        int(joint_lifecycle_log_metrics["joint_split_trigger_count"]),
                        int(joint_lifecycle_log_metrics["split_suppressed_by_budget_count"]),
                        joint_lifecycle_log_metrics["joint_prune_candidate_ratio"],
                        joint_lifecycle_log_metrics["joint_prune_veto_ratio"],
                    )
                )
                print(
                    "[ITER {}] Joint lifecycle prune_both {} | split_rgb {} split_th {} | "
                    "boost_gbm {} boost_thgeo {}".format(
                        iteration,
                        int(joint_lifecycle_log_metrics["prune_by_both_modalities_count"]),
                        int(joint_lifecycle_log_metrics["split_triggered_by_rgb_count"]),
                        int(joint_lifecycle_log_metrics["split_triggered_by_th_count"]),
                        int(joint_lifecycle_log_metrics["split_boosted_by_gbm_count"]),
                        int(joint_lifecycle_log_metrics["split_boosted_by_thgeo_count"]),
                    )
                )

            training_report(
                tb_writer,
                iteration,
                Ll1,
                Ll1_thermal,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                train_metrics=train_metric_scalars,
            )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        iteration=iteration,
                    )
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            elif late_prune_phase_active:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                if getattr(opt, "late_prune_interval", 0) > 0 and iteration % opt.late_prune_interval == 0:
                    removed_count = gaussians.late_prune_only(
                        min_opacity=0.005,
                        extent=scene.cameras_extent,
                        max_screen_size=None,
                        iteration=iteration,
                    )
                    print("[ITER {}] Late prune-only removed {}".format(iteration, removed_count))

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                pass

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, Ll1_thermal, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_metrics=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/l1_thermal_loss', Ll1_thermal.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        if train_metrics:
            for metric_name, metric_value in train_metrics.items():
                tb_writer.add_scalar(metric_name, metric_value, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                l1_thermal_test = 0.0
                psnr_thermal_test = 0.0
                ssim_thermal_test = 0.0
                lpips_thermal_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render_color"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    thermal = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render_thermal"], 0.0, 1.0)
                    gt_thermal = torch.clamp(viewpoint.original_thermal.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/thermal_render".format(viewpoint.image_name), thermal[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/thermal_ground_truth".format(viewpoint.image_name), gt_thermal[None], global_step=iteration)
                            
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()

                    l1_thermal_test += l1_loss(thermal, gt_thermal).mean().double()
                    psnr_thermal_test += psnr(thermal, gt_thermal).mean().double()
                    ssim_thermal_test += ssim(thermal, gt_thermal).mean().double()
                    lpips_thermal_test += lpips(thermal, gt_thermal, net_type='vgg').mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])   
                
                psnr_thermal_test /= len(config['cameras'])
                l1_thermal_test /= len(config['cameras'])
                ssim_thermal_test /= len(config['cameras'])
                lpips_thermal_test /= len(config['cameras'])


                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                print("\n[ITER {}] Thermal Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(iteration, config['name'], l1_thermal_test, psnr_thermal_test, ssim_thermal_test, lpips_thermal_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss_thermal', l1_thermal_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_thermal', psnr_thermal_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim_thermal', ssim_thermal_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips_thermal', lpips_thermal_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
