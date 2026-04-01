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
import random
import json
from collections import Counter
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    

    gaussians : GaussianModel

    @staticmethod
    def _summarize_pairing(cam_infos):
        strategy_counts = Counter(getattr(cam_info, "pair_strategy", "camera_bundle") for cam_info in cam_infos)
        paired_camera_count = sum(1 for cam_info in cam_infos if getattr(cam_info, "has_paired_view", False))
        fallback_count = sum(1 for cam_info in cam_infos if getattr(cam_info, "pair_fallback_used", False))
        return {
            "total_cameras_rgb": len(cam_infos),
            "total_cameras_thermal": sum(1 for cam_info in cam_infos if getattr(cam_info, "thermal_path", "")),
            "paired_camera_count": paired_camera_count,
            "fallback_count": fallback_count,
            "strategy_counts": dict(strategy_counts),
        }

    @staticmethod
    def _print_pairing_summary(split_name, summary):
        print(
            "[Pairing:{}] rgb={} thermal={} paired={} fallback={}".format(
                split_name,
                summary["total_cameras_rgb"],
                summary["total_cameras_thermal"],
                summary["paired_camera_count"],
                summary["fallback_count"],
            )
        )
        if summary["strategy_counts"]:
            strategy_summary = ", ".join(
                "{}={}".format(strategy_name, count)
                for strategy_name, count in sorted(summary["strategy_counts"].items())
            )
            print("[Pairing:{}] strategies {}".format(split_name, strategy_summary))

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.use_paired_views = getattr(args, "use_paired_views", False)

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.paired_train_cameras = {}
        self.paired_test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background)
        else:
            assert False, "Could not recognize scene type!"

        self.train_pairing_summary = self._summarize_pairing(scene_info.train_cameras)
        self.test_pairing_summary = self._summarize_pairing(scene_info.test_cameras)
        self._print_pairing_summary("train", self.train_pairing_summary)
        self._print_pairing_summary("test", self.test_pairing_summary)

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling


        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            self.paired_train_cameras[resolution_scale] = [
                camera for camera in self.train_cameras[resolution_scale] if getattr(camera, "has_paired_view", False)
            ]
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            self.paired_test_cameras[resolution_scale] = [
                camera for camera in self.test_cameras[resolution_scale] if getattr(camera, "has_paired_view", False)
            ]
            
        if self.loaded_iter:
            point_cloud_path = os.path.join(
                self.model_path,
                "point_cloud",
                "iteration_" + str(self.loaded_iter),
            )
            self.gaussians.load_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            self.gaussians.load_feature_modules(point_cloud_path)
            self.gaussians.load_anchor_stats(point_cloud_path)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_feature_modules(point_cloud_path)
        self.gaussians.save_anchor_stats(point_cloud_path)


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTrainSamplingCameras(self, scale=1.0, paired_only=False):
        if paired_only and self.paired_train_cameras.get(scale):
            return self.paired_train_cameras[scale]
        return self.getTrainCameras(scale)

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getPairingSummary(self, split="train"):
        if split == "test":
            return self.test_pairing_summary
        return self.train_pairing_summary
    
