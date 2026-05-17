#!/usr/bin/env python3
"""
Render held-out test views using a trained S3PO-GS Gaussian model.

Must be run inside the S3PO-GS conda environment (with gaussian_splatting available).

Usage (from S3PO-GS root):
    python /path/to/render_test_views_s3po.py \
        --model_path results/waymo_405841/point_cloud/final/point_cloud.ply \
        --test_cameras /path/to/dataset_sparse_p2/waymo/405841/FRONT/test_set/gt/ \
        --test_images /path/to/dataset_sparse_p2/waymo/405841/FRONT/test_set/rgb/ \
        --calib_file /path/to/dataset_sparse_p2/waymo/405841/FRONT/calib.txt \
        --dataset_type waymo \
        --output_dir /path/to/running_result/5_2/S3PO_GS_SLAM/waymo_405841/test_renders/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import focal2fov
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

from scipy.spatial.transform import Rotation as R
from munch import munchify


class SimpleCamera:
    """Minimal camera object compatible with S3PO-GS render()."""
    def __init__(self, R_mat, T_vec, fx, fy, cx, cy, width, height, device="cuda"):
        self.R = torch.from_numpy(R_mat).float().to(device)
        self.T = torch.from_numpy(T_vec).float().to(device)

        self.image_width = width
        self.image_height = height
        self.FoVx = focal2fov(fx, width)
        self.FoVy = focal2fov(fy, height)

        self.znear = 0.01
        self.zfar = 100.0

        world_view_transform = torch.zeros(4, 4, device=device)
        world_view_transform[:3, :3] = self.R.T
        world_view_transform[:3, 3] = self.T
        world_view_transform[3, 3] = 1.0
        self.world_view_transform = world_view_transform.T

        self.projection_matrix = self._get_projection_matrix(
            self.znear, self.zfar, self.FoVx, self.FoVy, device
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = (-self.R.T @ self.T.unsqueeze(1)).squeeze(1)

    @staticmethod
    def _get_projection_matrix(znear, zfar, fovX, fovY, device):
        import math
        tanHalfFovY = math.tan(fovY / 2)
        tanHalfFovX = math.tan(fovX / 2)

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4, device=device)
        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = -1.0
        P[2, 2] = -(zfar + znear) / (zfar - znear)
        P[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
        return P


def load_waymo_test_cameras(gt_dir, calib_file, width=1920, height=1280):
    """Load Waymo test cameras from GT 4x4 matrix files + calibration."""
    calib_lines = open(calib_file).readlines()
    fx, fy, cx, cy = None, None, None, None
    for line in calib_lines:
        if line.startswith("fx:"):
            parts = line.split()
            fx = float(parts[1])
            fy = float(parts[3])
            cx = float(parts[5])
            cy = float(parts[7])

    gt_files = sorted(Path(gt_dir).glob("*.txt"))
    cameras = []
    for gt_file in gt_files:
        pose = np.loadtxt(gt_file, delimiter=" ").reshape(4, 4)
        # pose is world-to-camera (c2w); for render we need w2c decomposed as R, T
        w2c = np.linalg.inv(pose)
        R_mat = w2c[:3, :3]
        T_vec = w2c[:3, 3]
        cam = SimpleCamera(R_mat, T_vec, fx, fy, cx, cy, width, height)
        cameras.append(cam)
    return cameras


def load_json_test_cameras(cameras_json, width=256, height=256):
    """Load DL3DV/Re10k test cameras from cameras.json."""
    with open(cameras_json, "r") as f:
        all_cameras = json.load(f)

    intrinsics = all_cameras[0]
    fx = intrinsics["fx"] * width
    fy = intrinsics["fy"] * height
    cx = intrinsics["cx"] * width
    cy = intrinsics["cy"] * height

    cameras = []
    init_trans = np.array(all_cameras[0]["cam_trans"])
    for cam_data in all_cameras:
        qx, qy, qz, qw = cam_data["cam_quat"]
        tx, ty, tz = cam_data["cam_trans"]

        rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = np.array([tx, ty, tz]) - init_trans

        # transform is w2c; decompose for renderer
        R_mat = transform[:3, :3]
        T_vec = transform[:3, 3]
        cam = SimpleCamera(R_mat, T_vec, fx, fy, cx, cy, width, height)
        cameras.append(cam)
    return cameras


def main():
    parser = argparse.ArgumentParser(description="Render test views from trained S3PO-GS model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to point_cloud.ply")
    parser.add_argument("--test_cameras", type=str, required=True,
                        help="Path to test GT poses (directory for waymo, cameras.json for dl3dv/re10k)")
    parser.add_argument("--test_images", type=str, required=True,
                        help="Path to test GT images directory")
    parser.add_argument("--dataset_type", type=str, required=True,
                        choices=["waymo", "dl3dv", "re10k"])
    parser.add_argument("--calib_file", type=str, default=None,
                        help="Calibration file (required for waymo)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for rendered images and metrics")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir = output_dir / "rendered"
    render_dir.mkdir(exist_ok=True)

    # Load Gaussian model
    print(f"Loading model from: {args.model_path}")
    gaussians = GaussianModel(sh_degree=0)
    gaussians.load_ply(args.model_path)

    # Pipeline params (minimal)
    pipeline = munchify({
        "convert_SHs_python": False,
        "compute_cov3D_python": False,
    })
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    # Load test cameras
    print(f"Loading test cameras ({args.dataset_type})...")
    if args.dataset_type == "waymo":
        if args.calib_file is None:
            raise ValueError("--calib_file required for waymo dataset")
        w = args.width or 1920
        h = args.height or 1280
        cameras = load_waymo_test_cameras(args.test_cameras, args.calib_file, w, h)
    else:
        w = args.width or 256
        h = args.height or 256
        cameras = load_json_test_cameras(args.test_cameras, w, h)

    # Load GT images
    gt_images_dir = Path(args.test_images)
    gt_files = sorted(gt_images_dir.glob("*.png"))
    n_eval = min(len(cameras), len(gt_files))
    print(f"  {n_eval} test views to evaluate")

    # Render and evaluate
    psnr_scores, ssim_scores, lpips_scores = [], [], []
    if HAS_LPIPS:
        cal_lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)

    for idx in range(n_eval):
        cam = cameras[idx]
        render_pkg = render(cam, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        gt_img = np.array(Image.open(gt_files[idx])).astype(np.float32) / 255.0
        gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).to(device)

        # Resize if needed
        if image.shape[1:] != gt_tensor.shape[1:]:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(gt_tensor.shape[1], gt_tensor.shape[2]),
                mode="bilinear", align_corners=False
            ).squeeze(0)

        # Metrics
        psnr_val = psnr(image.unsqueeze(0), gt_tensor.unsqueeze(0)).item()
        ssim_val = ssim(image.unsqueeze(0), gt_tensor.unsqueeze(0)).item()
        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)

        if HAS_LPIPS:
            lpips_val = cal_lpips(image.unsqueeze(0), gt_tensor.unsqueeze(0)).item()
            lpips_scores.append(lpips_val)

        # Save rendered image
        pred_np = (image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(pred_np).save(render_dir / f"{idx:06d}_pred.png")

        if idx % 10 == 0:
            print(f"    [{idx+1}/{n_eval}] PSNR={psnr_val:.2f} SSIM={ssim_val:.4f}")

    # Summary
    results = {
        "n_images": n_eval,
        "mean_psnr": float(np.mean(psnr_scores)),
        "mean_ssim": float(np.mean(ssim_scores)),
        "std_psnr": float(np.std(psnr_scores)),
        "std_ssim": float(np.std(ssim_scores)),
        "per_image_psnr": psnr_scores,
        "per_image_ssim": ssim_scores,
    }
    if lpips_scores:
        results["mean_lpips"] = float(np.mean(lpips_scores))
        results["std_lpips"] = float(np.std(lpips_scores))
        results["per_image_lpips"] = lpips_scores

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Test View Results:")
    print(f"    PSNR:  {results['mean_psnr']:.2f} +/- {results['std_psnr']:.2f}")
    print(f"    SSIM:  {results['mean_ssim']:.4f} +/- {results['std_ssim']:.4f}")
    if lpips_scores:
        print(f"    LPIPS: {results['mean_lpips']:.4f} +/- {results['std_lpips']:.4f}")
    print(f"\n  Saved to: {output_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
