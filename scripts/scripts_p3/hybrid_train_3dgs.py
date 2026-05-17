#!/usr/bin/env python3
"""
Part 3: Hybrid 3DGS Optimization with Confidence-Weighted Pseudo-View Loss.

Loads the S3PO-GS trained Gaussian model and fine-tunes it by adding
pseudo-views (from generate_pseudo_views.py) with confidence masks
(from compute_confidence_masks.py) to the training set.

Following the BRPO approach and Difix3D distillation strategy:
  - Real sparse views use standard photometric loss
  - Pseudo-views use confidence-weighted loss (beta * C * L_photo)
  - Optional curriculum annealing: beta ramps up from 0 over training

Must be run from the S3PO-GS root directory with S3PO-GS conda env active.

Usage:
    python hybrid_train_3dgs.py \
        --model_path  running_result/5_2/results/.../point_cloud.ply \
        --pseudo_dir  running_result/5_3/pseudo_views/waymo_405841 \
        --conf_dir    running_result/5_3/confidence_masks/waymo_405841 \
        --sparse_dir  dataset_sparse_p2/waymo/405841/FRONT \
        --dataset_type waymo \
        --output_dir  running_result/5_3/hybrid_train/waymo_405841

    python hybrid_train_3dgs.py --config scripts/scripts_p3/configs/waymo_p3.yaml
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

# Will be set up after parsing args
S3PO_ROOT = None


def setup_s3po_imports(s3po_root):
    """Add S3PO-GS to sys.path for Gaussian model imports."""
    global S3PO_ROOT
    S3PO_ROOT = s3po_root
    if s3po_root not in sys.path:
        sys.path.insert(0, s3po_root)


class PseudoViewDataset:
    """Dataset that yields both real sparse views and pseudo-views."""

    def __init__(
        self,
        sparse_dir,
        pseudo_dir,
        conf_dir,
        dataset_type,
        device="cuda",
    ):
        self.device = device
        self.dataset_type = dataset_type

        # Load pseudo-view metadata
        meta_path = Path(pseudo_dir) / "pseudo_meta.json"
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.intrinsics = self.meta["intrinsics"]

        # Real sparse frames
        self.real_frames = []
        for entry in self.meta["sparse_frames"]:
            img = np.array(Image.open(entry["image_path"])).astype(np.float32) / 255.0
            pose = np.array(entry["pose"])
            self.real_frames.append({
                "image": img,
                "pose": pose,
                "is_pseudo": False,
                "confidence": np.ones(img.shape[:2], dtype=np.float32),
            })

        # Pseudo frames with confidence
        self.pseudo_frames = []
        conf_dir = Path(conf_dir)
        for entry in self.meta["pseudo_frames"]:
            idx = entry["pseudo_idx"]
            img = np.array(Image.open(entry["pseudo_path"])).astype(np.float32) / 255.0
            pose = np.array(entry["pose"])

            conf_path = conf_dir / f"{idx:06d}_conf.npy"
            if conf_path.exists():
                confidence = np.load(conf_path)
                if confidence.shape[:2] != img.shape[:2]:
                    confidence = cv2.resize(confidence, (img.shape[1], img.shape[0]))
            else:
                confidence = np.ones(img.shape[:2], dtype=np.float32) * 0.5

            self.pseudo_frames.append({
                "image": img,
                "pose": pose,
                "is_pseudo": True,
                "confidence": confidence,
            })

        print(f"  Loaded {len(self.real_frames)} real + {len(self.pseudo_frames)} pseudo frames")

    def __len__(self):
        return len(self.real_frames) + len(self.pseudo_frames)

    def get_random_real(self):
        return random.choice(self.real_frames)

    def get_random_pseudo(self):
        return random.choice(self.pseudo_frames)

    def get_random_view(self, pseudo_ratio=0.5):
        """Sample a view; pseudo_ratio controls the probability of sampling pseudo."""
        if random.random() < pseudo_ratio and len(self.pseudo_frames) > 0:
            return self.get_random_pseudo()
        return self.get_random_real()


def make_camera(pose_c2w, intrinsics, device="cuda"):
    """Create a camera object compatible with S3PO-GS's render()."""
    from gaussian_splatting.utils.graphics_utils import (
        focal2fov, getWorld2View2, getProjectionMatrix2,
    )

    w2c = np.linalg.inv(pose_c2w)
    R_mat = w2c[:3, :3]
    T_vec = w2c[:3, 3]

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics.get("cx", intrinsics["width"] / 2)
    cy = intrinsics.get("cy", intrinsics["height"] / 2)
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])

    class _Cam:
        pass

    cam = _Cam()
    cam.R = torch.from_numpy(R_mat).float().to(device)
    cam.T = torch.from_numpy(T_vec).float().to(device)
    cam.image_width = width
    cam.image_height = height
    cam.FoVx = focal2fov(fx, width)
    cam.FoVy = focal2fov(fy, height)
    cam.znear = 0.01
    cam.zfar = 100.0

    cam.cam_rot_delta = torch.zeros(3, device=device)
    cam.cam_trans_delta = torch.zeros(3, device=device)
    cam.exposure_a = torch.tensor([0.0], device=device)
    cam.exposure_b = torch.tensor([0.0], device=device)

    cam.world_view_transform = getWorld2View2(cam.R, cam.T).transpose(0, 1)
    cam.projection_matrix = getProjectionMatrix2(
        znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=width, H=height
    ).transpose(0, 1).to(device)
    cam.full_proj_transform = (
        cam.world_view_transform.unsqueeze(0).bmm(
            cam.projection_matrix.unsqueeze(0)
        )
    ).squeeze(0)
    cam.camera_center = cam.world_view_transform.inverse()[3, :3]
    return cam


def hybrid_training_loop(
    gaussians,
    dataset,
    output_dir,
    total_iterations=15000,
    beta_max=0.4,
    beta_warmup=3000,
    pseudo_ratio=0.5,
    lambda_dssim=0.2,
    lr_scale=0.5,
    densify_until=10000,
    densify_interval=100,
    densify_grad_thresh=0.0002,
    save_interval=5000,
    device="cuda",
    mode="sparse_pseudo_confidence",
    lambda_depth=0.1,
    lambda_iso=2.0,
    opacity_prune_thresh=0.005,
    conf_floor=0.05,
):
    """
    Hybrid training: fine-tune 3DGS with real + pseudo views.

    Modes:
        "sparse_only"             - train only on sparse real views (baseline)
        "sparse_pseudo"           - train on sparse + pseudo, uniform beta weight
        "sparse_pseudo_confidence"- train on sparse + pseudo, per-pixel confidence
                                    weighting (includes optical-flow / reprojection
                                    consistency from confidence masks)
    """
    from gaussian_splatting.gaussian_renderer import render
    from gaussian_splatting.utils.loss_utils import l1_loss, ssim
    from munch import munchify

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = munchify({"convert_SHs_python": False, "compute_cov3D_python": False})
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    opt_params = munchify({
        "position_lr_init": 0.00016 * lr_scale,
        "position_lr_final": 0.0000016 * lr_scale,
        "position_lr_delay_mult": 0.01,
        "position_lr_max_steps": total_iterations,
        "feature_lr": 0.0025 * lr_scale,
        "opacity_lr": 0.05 * lr_scale,
        "scaling_lr": 0.001 * lr_scale,
        "rotation_lr": 0.001 * lr_scale,
        "lambda_dssim": lambda_dssim,
        "densification_interval": densify_interval,
        "densify_from_iter": 500,
        "densify_until_iter": densify_until,
        "densify_grad_threshold": densify_grad_thresh,
        "opacity_reset_interval": 3000,
        "percent_dense": 0.01,
    })

    gaussians.training_setup(opt_params)

    xyz = gaussians.get_xyz.detach()
    cameras_extent = (xyz.max(dim=0).values - xyz.min(dim=0).values).norm().item() / 2.0

    intrinsics = dataset.intrinsics
    loss_log = []
    best_loss = float("inf")

    print(f"\n  Hybrid training: {total_iterations} iterations, mode={mode}")
    print(f"  Beta max: {beta_max}, warmup: {beta_warmup}, pseudo ratio: {pseudo_ratio}")
    print(f"  Depth reg: {lambda_depth}, Iso reg: {lambda_iso}")
    print(f"  Scene extent: {cameras_extent:.2f}")

    for iteration in tqdm(range(1, total_iterations + 1), desc="Hybrid training"):
        gaussians.update_learning_rate(iteration)

        if iteration < beta_warmup:
            beta = beta_max * (iteration / beta_warmup)
        else:
            beta = beta_max

        if mode == "sparse_only":
            view_data = dataset.get_random_real()
        else:
            view_data = dataset.get_random_view(pseudo_ratio=pseudo_ratio)

        gt_image = torch.from_numpy(view_data["image"]).permute(2, 0, 1).float().to(device)
        cam = make_camera(view_data["pose"], intrinsics, device)

        render_pkg = render(cam, gaussians, pipeline, background)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        depth = render_pkg.get("depth", None)

        if image.shape[1:] != gt_image.shape[1:]:
            gt_image = F.interpolate(
                gt_image.unsqueeze(0),
                size=(image.shape[1], image.shape[2]),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        Ll1 = l1_loss(image, gt_image)
        loss_ssim = 1.0 - ssim(image, gt_image)
        loss_photo = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * loss_ssim

        if view_data["is_pseudo"]:
            if mode == "sparse_pseudo_confidence":
                conf = torch.from_numpy(view_data["confidence"]).float().to(device)
                if conf.shape != image.shape[1:]:
                    conf = F.interpolate(
                        conf.unsqueeze(0).unsqueeze(0),
                        size=(image.shape[1], image.shape[2]),
                        mode="bilinear", align_corners=False,
                    ).squeeze()

                conf = torch.clamp(conf, min=conf_floor)
                conf3 = conf.unsqueeze(0).expand_as(image)
                weighted_l1 = (conf3 * torch.abs(image - gt_image)).mean()
                weighted_ssim = 1.0 - ssim(image * conf3, gt_image * conf3)
                loss = beta * ((1.0 - lambda_dssim) * weighted_l1 + lambda_dssim * weighted_ssim)
            else:
                loss = beta * loss_photo
        else:
            loss = loss_photo

        # Depth smoothness regularization: penalize large depth gradients
        if depth is not None and lambda_depth > 0:
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)
            depth_dx = torch.abs(depth[:, :, 1:] - depth[:, :, :-1])
            depth_dy = torch.abs(depth[:, 1:, :] - depth[:, :-1, :])
            depth_smooth = depth_dx.mean() + depth_dy.mean()
            loss = loss + lambda_depth * depth_smooth

        # Isotropic regularization with reduced weight to avoid over-constraining
        scaling = gaussians.get_scaling
        iso_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
        loss = loss + lambda_iso * iso_loss.mean()

        loss.backward()

        with torch.no_grad():
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter],
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration % densify_interval == 0 and iteration < densify_until:
                gaussians.densify_and_prune(
                    densify_grad_thresh, opacity_prune_thresh, cameras_extent, None
                )

            if iteration % opt_params.opacity_reset_interval == 0:
                gaussians.reset_opacity()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        if iteration % 100 == 0:
            cur_loss = loss.item()
            loss_log.append({"iter": iteration, "loss": cur_loss, "beta": beta})
            if cur_loss < best_loss:
                best_loss = cur_loss

        if iteration % save_interval == 0 or iteration == total_iterations:
            save_dir = output_dir / "point_cloud" / f"iteration_{iteration}"
            save_dir.mkdir(parents=True, exist_ok=True)
            gaussians.save_ply(str(save_dir / "point_cloud.ply"))

    # Save final model
    final_dir = output_dir / "point_cloud" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(str(final_dir / "point_cloud.ply"))

    # Save training log
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(loss_log, f, indent=2)

    print(f"\n  Training complete. Model saved to: {final_dir}")
    return str(final_dir / "point_cloud.ply")


def main():
    parser = argparse.ArgumentParser(
        description="Part 3: Hybrid 3DGS training with pseudo-views"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to S3PO-GS point_cloud.ply to fine-tune")
    parser.add_argument("--pseudo_dir", type=str, default=None)
    parser.add_argument("--conf_dir", type=str, default=None)
    parser.add_argument("--sparse_dir", type=str, default=None)
    parser.add_argument("--dataset_type", type=str, default="waymo",
                        choices=["waymo", "dl3dv", "re10k"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--proj_root", type=str, default=None)
    parser.add_argument("--mode", type=str, default="sparse_pseudo_confidence",
                        choices=["sparse_only", "sparse_pseudo", "sparse_pseudo_confidence"])
    parser.add_argument("--total_iterations", type=int, default=15000)
    parser.add_argument("--beta_max", type=float, default=0.3,
                        help="Max weight for pseudo-view loss")
    parser.add_argument("--beta_warmup", type=int, default=3000,
                        help="Iterations to ramp beta from 0 to beta_max")
    parser.add_argument("--pseudo_ratio", type=float, default=0.5,
                        help="Probability of sampling a pseudo-view per iteration")
    parser.add_argument("--conf_floor", type=float, default=0.05,
                        help="Minimum per-pixel confidence (soft floor, no views skipped)")
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--lambda_depth", type=float, default=0.1,
                        help="Depth smoothness regularization weight")
    parser.add_argument("--lambda_iso", type=float, default=2.0,
                        help="Isotropic Gaussian regularization weight (reduced from 5.0)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k) and getattr(args, k) is None:
                setattr(args, k, v)

    if args.proj_root is None:
        args.proj_root = str(Path(__file__).resolve().parents[2])

    proj = Path(args.proj_root)
    s3po_root = str(proj / "S3PO-GS")
    setup_s3po_imports(s3po_root)

    from gaussian_splatting.scene.gaussian_model import GaussianModel

    ds_names = {"waymo": "waymo_405841", "dl3dv": "dl3dv_2", "re10k": "re10k_1"}
    ds_name = ds_names[args.dataset_type]

    # Auto-detect paths
    if args.model_path is None:
        scripts_p3 = Path(__file__).resolve().parent
        sys.path.insert(0, str(scripts_p3))
        from generate_pseudo_views import find_s3po_model

        search_dirs = [
            proj / "running_result" / "5_2" / "results",
            proj / "S3PO-GS" / "results",
        ]
        ds_key = {"waymo": "waymo_sparse_405841", "dl3dv": "dl3dv_sparse_2",
                  "re10k": "re10k_sparse_1"}[args.dataset_type]
        for search_dir in search_dirs:
            if search_dir.exists():
                args.model_path, _ = find_s3po_model(search_dir, ds_key)
                if args.model_path:
                    break
        if args.model_path is None:
            print(f"ERROR: S3PO-GS model not found for {args.dataset_type}")
            print(f"  Provide explicitly: --model_path /path/to/point_cloud.ply")
            sys.exit(1)

    if args.sparse_dir is None:
        ds_map = {"waymo": "waymo/405841/FRONT", "dl3dv": "dl3dv/2", "re10k": "re10k/1"}
        args.sparse_dir = str(proj / "dataset_sparse_p2" / ds_map[args.dataset_type])

    if args.pseudo_dir is None:
        args.pseudo_dir = str(proj / "running_result" / "5_3" / "pseudo_views" / ds_name)

    if args.conf_dir is None:
        args.conf_dir = str(proj / "running_result" / "5_3" / "confidence_masks" / ds_name)

    if args.output_dir is None:
        args.output_dir = str(
            proj / "running_result" / "5_3" / "hybrid_train" / ds_name / args.mode
        )

    print("=" * 60)
    print("  Part 3: Hybrid 3DGS Training")
    print("=" * 60)
    print(f"  Mode:          {args.mode}")
    print(f"  Model:         {args.model_path}")
    print(f"  Pseudo views:  {args.pseudo_dir}")
    print(f"  Confidence:    {args.conf_dir}")
    print(f"  Output:        {args.output_dir}")
    print(f"  Iterations:    {args.total_iterations}")
    print(f"  Beta max:      {args.beta_max}")
    print()

    # Load pre-trained Gaussian model
    print("  Loading pre-trained Gaussian model...")
    gaussians = GaussianModel(sh_degree=0)
    gaussians.load_ply(args.model_path)
    # spatial_lr_scale is normally set during scene init; compute from point cloud extent
    xyz = gaussians.get_xyz.detach()
    scene_extent = (xyz.max(dim=0).values - xyz.min(dim=0).values).max().item()
    gaussians.init_lr(scene_extent)
    print(f"  Loaded {gaussians.get_xyz.shape[0]} Gaussians (scene extent: {scene_extent:.2f})")

    # Load dataset
    print("  Loading hybrid dataset...")
    dataset = PseudoViewDataset(
        sparse_dir=args.sparse_dir,
        pseudo_dir=args.pseudo_dir,
        conf_dir=args.conf_dir,
        dataset_type=args.dataset_type,
        device=args.device,
    )

    # Train
    final_model = hybrid_training_loop(
        gaussians=gaussians,
        dataset=dataset,
        output_dir=args.output_dir,
        total_iterations=args.total_iterations,
        beta_max=args.beta_max,
        beta_warmup=args.beta_warmup,
        pseudo_ratio=args.pseudo_ratio,
        lambda_dssim=args.lambda_dssim,
        lambda_depth=args.lambda_depth,
        lambda_iso=args.lambda_iso,
        device=args.device,
        mode=args.mode,
        conf_floor=args.conf_floor,
    )

    print(f"\n  Final model: {final_model}")
    print("  Done!")


if __name__ == "__main__":
    main()
