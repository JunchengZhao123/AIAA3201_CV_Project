#!/usr/bin/env python3
"""
Part 3: Evaluation - Compare Sparse-Only vs Sparse+Pseudo vs Sparse+Pseudo+Confidence.

Renders test views from each trained model and computes PSNR, SSIM, LPIPS.
Also generates side-by-side comparison visualizations.

Usage:
    python evaluate_p3.py --proj_root /path/to/project
    python evaluate_p3.py --proj_root /path/to/project --datasets waymo
    python evaluate_p3.py --config scripts/scripts_p3/configs/waymo_p3.yaml
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

try:
    from skimage.metrics import structural_similarity as compare_ssim
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("WARNING: scikit-image not installed. PSNR/SSIM will be unavailable.")

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("WARNING: torchmetrics not installed. LPIPS will be unavailable.")


def render_test_views(model_path, test_poses, intrinsics, s3po_root, device="cuda"):
    """Render test views from a Gaussian model."""
    if s3po_root not in sys.path:
        sys.path.insert(0, s3po_root)

    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.gaussian_renderer import render
    from gaussian_splatting.utils.graphics_utils import (
        focal2fov, getWorld2View2, getProjectionMatrix2,
    )
    from munch import munchify

    gaussians = GaussianModel(sh_degree=0)
    gaussians.load_ply(model_path)

    pipeline = munchify({"convert_SHs_python": False, "compute_cov3D_python": False})
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics.get("cx", intrinsics["width"] / 2)
    cy = intrinsics.get("cy", intrinsics["height"] / 2)
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])

    proj = getProjectionMatrix2(
        znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=width, H=height
    ).transpose(0, 1).to(device)

    renders = []
    for pose_c2w in tqdm(test_poses, desc="Rendering", leave=False):
        w2c = np.linalg.inv(pose_c2w)
        R_mat = w2c[:3, :3]
        T_vec = w2c[:3, 3]

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
        cam.projection_matrix = proj
        cam.full_proj_transform = (
            cam.world_view_transform.unsqueeze(0).bmm(
                cam.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        cam.camera_center = cam.world_view_transform.inverse()[3, :3]

        with torch.no_grad():
            render_pkg = render(cam, gaussians, pipeline, background)
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        renders.append(image)

    return renders


def load_test_set(sparse_dir, dataset_type):
    """Load test set images and poses."""
    sparse_dir = Path(sparse_dir)
    test_dir = sparse_dir / "test_set"

    if not test_dir.exists():
        print(f"  WARNING: Test set not found at {test_dir}")
        return [], [], {}

    test_images = sorted((test_dir / "rgb").glob("*.png"))

    poses = []
    intrinsics = {}

    if dataset_type == "waymo":
        gt_dir = test_dir / "gt"
        gt_files = sorted(gt_dir.glob("*.txt"))
        for f in gt_files:
            pose = np.loadtxt(f, delimiter=" ").reshape(4, 4)
            poses.append(pose)

        calib_path = sparse_dir / "calib.txt"
        if calib_path.exists():
            for line in open(calib_path).readlines():
                if line.startswith("fx:"):
                    parts = line.split()
                    intrinsics = {
                        "fx": float(parts[1]), "fy": float(parts[3]),
                        "cx": float(parts[5]), "cy": float(parts[7]),
                        "width": 1920, "height": 1280,
                    }
    else:
        cameras_json = test_dir / "cameras.json"
        if cameras_json.exists():
            with open(cameras_json, "r") as f:
                all_cameras = json.load(f)

            # S3PO-GS dl3dvParser subtracts the first *sparse* camera's
            # translation to center the scene.  We must apply the same
            # offset so test cameras are in the same coordinate frame.
            sparse_cameras_json = sparse_dir / "cameras.json"
            init_trans = np.zeros(3)
            if sparse_cameras_json.exists():
                with open(sparse_cameras_json, "r") as f:
                    sparse_cams = json.load(f)
                init_trans = np.array(sparse_cams[0]["cam_trans"])

            from scipy.spatial.transform import Rotation as Rot
            for cam_data in all_cameras:
                qx, qy, qz, qw = cam_data["cam_quat"]
                tx, ty, tz = cam_data["cam_trans"]
                rot = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
                T_w2c = np.eye(4)
                T_w2c[:3, :3] = rot
                T_w2c[:3, 3] = np.array([tx, ty, tz]) - init_trans
                T_c2w = np.linalg.inv(T_w2c)
                poses.append(T_c2w)

            w, h = 256, 256
            intrinsics = {
                "fx": all_cameras[0].get("fx", 0.5) * w,
                "fy": all_cameras[0].get("fy", 0.5) * h,
                "cx": all_cameras[0].get("cx", 0.5) * w,
                "cy": all_cameras[0].get("cy", 0.5) * h,
                "width": w, "height": h,
            }

    n_eval = min(len(test_images), len(poses))
    return test_images[:n_eval], poses[:n_eval], intrinsics


def compute_metrics(renders, gt_images, device="cuda"):
    """Compute PSNR, SSIM, LPIPS between rendered and GT images."""
    psnr_scores, ssim_scores, lpips_scores = [], [], []

    if HAS_LPIPS:
        cal_lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)

    n_eval = min(len(renders), len(gt_images))

    for idx in range(n_eval):
        pred = renders[idx]
        gt_img = np.array(Image.open(gt_images[idx])).astype(np.float64) / 255.0
        gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).float().to(device)

        pred_np = pred.cpu().numpy().transpose(1, 2, 0)

        # Resize if needed
        if pred.shape[1:] != gt_tensor.shape[1:]:
            pred = F.interpolate(
                pred.unsqueeze(0),
                size=(gt_tensor.shape[1], gt_tensor.shape[2]),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
            pred_np = pred.cpu().numpy().transpose(1, 2, 0)

        gt_np = gt_img

        if HAS_SKIMAGE:
            psnr_val = compare_psnr(gt_np, pred_np.astype(np.float64), data_range=1.0)
            ssim_val = compare_ssim(gt_np, pred_np.astype(np.float64),
                                    channel_axis=2, data_range=1.0)
            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)

        if HAS_LPIPS:
            lpips_val = cal_lpips(
                pred.unsqueeze(0).float(),
                gt_tensor.unsqueeze(0).float()
            ).item()
            lpips_scores.append(lpips_val)

    results = {"n_images": n_eval}
    if psnr_scores:
        results["mean_psnr"] = float(np.mean(psnr_scores))
        results["std_psnr"] = float(np.std(psnr_scores))
        results["mean_ssim"] = float(np.mean(ssim_scores))
        results["std_ssim"] = float(np.std(ssim_scores))
    if lpips_scores:
        results["mean_lpips"] = float(np.mean(lpips_scores))
        results["std_lpips"] = float(np.std(lpips_scores))

    return results


def create_comparison_grid(renders_dict, gt_images, output_dir, n_samples=5):
    """Create side-by-side visual comparison images."""
    output_dir = Path(output_dir)
    (output_dir / "comparisons").mkdir(parents=True, exist_ok=True)

    n_total = min(len(gt_images), *[len(r) for r in renders_dict.values()])
    indices = np.linspace(0, n_total - 1, min(n_samples, n_total), dtype=int)

    for idx in indices:
        gt_img = np.array(Image.open(gt_images[idx]))
        H, W = gt_img.shape[:2]

        images = [gt_img]
        labels = ["GT"]

        for label, renders in renders_dict.items():
            if idx < len(renders):
                pred_np = (renders[idx].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                if pred_np.shape[:2] != (H, W):
                    pred_np = cv2.resize(pred_np, (W, H))
                images.append(pred_np)
                labels.append(label)

        # Create grid
        n_cols = len(images)
        grid = np.zeros((H + 30, W * n_cols, 3), dtype=np.uint8)
        for i, (img, label) in enumerate(zip(images, labels)):
            grid[30:30 + H, i * W:(i + 1) * W] = img[:, :, :3]
            cv2.putText(grid, label, (i * W + 10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imwrite(str(output_dir / "comparisons" / f"compare_{idx:04d}.png"), grid[:, :, ::-1])

    print(f"  Saved {len(indices)} comparison images to: {output_dir / 'comparisons'}")


def evaluate_dataset(
    dataset_type, proj_root, s3po_root, device="cuda"
):
    """Evaluate all Part 3 conditions for one dataset."""
    proj = Path(proj_root)
    ds_names = {"waymo": "waymo_405841", "dl3dv": "dl3dv_2", "re10k": "re10k_1"}
    ds_name = ds_names[dataset_type]
    ds_map = {"waymo": "waymo/405841/FRONT", "dl3dv": "dl3dv/2", "re10k": "re10k/1"}

    sparse_dir = proj / "dataset_sparse_p2" / ds_map[dataset_type]
    results_base = proj / "running_result" / "5_3"

    print(f"\n{'='*60}")
    print(f"  Evaluating: {dataset_type} ({ds_name})")
    print(f"{'='*60}")

    # Load test set
    test_images, test_poses, intrinsics = load_test_set(sparse_dir, dataset_type)
    if not test_images:
        print(f"  SKIP: No test set found for {dataset_type}")
        return None

    print(f"  Test images: {len(test_images)}")

    # Find models for each condition
    conditions = {}

    # 1. Baseline: S3PO-GS model from Part 2 (sparse only)
    ds_key = {"waymo": "waymo_sparse_405841", "dl3dv": "dl3dv_sparse_2",
              "re10k": "re10k_sparse_1"}[dataset_type]
    p2_search_dirs = [
        proj / "running_result" / "5_2" / "results",
        proj / "S3PO-GS" / "results",
    ]
    ply_candidates_patterns = [
        "point_cloud/final/point_cloud.ply",
        "point_cloud/final_after_opt/point_cloud.ply",
    ]
    for p2_results in p2_search_dirs:
        if not p2_results.exists():
            continue
        for ds_dir in sorted(p2_results.glob(f"*{ds_key}*"), reverse=True):
            ts_dirs = sorted([d for d in ds_dir.iterdir() if d.is_dir()], reverse=True)
            for ts_dir in ts_dirs:
                for cand in ply_candidates_patterns:
                    matches = sorted(ts_dir.glob(cand), reverse=True)
                    if matches:
                        conditions["P2_Sparse_Only"] = str(matches[0])
                        break
                if "P2_Sparse_Only" in conditions:
                    break
            if "P2_Sparse_Only" in conditions:
                break
        if "P2_Sparse_Only" in conditions:
            break

    # 2-4. Part 3 hybrid-trained models
    for mode_name, mode_dir in [
        ("Sparse_Only_Retrain", "sparse_only"),
        ("Sparse+Pseudo", "sparse_pseudo"),
        ("Sparse+Pseudo+Conf", "sparse_pseudo_confidence"),
    ]:
        model_dir = results_base / "hybrid_train" / ds_name / mode_dir
        for cand in ["point_cloud/final/point_cloud.ply",
                     "point_cloud/iteration_15000/point_cloud.ply"]:
            ply = model_dir / cand
            if ply.exists():
                conditions[mode_name] = str(ply)
                break

    if not conditions:
        print(f"  No trained models found for {dataset_type}")
        return None

    print(f"  Found {len(conditions)} conditions to evaluate:")
    for name, path in conditions.items():
        print(f"    {name}: {path}")

    # Render and evaluate each condition
    all_renders = {}
    all_metrics = {}
    eval_output = results_base / "evaluation" / ds_name
    eval_output.mkdir(parents=True, exist_ok=True)

    for cond_name, model_path in conditions.items():
        print(f"\n  --- {cond_name} ---")
        renders = render_test_views(
            model_path, test_poses, intrinsics, s3po_root, device
        )
        all_renders[cond_name] = renders

        metrics = compute_metrics(renders, test_images, device)
        all_metrics[cond_name] = metrics

        if metrics.get("mean_psnr"):
            print(f"      PSNR:  {metrics['mean_psnr']:.2f}")
            print(f"      SSIM:  {metrics['mean_ssim']:.4f}")
        if metrics.get("mean_lpips"):
            print(f"      LPIPS: {metrics['mean_lpips']:.4f}")

        # Save per-condition renders
        render_out = eval_output / cond_name / "renders"
        render_out.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(renders[:20]):
            img_np = (r.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            Image.fromarray(img_np).save(render_out / f"{i:04d}.png")

    # Create comparison grid
    create_comparison_grid(all_renders, test_images, eval_output, n_samples=8)

    # Save metrics
    with open(eval_output / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Part 3: Evaluation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--proj_root", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=["waymo", "dl3dv", "re10k"],
                        choices=["waymo", "dl3dv", "re10k"])
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

    print("=" * 60)
    print("  Part 3: Generative Enhancement Evaluation")
    print("=" * 60)
    print(f"  Project root: {proj}")
    print(f"  Datasets:    {args.datasets}")
    print()

    all_results = {}
    for ds in args.datasets:
        result = evaluate_dataset(ds, args.proj_root, s3po_root, args.device)
        if result:
            all_results[ds] = result

    # Summary table
    print("\n\n" + "=" * 80)
    print("  PART 3 EVALUATION SUMMARY")
    print("=" * 80)
    header = f"  {'Dataset':<10} {'Condition':<25} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8}"
    print(header)
    print(f"  {'-'*10} {'-'*25} {'-'*8} {'-'*8} {'-'*8}")

    for ds, metrics_dict in all_results.items():
        for cond, metrics in metrics_dict.items():
            psnr_s = f"{metrics.get('mean_psnr', 0):.2f}" if metrics.get("mean_psnr") else "N/A"
            ssim_s = f"{metrics.get('mean_ssim', 0):.4f}" if metrics.get("mean_ssim") else "N/A"
            lpips_s = f"{metrics.get('mean_lpips', 0):.4f}" if metrics.get("mean_lpips") else "N/A"
            print(f"  {ds:<10} {cond:<25} {psnr_s:<8} {ssim_s:<8} {lpips_s:<8}")
        print()

    # Save overall summary
    summary_path = proj / "running_result" / "5_3" / "evaluation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
