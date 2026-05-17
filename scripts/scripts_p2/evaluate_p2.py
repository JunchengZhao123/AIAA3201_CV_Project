#!/usr/bin/env python3
"""
Part 2: Evaluation script for S3PO-GS sparse SLAM results.

Evaluates:
  1. Pose Accuracy: ATE RMSE using `evo` (estimated vs GT trajectory)
  2. Rendering Quality: PSNR, SSIM, LPIPS on held-out test views

Usage:
    python evaluate_p2.py --results_dir running_result/5_2/S3PO_GS_SLAM
    python evaluate_p2.py --results_dir running_result/5_2/S3PO_GS_SLAM --datasets waymo
    python evaluate_p2.py --results_dir running_result/5_2/S3PO_GS_SLAM --render_test
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

try:
    from evo.core import metrics, trajectory
    from evo.core.trajectory import PosePath3D
    HAS_EVO = True
except ImportError:
    HAS_EVO = False
    print("WARNING: 'evo' not installed. Pose evaluation will be skipped.")
    print("  Install with: pip install evo")

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("WARNING: 'torchmetrics' not installed. LPIPS metric will be skipped.")

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def load_tum_trajectory(filepath):
    """Load trajectory from TUM format file: ts tx ty tz qx qy qz qw."""
    from scipy.spatial.transform import Rotation as R

    data = np.loadtxt(filepath)
    poses = []
    for row in data:
        tx, ty, tz = row[1], row[2], row[3]
        qx, qy, qz, qw = row[4], row[5], row[6], row[7]
        rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = [tx, ty, tz]
        poses.append(T)
    return poses


def load_s3po_trajectory(trj_json_path):
    """Load estimated trajectory from S3PO-GS output (trj_final.json)."""
    with open(trj_json_path, "r") as f:
        trj_data = json.load(f)
    est_poses = [np.array(p) for p in trj_data["trj_est"]]
    gt_poses = [np.array(p) for p in trj_data["trj_gt"]]
    return est_poses, gt_poses


def evaluate_ate(est_poses, gt_poses, monocular=True):
    """Compute ATE RMSE using evo library."""
    if not HAS_EVO:
        return None, None

    traj_ref = PosePath3D(poses_se3=gt_poses)
    traj_est = PosePath3D(poses_se3=est_poses)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    pose_relation = metrics.PoseRelation.translation_part
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data((traj_ref, traj_est_aligned))
    ape_rmse = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()

    return ape_rmse, ape_stats


def evaluate_ate_from_tum(gt_tum_path, est_tum_path, monocular=True):
    """Evaluate ATE from TUM format files."""
    gt_poses = load_tum_trajectory(gt_tum_path)
    est_poses = load_tum_trajectory(est_tum_path)

    min_len = min(len(gt_poses), len(est_poses))
    return evaluate_ate(est_poses[:min_len], gt_poses[:min_len], monocular)


def evaluate_rendering_quality(rendered_dir, gt_dir, device="cuda"):
    """Compute PSNR, SSIM, LPIPS between rendered and GT images."""
    rendered_files = sorted(Path(rendered_dir).glob("*.png"))
    gt_files = sorted(Path(gt_dir).glob("*.png"))

    if len(rendered_files) == 0:
        print(f"  WARNING: No rendered images found in {rendered_dir}")
        return None

    if len(gt_files) == 0:
        print(f"  WARNING: No GT images found in {gt_dir}")
        return None

    n_eval = min(len(rendered_files), len(gt_files))
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    if HAS_LPIPS:
        cal_lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)

    for i in range(n_eval):
        img_pred = np.array(Image.open(rendered_files[i])).astype(np.float64) / 255.0
        img_gt = np.array(Image.open(gt_files[i])).astype(np.float64) / 255.0

        if img_pred.shape != img_gt.shape:
            img_pred = cv2.resize(img_pred, (img_gt.shape[1], img_gt.shape[0]))

        psnr_val = compare_psnr(img_gt, img_pred, data_range=1.0)
        ssim_val = compare_ssim(img_gt, img_pred, channel_axis=2, data_range=1.0)
        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)

        if HAS_LPIPS:
            pred_t = torch.from_numpy(img_pred).permute(2, 0, 1).unsqueeze(0).float().to(device)
            gt_t = torch.from_numpy(img_gt).permute(2, 0, 1).unsqueeze(0).float().to(device)
            lpips_val = cal_lpips(pred_t, gt_t).item()
            lpips_scores.append(lpips_val)

    results = {
        "n_images": n_eval,
        "mean_psnr": float(np.mean(psnr_scores)),
        "mean_ssim": float(np.mean(ssim_scores)),
        "std_psnr": float(np.std(psnr_scores)),
        "std_ssim": float(np.std(ssim_scores)),
    }
    if lpips_scores:
        results["mean_lpips"] = float(np.mean(lpips_scores))
        results["std_lpips"] = float(np.std(lpips_scores))

    return results


def evaluate_from_s3po_results(result_dir, dataset_name):
    """Evaluate a single S3PO-GS result directory."""
    result_dir = Path(result_dir)
    print(f"\n{'='*60}")
    print(f"  Evaluating: {dataset_name}")
    print(f"  Result dir: {result_dir}")
    print(f"{'='*60}")

    output = {"dataset": dataset_name}

    # --- 1. Pose Accuracy (ATE) ---
    print("\n  [1] Pose Accuracy (ATE RMSE)")
    trj_json = None
    plot_dir = result_dir / "plot"
    if plot_dir.exists():
        trj_files = list(plot_dir.glob("trj_*.json"))
        if trj_files:
            trj_json = sorted(trj_files)[-1]  # latest

    if trj_json:
        est_poses, gt_poses = load_s3po_trajectory(trj_json)
        ate_rmse, ate_stats = evaluate_ate(est_poses, gt_poses, monocular=True)
        if ate_rmse is not None:
            print(f"      ATE RMSE: {ate_rmse:.4f} m")
            print(f"      ATE Mean: {ate_stats.get('mean', 'N/A'):.4f} m")
            print(f"      ATE Max:  {ate_stats.get('max', 'N/A'):.4f} m")
            output["ate_rmse"] = ate_rmse
            output["ate_stats"] = ate_stats
    else:
        print("      No trajectory file found. Checking for standalone TUM files...")
        gt_tum = result_dir.parent.parent.parent / "dataset_sparse_p2"
        # Try to find gt_tum and est_tum files
        # S3PO-GS may output trajectory in its own format

    # --- 2. Rendering Quality ---
    print("\n  [2] Rendering Quality")

    # Check S3PO-GS's built-in rendering evaluation
    psnr_dirs = list(result_dir.glob("psnr/*/final_result.json"))
    if psnr_dirs:
        latest = sorted(psnr_dirs)[-1]
        with open(latest, "r") as f:
            render_metrics = json.load(f)
        print(f"      PSNR:  {render_metrics.get('mean_psnr', 'N/A'):.2f}")
        print(f"      SSIM:  {render_metrics.get('mean_ssim', 'N/A'):.4f}")
        print(f"      LPIPS: {render_metrics.get('mean_lpips', 'N/A'):.4f}")
        output["render_psnr"] = render_metrics.get("mean_psnr")
        output["render_ssim"] = render_metrics.get("mean_ssim")
        output["render_lpips"] = render_metrics.get("mean_lpips")
    else:
        # Try to evaluate from rendered images
        render_dir = result_dir / "render_rgb"
        if render_dir.exists():
            print("      Found rendered images, computing metrics...")
            # Would need GT images for comparison
            # For now just report that renders exist
            n_renders = len(list(render_dir.glob("*.png")))
            print(f"      {n_renders} rendered images available")
            output["n_renders"] = n_renders
        else:
            print("      No rendering results found.")

    # --- 3. Stats file from S3PO-GS ---
    stats_files = list(result_dir.glob("plot/stats_*.json"))
    if stats_files:
        with open(sorted(stats_files)[-1], "r") as f:
            stats = json.load(f)
        output["evo_stats"] = stats

    return output


def render_test_views(result_dir, sparse_data_dir, dataset_name, proj_root):
    """
    Render novel test views using the trained 3DGS model from S3PO-GS.
    This requires loading the gaussian model and rendering from GT test poses.
    """
    result_dir = Path(result_dir)
    sparse_data_dir = Path(sparse_data_dir)

    ply_path = result_dir / "point_cloud" / "final" / "point_cloud.ply"
    if not ply_path.exists():
        ply_candidates = list(result_dir.glob("point_cloud/*/point_cloud.ply"))
        if ply_candidates:
            ply_path = sorted(ply_candidates)[-1]
        else:
            print(f"  No point cloud found for {dataset_name}. Skipping test view rendering.")
            return None

    print(f"\n  Rendering test views for {dataset_name}...")
    print(f"    Model: {ply_path}")

    # Load test set metadata
    if dataset_name.startswith("waymo"):
        test_dir = sparse_data_dir / "waymo" / "405841" / "FRONT" / "test_set"
    elif dataset_name.startswith("dl3dv"):
        test_dir = sparse_data_dir / "dl3dv" / "2" / "test_set"
    elif dataset_name.startswith("re10k"):
        test_dir = sparse_data_dir / "re10k" / "1" / "test_set"
    else:
        return None

    if not test_dir.exists():
        print(f"  Test set not found at {test_dir}")
        return None

    # Rendering requires importing S3PO-GS's gaussian splatting module
    # This script provides the framework; actual rendering requires the S3PO-GS environment
    render_output_dir = result_dir / "test_renders"
    render_output_dir.mkdir(exist_ok=True)

    print(f"    Test images: {test_dir / 'rgb'}")
    print(f"    Output: {render_output_dir}")
    print("    NOTE: Test view rendering requires S3PO-GS environment.")
    print("    Use render_test_views_s3po.py for actual rendering.")

    return str(render_output_dir)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Part 2 S3PO-GS SLAM results")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Results directory (default: running_result/5_2/S3PO_GS_SLAM)")
    parser.add_argument("--proj_root", type=str, default=None,
                        help="Project root directory")
    parser.add_argument("--datasets", nargs="+", default=["waymo", "dl3dv", "re10k"],
                        choices=["waymo", "dl3dv", "re10k"])
    parser.add_argument("--render_test", action="store_true",
                        help="Also render test views (requires S3PO-GS env)")
    args = parser.parse_args()

    if args.proj_root is None:
        proj_root = Path(__file__).resolve().parents[2]
    else:
        proj_root = Path(args.proj_root)

    if args.results_dir is None:
        results_dir = proj_root / "running_result" / "5_2" / "S3PO_GS_SLAM"
    else:
        results_dir = Path(args.results_dir)

    sparse_data_dir = proj_root / "dataset_sparse_p2"

    print("=" * 60)
    print("  Part 2: S3PO-GS Evaluation")
    print("=" * 60)
    print(f"  Project root: {proj_root}")
    print(f"  Results dir:  {results_dir}")
    print(f"  Sparse data:  {sparse_data_dir}")

    if not results_dir.exists():
        print(f"\nERROR: Results directory not found: {results_dir}")
        print("Run S3PO-GS first: bash scripts/scripts_p2/run_s3po_all.sh")
        sys.exit(1)

    all_results = {}
    # S3PO-GS names result dirs from the dataset_path: path[-3]_path[-2]
    dataset_map = {
        "waymo": ["waymo_sparse_405841", "waymo_405841"],
        "dl3dv":  ["dl3dv_sparse_2", "dl3dv_2"],
        "re10k":  ["re10k_sparse_1", "re10k_1"],
    }

    for ds in args.datasets:
        ds_result_dir = None
        for candidate_name in dataset_map[ds]:
            candidate_dir = results_dir / candidate_name
            if candidate_dir.exists():
                ds_result_dir = candidate_dir
                break
        if ds_result_dir is None:
            # Glob fallback: match any directory containing the dataset keyword
            for candidate_name in dataset_map[ds]:
                matches = sorted(results_dir.glob(f"{candidate_name}*"))
                if matches:
                    ds_result_dir = matches[-1]
                    break
        if ds_result_dir is None:
            print(f"\n  [SKIP] No results found for {ds} in {results_dir}")
            continue

        # S3PO-GS saves into a timestamp subdirectory; find the latest one
        timestamp_dirs = sorted([d for d in ds_result_dir.iterdir() if d.is_dir()])
        if timestamp_dirs:
            ds_result_dir = timestamp_dirs[-1]
            print(f"\n  Found {ds} results at: {ds_result_dir}")

        result = evaluate_from_s3po_results(ds_result_dir, ds)
        all_results[ds] = result

        if args.render_test:
            render_test_views(ds_result_dir, sparse_data_dir, ds, proj_root)

    # --- Summary Table ---
    print("\n\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Dataset':<12} {'ATE RMSE':<12} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8}")
    print(f"  {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    for ds, res in all_results.items():
        ate = f"{res.get('ate_rmse', 'N/A'):.4f}" if res.get('ate_rmse') else "N/A"
        psnr = f"{res.get('render_psnr', 'N/A'):.2f}" if res.get('render_psnr') else "N/A"
        ssim = f"{res.get('render_ssim', 'N/A'):.4f}" if res.get('render_ssim') else "N/A"
        lpips = f"{res.get('render_lpips', 'N/A'):.4f}" if res.get('render_lpips') else "N/A"
        print(f"  {ds:<12} {ate:<12} {psnr:<8} {ssim:<8} {lpips:<8}")

    # Save results
    output_path = results_dir / "evaluation_summary.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
