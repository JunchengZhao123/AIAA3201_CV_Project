#!/usr/bin/env python3
"""
Part 2: Sub-sample dense datasets to sparse frames for S3PO-GS SLAM.

Creates S3PO-GS-compatible directory structures with:
  - Waymo-405841: 1/10 sparsity (every 10th frame)
  - DL3DV-2:     1/30 sparsity (every 30th frame)
  - Re10k-1:     1/30 sparsity (every 30th frame)

Also exports GT poses in TUM format for standalone ATE evaluation with `evo`.

Usage:
    python subsample_sparse.py [--proj_root /path/to/project] [--output_root /path/to/output]
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_trans_to_4x4(quat_xyzw, trans):
    """Convert quaternion (x,y,z,w) + translation to 4x4 matrix."""
    rot = R.from_quat(quat_xyzw).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T


def mat4x4_to_tum(timestamp, pose_c2w):
    """Convert 4x4 camera-to-world pose to TUM format line: ts tx ty tz qx qy qz qw."""
    trans = pose_c2w[:3, 3]
    quat = R.from_matrix(pose_c2w[:3, :3]).as_quat()  # xyzw
    return f"{timestamp:.6f} {trans[0]:.8f} {trans[1]:.8f} {trans[2]:.8f} {quat[0]:.8f} {quat[1]:.8f} {quat[2]:.8f} {quat[3]:.8f}"


def subsample_waymo(proj_root, output_root, sparsity=10, downscale=1):
    """Sub-sample Waymo-405841 to S3PO-GS format.

    Args:
        downscale: integer factor to downsample images (2 = half resolution).
                   Reduces training cost while preserving more geometric detail
                   than aggressive sparsity.
    """
    src_dir = proj_root / "dataset" / "405841" / "FRONT"
    out_dir = output_root / "waymo" / "405841" / "FRONT"

    src_images = sorted((src_dir / "images").glob("*.png"))
    src_gts = sorted((src_dir / "gt").glob("*.txt"))
    src_depths = sorted((src_dir / "depth").glob("*.png"))

    n_total = len(src_images)
    sparse_indices = list(range(0, n_total, sparsity))
    test_indices = [i for i in range(n_total) if i not in sparse_indices]

    print(f"[Waymo-405841] Total: {n_total}, Sparse (1/{sparsity}): {len(sparse_indices)}, Test: {len(test_indices)}")
    if downscale > 1:
        print(f"  Downscaling images by {downscale}x")

    for subdir in ["rgb", "gt", "depth", "mono_depth"]:
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    for new_idx, orig_idx in enumerate(sparse_indices):
        fname = f"{new_idx:06d}.png"
        fname_txt = f"{new_idx:06d}.txt"

        if downscale > 1:
            from PIL import Image as PILImage
            img = PILImage.open(src_images[orig_idx])
            new_w, new_h = img.width // downscale, img.height // downscale
            img.resize((new_w, new_h), PILImage.LANCZOS).save(out_dir / "rgb" / fname)

            depth = PILImage.open(src_depths[orig_idx])
            depth.resize((new_w, new_h), PILImage.NEAREST).save(out_dir / "depth" / fname)
            depth.resize((new_w, new_h), PILImage.NEAREST).save(out_dir / "mono_depth" / fname)
        else:
            shutil.copy2(src_images[orig_idx], out_dir / "rgb" / fname)
            shutil.copy2(src_depths[orig_idx], out_dir / "depth" / fname)
            shutil.copy2(src_depths[orig_idx], out_dir / "mono_depth" / fname)

        shutil.copy2(src_gts[orig_idx], out_dir / "gt" / fname_txt)

    # Copy calibration (same for all frames)
    shutil.copy2(src_dir / "calib" / "000000.txt", out_dir / "calib.txt")

    # Save test set info for evaluation
    test_dir = out_dir / "test_set"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "rgb").mkdir(exist_ok=True)

    for new_idx, orig_idx in enumerate(test_indices):
        fname = f"{new_idx:06d}.png"
        fname_txt = f"{new_idx:06d}.txt"
        if downscale > 1:
            from PIL import Image as PILImage
            img = PILImage.open(src_images[orig_idx])
            new_w, new_h = img.width // downscale, img.height // downscale
            img.resize((new_w, new_h), PILImage.LANCZOS).save(test_dir / "rgb" / fname)
        else:
            shutil.copy2(src_images[orig_idx], test_dir / "rgb" / fname)

    # Save GT poses in TUM format (for evo evaluation)
    _export_waymo_tum_poses(src_gts, sparse_indices, out_dir / "gt_tum_sparse.txt")
    _export_waymo_tum_poses(src_gts, test_indices, out_dir / "gt_tum_test.txt")
    _export_waymo_tum_poses(src_gts, list(range(n_total)), out_dir / "gt_tum_all.txt")

    # Save test GT poses as 4x4 matrices (for rendering evaluation)
    test_gt_dir = test_dir / "gt"
    test_gt_dir.mkdir(exist_ok=True)
    for new_idx, orig_idx in enumerate(test_indices):
        shutil.copy2(src_gts[orig_idx], test_gt_dir / f"{new_idx:06d}.txt")

    # Save index mappings
    meta = {
        "dataset": "waymo-405841",
        "sparsity": sparsity,
        "n_total": n_total,
        "n_sparse": len(sparse_indices),
        "n_test": len(test_indices),
        "sparse_indices": sparse_indices,
        "test_indices": test_indices,
    }
    with open(out_dir / "split_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  -> Output: {out_dir}")


def _export_waymo_tum_poses(gt_files, indices, out_path):
    """Export Waymo GT poses (4x4 matrices) to TUM format."""
    lines = []
    for i, idx in enumerate(indices):
        pose = np.loadtxt(gt_files[idx], delimiter=" ").reshape(4, 4)
        # pose is cam-to-world in Waymo format
        lines.append(mat4x4_to_tum(float(i), pose))
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def subsample_dl3dv(proj_root, output_root, sparsity=30):
    """Sub-sample DL3DV-2 to S3PO-GS format."""
    src_dir = proj_root / "dataset" / "DL3DV-2"
    out_dir = output_root / "dl3dv" / "2"

    with open(src_dir / "cameras.json", "r") as f:
        all_cameras = json.load(f)

    src_images = sorted((src_dir / "images").glob("*.png"))
    n_total = len(src_images)
    sparse_indices = list(range(0, n_total, sparsity))
    test_indices = [i for i in range(n_total) if i not in sparse_indices]

    print(f"[DL3DV-2] Total: {n_total}, Sparse (1/{sparsity}): {len(sparse_indices)}, Test: {len(test_indices)}")

    (out_dir / "rgb").mkdir(parents=True, exist_ok=True)

    # Create sparse images (renamed to sequential for S3PO-GS dl3dv parser)
    sparse_cameras = []
    for new_idx, orig_idx in enumerate(sparse_indices):
        fname = f"frame_{new_idx+1:05d}.png"
        shutil.copy2(src_images[orig_idx], out_dir / "rgb" / fname)
        cam = all_cameras[orig_idx].copy()
        cam["cam_id"] = new_idx + 1
        cam["image_name"] = fname
        sparse_cameras.append(cam)

    with open(out_dir / "cameras.json", "w") as f:
        json.dump(sparse_cameras, f, indent=4)

    # Save test set
    test_dir = out_dir / "test_set"
    (test_dir / "rgb").mkdir(parents=True, exist_ok=True)
    test_cameras = []
    for new_idx, orig_idx in enumerate(test_indices):
        fname = f"frame_{new_idx+1:05d}.png"
        shutil.copy2(src_images[orig_idx], test_dir / "rgb" / fname)
        cam = all_cameras[orig_idx].copy()
        cam["cam_id"] = new_idx + 1
        cam["image_name"] = fname
        test_cameras.append(cam)

    with open(test_dir / "cameras.json", "w") as f:
        json.dump(test_cameras, f, indent=4)

    # Export TUM format poses
    _export_dl3dv_tum_poses(all_cameras, sparse_indices, out_dir / "gt_tum_sparse.txt")
    _export_dl3dv_tum_poses(all_cameras, test_indices, out_dir / "gt_tum_test.txt")
    _export_dl3dv_tum_poses(all_cameras, list(range(n_total)), out_dir / "gt_tum_all.txt")

    # Copy intrinsics
    shutil.copy2(src_dir / "intrinsics.json", out_dir / "intrinsics.json")

    meta = {
        "dataset": "dl3dv-2",
        "sparsity": sparsity,
        "n_total": n_total,
        "n_sparse": len(sparse_indices),
        "n_test": len(test_indices),
        "sparse_indices": sparse_indices,
        "test_indices": test_indices,
    }
    with open(out_dir / "split_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  -> Output: {out_dir}")


def _export_dl3dv_tum_poses(cameras, indices, out_path):
    """Export DL3DV/Re10k camera poses (quat+trans) to TUM format."""
    lines = []
    for i, idx in enumerate(indices):
        cam = cameras[idx]
        qx, qy, qz, qw = cam["cam_quat"]
        tx, ty, tz = cam["cam_trans"]
        rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
        # cameras.json stores world-to-camera transform; invert for c2w
        T_w2c = np.eye(4)
        T_w2c[:3, :3] = rot
        T_w2c[:3, 3] = [tx, ty, tz]
        T_c2w = np.linalg.inv(T_w2c)
        lines.append(mat4x4_to_tum(float(i), T_c2w))
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def subsample_re10k(proj_root, output_root, sparsity=30):
    """Sub-sample Re10k-1 to S3PO-GS format (uses DL3DV-compatible layout)."""
    src_dir = proj_root / "dataset" / "Re10k-1"
    out_dir = output_root / "re10k" / "1"

    with open(src_dir / "cameras.json", "r") as f:
        all_cameras = json.load(f)

    src_images = sorted((src_dir / "images").glob("*.png"))
    n_total = len(src_images)
    sparse_indices = list(range(0, n_total, sparsity))
    test_indices = [i for i in range(n_total) if i not in sparse_indices]

    print(f"[Re10k-1] Total: {n_total}, Sparse (1/{sparsity}): {len(sparse_indices)}, Test: {len(test_indices)}")

    (out_dir / "rgb").mkdir(parents=True, exist_ok=True)

    sparse_cameras = []
    for new_idx, orig_idx in enumerate(sparse_indices):
        fname = f"frame_{new_idx+1:05d}.png"
        shutil.copy2(src_images[orig_idx], out_dir / "rgb" / fname)
        cam = all_cameras[orig_idx].copy()
        cam["cam_id"] = new_idx + 1
        cam["image_name"] = fname
        sparse_cameras.append(cam)

    with open(out_dir / "cameras.json", "w") as f:
        json.dump(sparse_cameras, f, indent=4)

    # Save test set
    test_dir = out_dir / "test_set"
    (test_dir / "rgb").mkdir(parents=True, exist_ok=True)
    test_cameras = []
    for new_idx, orig_idx in enumerate(test_indices):
        fname = f"frame_{new_idx+1:05d}.png"
        shutil.copy2(src_images[orig_idx], test_dir / "rgb" / fname)
        cam = all_cameras[orig_idx].copy()
        cam["cam_id"] = new_idx + 1
        cam["image_name"] = fname
        test_cameras.append(cam)

    with open(test_dir / "cameras.json", "w") as f:
        json.dump(test_cameras, f, indent=4)

    # Export TUM poses
    _export_dl3dv_tum_poses(all_cameras, sparse_indices, out_dir / "gt_tum_sparse.txt")
    _export_dl3dv_tum_poses(all_cameras, test_indices, out_dir / "gt_tum_test.txt")
    _export_dl3dv_tum_poses(all_cameras, list(range(n_total)), out_dir / "gt_tum_all.txt")

    shutil.copy2(src_dir / "intrinsics.json", out_dir / "intrinsics.json")

    meta = {
        "dataset": "re10k-1",
        "sparsity": sparsity,
        "n_total": n_total,
        "n_sparse": len(sparse_indices),
        "n_test": len(test_indices),
        "sparse_indices": sparse_indices,
        "test_indices": test_indices,
    }
    with open(out_dir / "split_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  -> Output: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sub-sample datasets for Part 2 sparse reconstruction")
    parser.add_argument("--proj_root", type=str, default=None,
                        help="Project root directory (auto-detected if not set)")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Output root for sparse datasets (default: {proj_root}/dataset_sparse_p2)")
    parser.add_argument("--datasets", nargs="+", default=["waymo", "dl3dv", "re10k"],
                        choices=["waymo", "dl3dv", "re10k"],
                        help="Which datasets to process")
    parser.add_argument("--waymo_sparsity", type=int, default=10,
                        help="Waymo sparsity factor (default: 10, try 5 for better quality)")
    parser.add_argument("--downscale", type=int, default=1,
                        help="Image downscale factor for Waymo (2 = half resolution)")
    args = parser.parse_args()

    if args.proj_root is None:
        proj_root = Path(__file__).resolve().parents[2]
    else:
        proj_root = Path(args.proj_root)

    if args.output_root is None:
        output_root = proj_root / "dataset_sparse_p2"
    else:
        output_root = Path(args.output_root)

    print(f"Project root: {proj_root}")
    print(f"Output root:  {output_root}")
    print("=" * 60)

    if "waymo" in args.datasets:
        subsample_waymo(proj_root, output_root, sparsity=args.waymo_sparsity,
                        downscale=args.downscale)
    if "dl3dv" in args.datasets:
        subsample_dl3dv(proj_root, output_root, sparsity=30)
    if "re10k" in args.datasets:
        subsample_re10k(proj_root, output_root, sparsity=30)

    print("=" * 60)
    print("Done! Sparse datasets created at:", output_root)
    print("\nNext steps:")
    print("  1. Copy sparse datasets to the S3PO-GS/datasets/ directory")
    print("  2. Run: bash scripts/scripts_p2/run_s3po_all.sh")


if __name__ == "__main__":
    main()
