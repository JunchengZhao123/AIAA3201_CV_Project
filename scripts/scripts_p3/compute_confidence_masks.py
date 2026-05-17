#!/usr/bin/env python3
"""
Part 3: Confidence Mask Computation for Pseudo-Views.

Implements multi-signal confidence inference:
  1. RIFE cross-check: compare the pseudo-view against the RIFE-interpolated
     frame (same pair, same alpha) to detect hallucinations / blending errors.
  2. Patch-SSIM photometric consistency: structural similarity between the
     pseudo-view and the warped reference views (more robust than raw pixel diff).
  3. Depth warp consistency: reproject reference view via depth + pose,
     measure geometric agreement.
  4. Normalized pose consistency: soft weighting by relative translation,
     normalized by the dataset's typical inter-frame distance.

Usage:
    python compute_confidence_masks.py \
        --pseudo_dir running_result/5_3/pseudo_views/waymo_405841 \
        --output_dir running_result/5_3/confidence_masks/waymo_405841

    python compute_confidence_masks.py --config scripts/scripts_p3/configs/waymo_p3.yaml
"""

import argparse
import json
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


def photometric_confidence(pseudo_img, ref_img, alpha=5.0):
    """
    Compute per-pixel photometric confidence between pseudo and reference.
    C(p) = exp(-alpha * |pseudo(p) - ref(p)|)
    """
    diff = np.abs(pseudo_img.astype(np.float32) - ref_img.astype(np.float32)) / 255.0
    diff_gray = np.mean(diff, axis=2)
    confidence = np.exp(-alpha * diff_gray)
    return confidence


def ssim_confidence(pseudo_img, ref_img, win_size=11):
    """
    Per-pixel confidence based on local structural similarity (SSIM).

    Unlike raw pixel diff, SSIM is robust to small spatial shifts (parallax)
    and uniform brightness changes — exactly the issues that plague direct
    photometric comparison between different viewpoints.

    Returns (H, W) array in [0, 1].
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    a = pseudo_img.astype(np.float64)
    b = ref_img.astype(np.float64)

    if a.ndim == 3:
        a = np.mean(a, axis=2)
    if b.ndim == 3:
        b = np.mean(b, axis=2)

    k = cv2.getGaussianKernel(win_size, 1.5)
    window = k @ k.T

    mu_a = cv2.filter2D(a, -1, window)
    mu_b = cv2.filter2D(b, -1, window)
    mu_a_sq = mu_a ** 2
    mu_b_sq = mu_b ** 2
    mu_ab = mu_a * mu_b

    sig_a_sq = cv2.filter2D(a ** 2, -1, window) - mu_a_sq
    sig_b_sq = cv2.filter2D(b ** 2, -1, window) - mu_b_sq
    sig_ab = cv2.filter2D(a * b, -1, window) - mu_ab

    ssim_map = ((2 * mu_ab + C1) * (2 * sig_ab + C2)) / \
               ((mu_a_sq + mu_b_sq + C1) * (sig_a_sq + sig_b_sq + C2))

    return np.clip((ssim_map + 1.0) / 2.0, 0.0, 1.0)


def optical_flow_fb_consistency(img_a, img_b, sigma=3.0):
    """
    Forward-backward optical flow consistency check.

    Computes dense optical flow A->B and B->A, then measures the forward-
    backward error at each pixel.  Low error means the correspondence is
    reliable (both directions agree), high error indicates occlusion,
    dis-occlusion, or hallucinated content.

    Returns (H, W) confidence in [0, 1] (in img_a's coordinate frame).
    """
    if img_a.ndim == 3:
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    else:
        gray_a = img_a
    if img_b.ndim == 3:
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    else:
        gray_b = img_b

    flow_ab = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    flow_ba = cv2.calcOpticalFlowFarneback(
        gray_b, gray_a, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    H, W = gray_a.shape
    coords = np.stack(
        np.meshgrid(np.arange(W, dtype=np.float32),
                     np.arange(H, dtype=np.float32)),
        axis=-1,
    )

    warped_x = (coords[:, :, 0] + flow_ab[:, :, 0]).astype(np.float32)
    warped_y = (coords[:, :, 1] + flow_ab[:, :, 1]).astype(np.float32)

    flow_ba_x_back = cv2.remap(
        flow_ba[:, :, 0], warped_x, warped_y,
        cv2.INTER_LINEAR, borderValue=0,
    )
    flow_ba_y_back = cv2.remap(
        flow_ba[:, :, 1], warped_x, warped_y,
        cv2.INTER_LINEAR, borderValue=0,
    )

    fb_error = np.sqrt(
        (flow_ab[:, :, 0] + flow_ba_x_back) ** 2
        + (flow_ab[:, :, 1] + flow_ba_y_back) ** 2
    )

    return np.exp(-fb_error / sigma).astype(np.float32)


def reprojection_error_confidence(
    pseudo_img, ref_img, ref_pose_c2w, pseudo_pose_c2w, intrinsics, sigma=10.0,
):
    """
    Reprojection-error-based confidence using optical-flow warping.

    Warps the reference image toward the pseudo-view's viewpoint via dense
    optical flow, then measures per-pixel agreement.  Unlike depth-based
    warping this works even when no depth map is available.

    Returns (H, W) confidence in [0, 1].
    """
    if pseudo_img.ndim == 3:
        gray_p = cv2.cvtColor(pseudo_img, cv2.COLOR_RGB2GRAY)
    else:
        gray_p = pseudo_img
    if ref_img.ndim == 3:
        gray_r = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    else:
        gray_r = ref_img

    flow = cv2.calcOpticalFlowFarneback(
        gray_r, gray_p, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    H, W = gray_r.shape
    coords = np.stack(
        np.meshgrid(np.arange(W, dtype=np.float32),
                     np.arange(H, dtype=np.float32)),
        axis=-1,
    )
    map_x = (coords[:, :, 0] + flow[:, :, 0]).astype(np.float32)
    map_y = (coords[:, :, 1] + flow[:, :, 1]).astype(np.float32)

    warped_ref = cv2.remap(
        ref_img, map_x, map_y, cv2.INTER_LINEAR, borderValue=0,
    )

    diff = np.abs(pseudo_img.astype(np.float32) - warped_ref.astype(np.float32))
    if diff.ndim == 3:
        diff = np.mean(diff, axis=2)
    diff /= 255.0

    return np.exp(-diff * sigma).astype(np.float32)


def warp_image(src_img, src_depth, src_pose_c2w, dst_pose_c2w, intrinsics):
    """
    Warp source image to destination view using depth and relative pose.

    Args:
        src_img: (H, W, 3) source image
        src_depth: (H, W) source depth map
        src_pose_c2w: 4x4 camera-to-world of source
        dst_pose_c2w: 4x4 camera-to-world of destination
        intrinsics: dict with fx, fy, cx, cy
    Returns:
        warped_img: (H, W, 3) warped image
        valid_mask: (H, W) bool mask of valid pixels
    """
    H, W = src_depth.shape[:2]
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    d = src_depth.astype(np.float32)

    valid_depth = d > 0.01
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    z = d

    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)  # (H, W, 4)
    pts_cam_flat = pts_cam.reshape(-1, 4).T  # (4, N)

    # Source camera to world to destination camera
    T_rel = np.linalg.inv(dst_pose_c2w) @ src_pose_c2w  # w2c_dst @ c2w_src
    pts_dst = T_rel @ pts_cam_flat  # (4, N)
    pts_dst = pts_dst[:3].T.reshape(H, W, 3)

    # Project to destination image
    z_dst = pts_dst[:, :, 2]
    u_dst = (pts_dst[:, :, 0] * fx / (z_dst + 1e-8)) + cx
    v_dst = (pts_dst[:, :, 1] * fy / (z_dst + 1e-8)) + cy

    # Bilinear sampling
    u_dst_norm = 2.0 * u_dst / (W - 1) - 1.0
    v_dst_norm = 2.0 * v_dst / (H - 1) - 1.0
    grid = np.stack([u_dst_norm, v_dst_norm], axis=-1)

    src_tensor = torch.from_numpy(src_img.transpose(2, 0, 1)).float().unsqueeze(0)
    grid_tensor = torch.from_numpy(grid).float().unsqueeze(0)

    warped = F.grid_sample(src_tensor, grid_tensor, mode="bilinear",
                           padding_mode="zeros", align_corners=True)
    warped_img = warped.squeeze(0).numpy().transpose(1, 2, 0).astype(np.uint8)

    valid_mask = (valid_depth &
                  (z_dst > 0.01) &
                  (u_dst >= 0) & (u_dst < W) &
                  (v_dst >= 0) & (v_dst < H))
    return warped_img, valid_mask


def depth_consistency_score(depth_src, depth_warped, epsilon=1e-6):
    """
    Per-pixel depth consistency: s_d(p) = exp(-|d_a - d_b| / ((d_a + d_b)/2 + eps))
    Following BRPO Eq. 5.
    """
    d_sum = depth_src + depth_warped
    d_diff = np.abs(depth_src - depth_warped)
    score = np.exp(-d_diff / (d_sum / 2 + epsilon))
    score[depth_src < 0.01] = 0
    score[depth_warped < 0.01] = 0
    return score


def pose_consistency_score(pose_a, pose_b, baseline_dist=None):
    """
    Pose consistency scalar, normalized by the typical inter-frame distance.

    The old formula exp(-||t||) decays too aggressively for driving scenes where
    translations can be several meters.  Instead we normalize:
        s_t = exp(-||t_a - t_b|| / sigma)
    where sigma = baseline_dist (median inter-frame distance) so that adjacent
    frames get scores near ~0.37 instead of near-zero.
    """
    t_a = pose_a[:3, 3]
    t_b = pose_b[:3, 3]
    dist = float(np.linalg.norm(t_a - t_b))
    sigma = max(baseline_dist or 1.0, 0.01)
    return float(np.exp(-dist / sigma))


def overlap_score_fusion(pseudo_img, candidates, confidences_per_candidate, epsilon=1e-8):
    """
    Fuse multiple candidate restorations using overlap confidence.
    Following BRPO Eq. 7-8: I_fix = I + W_1 * r_1 + W_2 * r_2
    """
    assert len(candidates) == len(confidences_per_candidate)

    pseudo_float = pseudo_img.astype(np.float32)
    fused = pseudo_float.copy()

    conf_sum = np.zeros(pseudo_img.shape[:2], dtype=np.float32)
    for conf in confidences_per_candidate:
        conf_sum += conf

    for cand, conf in zip(candidates, confidences_per_candidate):
        weight = conf / (conf_sum + epsilon)
        residual = cand.astype(np.float32) - pseudo_float
        fused += weight[:, :, None] * residual

    return np.clip(fused, 0, 255).astype(np.uint8)


def compute_confidence_for_pseudo_view(
    pseudo_meta_entry,
    sparse_frames,
    intrinsics,
    baseline_dist=None,
    alpha_photo=5.0,
):
    """
    Compute the confidence mask for a single pseudo-view.

    Signals combined (higher = more trustworthy):
      1. RIFE cross-check — SSIM between pseudo-view and RIFE-interpolated
         frame.  High agreement → content matches simple optical-flow warping
         of real images → reliable.
      2. Optical-flow forward-backward consistency — dense flow from each
         adjacent real frame to the pseudo-view and back.  Pixels where
         the round-trip displacement is small are geometrically consistent
         (addresses temporal flickering / dis-occlusion artifacts).
      3. Reprojection error via flow warping — warp each reference image
         toward the pseudo-view using optical flow, compare appearance.
         Works without depth and directly quantifies how well the pseudo
         content can be explained by real observations.
      4. Depth-based warped-reference SSIM (when depth is available).
      5. Normalized pose consistency — soft weight favouring pseudo-views
         that sit between nearby keyframes.

    Returns:
        confidence_mask: (H, W) array in [0, 1]
    """
    pseudo_path = pseudo_meta_entry["pseudo_path"]
    rife_path = pseudo_meta_entry.get("rife_path")
    depth_path = pseudo_meta_entry.get("depth_path")
    pseudo_pose = np.array(pseudo_meta_entry["pose"])
    pair_idxs = pseudo_meta_entry["pair"]  # [i, i+1]

    pseudo_img = np.array(Image.open(pseudo_path))
    H, W = pseudo_img.shape[:2]

    # ---- Signal 1: RIFE cross-check ----
    rife_conf = None
    if rife_path and os.path.exists(rife_path):
        rife_img = np.array(Image.open(rife_path))
        if rife_img.shape[:2] != (H, W):
            rife_img = cv2.resize(rife_img, (W, H))
        rife_conf = ssim_confidence(pseudo_img, rife_img, win_size=11)

    # Load depth if available
    pseudo_depth = None
    if depth_path and os.path.exists(depth_path):
        pseudo_depth = np.load(depth_path)
        if pseudo_depth.shape[:2] != (H, W):
            pseudo_depth = cv2.resize(pseudo_depth, (W, H),
                                       interpolation=cv2.INTER_NEAREST)

    # ---- Signals 2-5: Per-reference confidence ----
    ref_confs = []
    flow_confs = []
    reproj_confs = []

    for ref_idx in pair_idxs:
        if ref_idx >= len(sparse_frames):
            continue

        ref_data = sparse_frames[ref_idx]
        ref_img = np.array(Image.open(ref_data["image_path"]))
        ref_pose = np.array(ref_data["pose"])

        if ref_img.shape[:2] != (H, W):
            ref_img = cv2.resize(ref_img, (W, H))

        # Signal 2: Optical-flow forward-backward consistency
        fb_conf = optical_flow_fb_consistency(pseudo_img, ref_img, sigma=3.0)
        flow_confs.append(fb_conf)

        # Signal 3: Reprojection error via flow warping
        rp_conf = reprojection_error_confidence(
            pseudo_img, ref_img, ref_pose, pseudo_pose, intrinsics, sigma=10.0,
        )
        reproj_confs.append(rp_conf)

        # Signal 4: Depth-based warped-reference SSIM
        if pseudo_depth is not None:
            warped_ref, warp_valid = warp_image(
                ref_img, pseudo_depth, pseudo_pose, ref_pose, intrinsics
            )
            warp_ssim = ssim_confidence(pseudo_img, warped_ref, win_size=11)
            warp_ssim[~warp_valid] = 0.0

            direct_ssim = ssim_confidence(pseudo_img, ref_img, win_size=11)
            per_ref = np.maximum(warp_ssim, direct_ssim * 0.7)
        else:
            per_ref = ssim_confidence(pseudo_img, ref_img, win_size=11)

        # Signal 5: Pose consistency (normalized)
        s_t = pose_consistency_score(pseudo_pose, ref_pose,
                                      baseline_dist=baseline_dist)
        per_ref *= s_t

        ref_confs.append(per_ref)

    # ---- Fuse per-reference signals (take best across references) ----
    def _fuse_list(lst, fallback_val=0.5):
        if len(lst) == 0:
            return np.full((H, W), fallback_val, dtype=np.float32)
        if len(lst) == 1:
            return lst[0]
        return np.maximum(lst[0], lst[1])

    ref_combined = _fuse_list(ref_confs)
    flow_combined = _fuse_list(flow_confs)
    reproj_combined = _fuse_list(reproj_confs)

    # ---- Final fusion ----
    # Weighted combination: flow-based signals address temporal flickering &
    # geometric drift; SSIM-based signals address appearance fidelity.
    consistency_score = 0.5 * flow_combined + 0.5 * reproj_combined
    appearance_score = ref_combined

    if rife_conf is not None:
        confidence = (0.35 * rife_conf
                      + 0.35 * consistency_score
                      + 0.30 * appearance_score)
    else:
        confidence = 0.50 * consistency_score + 0.50 * appearance_score

    confidence = cv2.GaussianBlur(confidence.astype(np.float32), (7, 7), 1.5)
    return np.clip(confidence, 0.0, 1.0)


def _compute_median_baseline(sparse_frames):
    """Median translation distance between consecutive sparse keyframes."""
    dists = []
    for i in range(len(sparse_frames) - 1):
        t_a = np.array(sparse_frames[i]["pose"])[:3, 3]
        t_b = np.array(sparse_frames[i + 1]["pose"])[:3, 3]
        dists.append(float(np.linalg.norm(t_a - t_b)))
    return float(np.median(dists)) if dists else 1.0


def compute_all_confidence_masks(pseudo_dir, output_dir, alpha_photo=5.0):
    """Compute confidence masks for all pseudo-views in a scene."""
    pseudo_dir = Path(pseudo_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = pseudo_dir / "pseudo_meta.json"
    if not meta_path.exists():
        print(f"ERROR: Metadata not found at {meta_path}")
        print("Run generate_pseudo_views.py first.")
        sys.exit(1)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    intrinsics = meta["intrinsics"]
    sparse_frames = meta["sparse_frames"]
    pseudo_frames = meta["pseudo_frames"]

    baseline_dist = _compute_median_baseline(sparse_frames)
    print(f"  Computing confidence masks for {len(pseudo_frames)} pseudo-views...")
    print(f"  Reference: {len(sparse_frames)} sparse frames")
    print(f"  Median inter-frame baseline: {baseline_dist:.3f}")

    mask_paths = []
    conf_stats = []
    for entry in tqdm(pseudo_frames, desc="Confidence masks"):
        idx = entry["pseudo_idx"]
        confidence = compute_confidence_for_pseudo_view(
            entry, sparse_frames, intrinsics,
            baseline_dist=baseline_dist,
            alpha_photo=alpha_photo,
        )

        mask_path = output_dir / f"{idx:06d}_conf.npy"
        np.save(mask_path, confidence)

        vis_path = output_dir / f"{idx:06d}_conf_vis.png"
        vis = (confidence * 255).astype(np.uint8)
        cv2.imwrite(str(vis_path), cv2.applyColorMap(vis, cv2.COLORMAP_JET))

        mask_paths.append(str(mask_path))
        conf_stats.append({
            "pseudo_idx": idx,
            "pair": entry["pair"],
            "mean_conf": float(np.mean(confidence)),
            "median_conf": float(np.median(confidence)),
        })

    # Save summary
    summary = {
        "n_masks": len(mask_paths),
        "alpha_photo": alpha_photo,
        "baseline_dist": baseline_dist,
        "mask_paths": mask_paths,
        "per_frame_stats": conf_stats,
        "source_meta": str(meta_path),
    }
    with open(output_dir / "confidence_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved {len(mask_paths)} confidence masks to: {output_dir}")
    for s in conf_stats:
        print(f"    Pseudo {s['pseudo_idx']} (pair {s['pair']}): "
              f"mean={s['mean_conf']:.3f}, median={s['median_conf']:.3f}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Part 3: Compute confidence masks for pseudo-views"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--pseudo_dir", type=str, default=None,
                        help="Directory with pseudo-views and pseudo_meta.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for confidence masks")
    parser.add_argument("--alpha_photo", type=float, default=5.0,
                        help="Photometric confidence sensitivity")
    parser.add_argument("--proj_root", type=str, default=None)
    parser.add_argument("--dataset_type", type=str, default="waymo",
                        choices=["waymo", "dl3dv", "re10k"])
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

    ds_names = {"waymo": "waymo_405841", "dl3dv": "dl3dv_2", "re10k": "re10k_1"}
    ds_name = ds_names[args.dataset_type]

    if args.pseudo_dir is None:
        args.pseudo_dir = str(proj / "running_result" / "5_3" / "pseudo_views" / ds_name)

    if args.output_dir is None:
        args.output_dir = str(proj / "running_result" / "5_3" / "confidence_masks" / ds_name)

    compute_all_confidence_masks(
        pseudo_dir=args.pseudo_dir,
        output_dir=args.output_dir,
        alpha_photo=args.alpha_photo,
    )


if __name__ == "__main__":
    main()
