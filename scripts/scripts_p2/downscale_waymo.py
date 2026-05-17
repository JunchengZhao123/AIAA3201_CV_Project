#!/usr/bin/env python3
"""
Downscale Waymo sparse images to reduce VRAM usage for S3PO-GS.

Original: 1920x1280 (~48GB VRAM on A6000 — too much)
Scale 2:   960x640  (~12-15GB VRAM — comfortable)
Scale 4:   480x320  (~4-6GB VRAM — fast but lower quality)

Usage:
    python downscale_waymo.py                          # default 2x downscale
    python downscale_waymo.py --scale 4                # 4x downscale
    python downscale_waymo.py --input_dir /path/to/sparse --scale 2
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def downscale_images(input_dir, output_dir, scale):
    """Downscale all PNG images in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.png"))
    print(f"  Downscaling {len(files)} images by {scale}x ...")
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        new_h, new_w = h // scale, w // scale
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(output_dir / f.name), img_resized)
    print(f"  -> {files[0].name} ... {files[-1].name}  ({w}x{h} -> {new_w}x{new_h})")


def main():
    parser = argparse.ArgumentParser(description="Downscale Waymo sparse images")
    parser.add_argument("--proj_root", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Sparse Waymo directory (default: dataset_sparse_p2/waymo/405841/FRONT)")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4],
                        help="Downscale factor (default: 2)")
    args = parser.parse_args()

    if args.proj_root is None:
        proj_root = Path(__file__).resolve().parents[2]
    else:
        proj_root = Path(args.proj_root)

    if args.input_dir is None:
        base_dir = proj_root / "dataset_sparse_p2" / "waymo" / "405841" / "FRONT"
    else:
        base_dir = Path(args.input_dir)

    scale = args.scale

    # Original calibration
    orig_fx = 2066.697564417299
    orig_fy = 2066.697564417299
    orig_cx = 950.5512774150723
    orig_cy = 641.1870541472169
    orig_w, orig_h = 1920, 1280

    new_w, new_h = orig_w // scale, orig_h // scale
    new_fx = orig_fx / scale
    new_fy = orig_fy / scale
    new_cx = orig_cx / scale
    new_cy = orig_cy / scale

    print(f"Downscaling Waymo sparse data by {scale}x")
    print(f"  Resolution: {orig_w}x{orig_h} -> {new_w}x{new_h}")
    print(f"  fx: {orig_fx:.2f} -> {new_fx:.2f}")
    print(f"  fy: {orig_fy:.2f} -> {new_fy:.2f}")
    print(f"  cx: {orig_cx:.2f} -> {new_cx:.2f}")
    print(f"  cy: {orig_cy:.2f} -> {new_cy:.2f}")
    print()

    # Downscale RGB
    rgb_dir = base_dir / "rgb"
    if rgb_dir.exists():
        downscale_images(rgb_dir, base_dir / f"rgb_{scale}x", scale)
        # Swap directories: backup original, make downscaled the active one
        rgb_backup = base_dir / "rgb_original"
        if not rgb_backup.exists():
            rgb_dir.rename(rgb_backup)
            (base_dir / f"rgb_{scale}x").rename(rgb_dir)
            print(f"  Original rgb -> rgb_original, downscaled -> rgb")
        else:
            print(f"  rgb_original already exists, keeping rgb_{scale}x as separate dir")
    else:
        print(f"  WARNING: {rgb_dir} not found")

    # Downscale depth
    depth_dir = base_dir / "depth"
    if depth_dir.exists():
        downscale_images(depth_dir, base_dir / f"depth_{scale}x", scale)
        depth_backup = base_dir / "depth_original"
        if not depth_backup.exists():
            depth_dir.rename(depth_backup)
            (base_dir / f"depth_{scale}x").rename(depth_dir)
            print(f"  Original depth -> depth_original, downscaled -> depth")
        else:
            print(f"  depth_original already exists, keeping depth_{scale}x as separate dir")

    # Downscale mono_depth
    mono_dir = base_dir / "mono_depth"
    if mono_dir.exists():
        downscale_images(mono_dir, base_dir / f"mono_depth_{scale}x", scale)
        mono_backup = base_dir / "mono_depth_original"
        if not mono_backup.exists():
            mono_dir.rename(mono_backup)
            (base_dir / f"mono_depth_{scale}x").rename(mono_dir)
            print(f"  Original mono_depth -> mono_depth_original, downscaled -> mono_depth")
        else:
            print(f"  mono_depth_original already exists")

    # Downscale test set images
    test_rgb = base_dir / "test_set" / "rgb"
    if test_rgb.exists():
        downscale_images(test_rgb, base_dir / "test_set" / f"rgb_{scale}x", scale)
        test_backup = base_dir / "test_set" / "rgb_original"
        if not test_backup.exists():
            test_rgb.rename(test_backup)
            (base_dir / "test_set" / f"rgb_{scale}x").rename(test_rgb)
            print(f"  Test set rgb downscaled")

    print()
    print("=" * 60)
    print(f"Done! Update your S3PO-GS config with these values:")
    print(f"  width: {new_w}")
    print(f"  height: {new_h}")
    print(f"  fx: {new_fx}")
    print(f"  fy: {new_fy}")
    print(f"  cx: {new_cx}")
    print(f"  cy: {new_cy}")
    print("=" * 60)


if __name__ == "__main__":
    main()
