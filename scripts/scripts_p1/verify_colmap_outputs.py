"""
Verify COLMAP-format sparse outputs from Plan A (COLMAP), Plan B (VGGT), and Plan B (Pi3).
Reads cameras.bin, images.bin, points3D.bin and prints summary statistics.

Usage:
    python scripts/verify_colmap_outputs.py --sparse_dir path/to/sparse/0
    python scripts/verify_colmap_outputs.py --all   # verify all known outputs
"""

import argparse
import struct
import os
import sys
import collections


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
            np_ = num_params.get(model_id, 4)
            params = struct.unpack(f"<{np_}d", f.read(8 * np_))
            cameras[cam_id] = {"model_id": model_id, "width": width, "height": height, "params": params}
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            name = name.decode("utf-8")
            num_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts * 24)  # skip point2D data
            images[img_id] = {"qvec": qvec, "tvec": tvec, "cam_id": cam_id, "name": name, "num_pts2d": num_pts}
    return images


def read_points3D_binary(path):
    count = 0
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            f.read(8)  # point_id
            f.read(24)  # xyz
            f.read(3)   # rgb
            f.read(8)   # error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)
            count += 1
    return count


def analyze_sparse(sparse_dir, label=""):
    cameras_path = os.path.join(sparse_dir, "cameras.bin")
    images_path = os.path.join(sparse_dir, "images.bin")
    points_path = os.path.join(sparse_dir, "points3D.bin")

    missing = []
    for p, name in [(cameras_path, "cameras.bin"), (images_path, "images.bin"), (points_path, "points3D.bin")]:
        if not os.path.exists(p):
            missing.append(name)

    if missing:
        print(f"  [{label}] MISSING: {', '.join(missing)}")
        return

    cameras = read_cameras_binary(cameras_path)
    images = read_images_binary(images_path)
    num_points = read_points3D_binary(points_path)

    cam0 = list(cameras.values())[0]
    model_names = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL", 4: "OPENCV"}
    model_name = model_names.get(cam0["model_id"], f"model_{cam0['model_id']}")

    print(f"  [{label}]")
    print(f"    Cameras:  {len(cameras):>6}  ({model_name}, {cam0['width']}x{cam0['height']})")
    print(f"    Images:   {len(images):>6}")
    print(f"    Points3D: {num_points:>6}")
    sizes = {n: os.path.getsize(p) for p, n in
             [(cameras_path, "cameras"), (images_path, "images"), (points_path, "points3D")]}
    print(f"    File sizes: cameras={sizes['cameras']//1024}KB, images={sizes['images']//1024}KB, points3D={sizes['points3D']//1024}KB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse_dir", type=str, default=None, help="Path to a single sparse/0/ directory")
    parser.add_argument("--all", action="store_true", help="Verify all known outputs")
    args = parser.parse_args()

    if args.sparse_dir:
        analyze_sparse(args.sparse_dir, label=args.sparse_dir)
        return

    if not args.all:
        parser.print_help()
        return

    proj = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    known_outputs = [
        ("Plan A COLMAP - 405841",  os.path.join(proj, "colmap/dataset/output/output_405841/sparse/0")),
        ("Plan A COLMAP - DL3DV-2", os.path.join(proj, "colmap/dataset/output/output_DL3DV-2/sparse/0")),
        ("Plan A COLMAP - Re10k-1", os.path.join(proj, "colmap/dataset/output/output_Re10k-1/sparse/0")),
        ("Plan B VGGT  - 405841",   os.path.join(proj, "running_result/5_1/5_1_1_Pose & Point Cloud Initialization/PlanB_VGGT_output/dataset_VGGT/405841/FRONT/sparse/0")),
        ("Plan B VGGT  - DL3DV-2",  os.path.join(proj, "running_result/5_1/5_1_1_Pose & Point Cloud Initialization/PlanB_VGGT_output/dataset_VGGT/DL3DV-2/sparse/0")),
        ("Plan B VGGT  - Re10k-1",  os.path.join(proj, "running_result/5_1/5_1_1_Pose & Point Cloud Initialization/PlanB_VGGT_output/dataset_VGGT/Re10k-1/sparse/0")),
        ("Plan B Pi3   - 405841",   os.path.join(proj, "running_result/5_1/5_1_1_Pose & Point Cloud Initialization/PlanB_Pi3_output/dataset_Pi3/405841/FRONT/sparse/0")),
        ("Plan B Pi3   - DL3DV-2",  os.path.join(proj, "running_result/5_1/5_1_1_Pose & Point Cloud Initialization/PlanB_Pi3_output/dataset_Pi3/DL3DV-2/sparse/0")),
        ("Plan B Pi3   - Re10k-1",  os.path.join(proj, "running_result/5_1/5_1_1_Pose & Point Cloud Initialization/PlanB_Pi3_output/dataset_Pi3/Re10k-1/sparse/0")),
    ]

    # Add Mip-NeRF scenes (pre-built COLMAP + VGGT + Pi3)
    for scene in ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]:
        known_outputs.append((
            f"Mip-NeRF COLMAP - {scene}",
            os.path.join(proj, f"dataset/Mip-NeRF/{scene}/sparse/0")
        ))
        known_outputs.append((
            f"Mip-NeRF VGGT  - {scene}",
            os.path.join(proj, f"running_result/5_1/5_1_1_Pose & Point Cloud Initialization/PlanB_VGGT_output/dataset_VGGT/Mip-NeRF/{scene}/sparse/0")
        ))
        known_outputs.append((
            f"Mip-NeRF Pi3   - {scene}",
            os.path.join(proj, f"running_result/5_1/5_1_1_Pose & Point Cloud Initialization/PlanB_Pi3_output/dataset_Pi3/Mip-NeRF/{scene}/sparse/0")
        ))

    for label, path in known_outputs:
        if os.path.exists(path):
            analyze_sparse(path, label)
        else:
            print(f"  [{label}] NOT FOUND: {path}")
        print()


if __name__ == "__main__":
    main()
