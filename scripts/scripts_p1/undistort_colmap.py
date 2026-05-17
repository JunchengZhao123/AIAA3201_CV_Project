"""
Undistort COLMAP SIMPLE_RADIAL cameras to PINHOLE (no COLMAP binary needed).

Reads cameras.bin + images.bin, undistorts all images with OpenCV, writes:
  - undistorted images to <scene>/images/
  - new cameras.bin with PINHOLE model to <scene>/sparse/0/cameras.bin
  - copies images.bin and points3D.bin as-is

Usage:
    python scripts/undistort_colmap.py --scene_dir ~/datasets_planA/405841
    python scripts/undistort_colmap.py --scene_dir ~/datasets_planA/DL3DV-2
    python scripts/undistort_colmap.py --scene_dir ~/datasets_planA/Re10k-1
"""

import os, sys, struct, argparse, shutil
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# COLMAP camera model IDs
CAMERA_MODEL_IDS = {
    0: ("SIMPLE_PINHOLE", 3),   # f, cx, cy
    1: ("PINHOLE", 4),          # fx, fy, cx, cy
    2: ("SIMPLE_RADIAL", 4),    # f, cx, cy, k1
    3: ("RADIAL", 5),           # f, cx, cy, k1, k2
    4: ("OPENCV", 8),           # fx, fy, cx, cy, k1, k2, p1, p2
}


def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            num_params = CAMERA_MODEL_IDS[model_id][1]
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[cam_id] = {
                "model_id": model_id,
                "model_name": CAMERA_MODEL_IDS[model_id][0],
                "width": width,
                "height": height,
                "params": list(params),
            }
    return cameras


def write_cameras_bin(cameras, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam_id in sorted(cameras.keys()):
            cam = cameras[cam_id]
            f.write(struct.pack("<i", cam_id))
            f.write(struct.pack("<i", cam["model_id"]))
            f.write(struct.pack("<Q", cam["width"]))
            f.write(struct.pack("<Q", cam["height"]))
            for p in cam["params"]:
                f.write(struct.pack("<d", p))


def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("<i", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            # Skip 2D points (x, y, point3D_id) per point
            f.read(num_points2D * 24)
            images[img_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }
    return images


def get_undistort_params(cam):
    """Build OpenCV camera matrix and distortion coeffs from COLMAP camera."""
    model = cam["model_name"]
    params = cam["params"]
    w, h = cam["width"], cam["height"]

    if model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, 0, 0, 0], dtype=np.float64)
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, 0, 0], dtype=np.float64)
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, p1, p2], dtype=np.float64)
    elif model in ("PINHOLE", "SIMPLE_PINHOLE"):
        return None, None, None  # already undistorted
    else:
        raise ValueError(f"Unsupported camera model: {model}")

    # Compute optimal new camera matrix (alpha=0 crops all black borders)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    return K, dist, new_K


def main():
    parser = argparse.ArgumentParser(description="Undistort COLMAP scene to PINHOLE")
    parser.add_argument("--scene_dir", required=True, help="Scene directory with images/ and sparse/0/")
    parser.add_argument("--images_subdir", default="images_distorted",
                        help="Where to move original distorted images (default: images_distorted)")
    args = parser.parse_args()

    scene = Path(args.scene_dir)
    sparse_dir = scene / "sparse" / "0"
    images_dir = scene / "images"

    cameras_bin = sparse_dir / "cameras.bin"
    images_bin = sparse_dir / "images.bin"
    points3d_bin = sparse_dir / "points3D.bin"

    if not cameras_bin.exists():
        print(f"ERROR: {cameras_bin} not found"); sys.exit(1)
    if not images_dir.exists():
        print(f"ERROR: {images_dir} not found"); sys.exit(1)

    cameras = read_cameras_bin(str(cameras_bin))
    colmap_images = read_images_bin(str(images_bin))

    # Check if any camera needs undistortion
    needs_undistort = any(c["model_name"] not in ("PINHOLE", "SIMPLE_PINHOLE")
                         for c in cameras.values())
    if not needs_undistort:
        print("All cameras are already PINHOLE/SIMPLE_PINHOLE. Nothing to do.")
        return

    print(f"Scene: {scene}")
    print(f"Cameras: {len(cameras)} (model: {cameras[1]['model_name']})")
    print(f"Images: {len(colmap_images)}")

    # Move original images, create new output dir
    distorted_dir = scene / args.images_subdir
    if not distorted_dir.exists():
        # Resolve symlink if images/ is a symlink
        if images_dir.is_symlink():
            real_target = images_dir.resolve()
            images_dir.unlink()
            distorted_dir = real_target
            os.makedirs(images_dir, exist_ok=True)
        else:
            images_dir.rename(distorted_dir)
            os.makedirs(images_dir, exist_ok=True)
    else:
        # distorted dir already exists from previous run, just ensure output exists
        if images_dir.is_symlink():
            images_dir.unlink()
        os.makedirs(images_dir, exist_ok=True)

    print(f"Distorted images: {distorted_dir}")
    print(f"Undistorted output: {images_dir}")

    # Process each image
    new_cameras = {}
    for cam_id, cam in cameras.items():
        result = get_undistort_params(cam)
        if result[0] is None:
            new_cameras[cam_id] = cam
            continue
        K, dist, new_K = result
        fx_new = new_K[0, 0]
        fy_new = new_K[1, 1]
        cx_new = new_K[0, 2]
        cy_new = new_K[1, 2]
        new_cameras[cam_id] = {
            "model_id": 1,  # PINHOLE
            "model_name": "PINHOLE",
            "width": cam["width"],
            "height": cam["height"],
            "params": [fx_new, fy_new, cx_new, cy_new],
        }

    # Build lookup: camera_id -> (K, dist, new_K)
    undistort_map = {}
    for cam_id, cam in cameras.items():
        result = get_undistort_params(cam)
        if result[0] is not None:
            undistort_map[cam_id] = result

    processed = 0
    for img_id, img_info in tqdm(colmap_images.items(), desc="Undistorting"):
        name = img_info["name"]
        cam_id = img_info["camera_id"]

        src_path = distorted_dir / name
        dst_path = images_dir / name

        if dst_path.exists():
            processed += 1
            continue

        if not src_path.exists():
            print(f"  WARNING: {src_path} not found, skipping")
            continue

        img = cv2.imread(str(src_path))
        if img is None:
            print(f"  WARNING: cannot read {src_path}")
            continue

        if cam_id in undistort_map:
            K, dist, new_K = undistort_map[cam_id]
            img_undist = cv2.undistort(img, K, dist, None, new_K)
        else:
            img_undist = img

        os.makedirs(dst_path.parent, exist_ok=True)
        cv2.imwrite(str(dst_path), img_undist)
        processed += 1

    print(f"Undistorted {processed} images")

    # Write new cameras.bin (PINHOLE)
    backup = str(cameras_bin) + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(str(cameras_bin), backup)
    write_cameras_bin(new_cameras, str(cameras_bin))
    print(f"Wrote PINHOLE cameras.bin ({len(new_cameras)} cameras)")
    print("Done!")


if __name__ == "__main__":
    main()
