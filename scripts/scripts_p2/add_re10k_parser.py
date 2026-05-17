#!/usr/bin/env python3
"""
Patch S3PO-GS to support the Re10k dataset format.

Re10k uses the same cameras.json format as DL3DV (quaternion + translation).
This script adds a Re10kParser class and Re10kDataset to S3PO-GS's utils/dataset.py.

Usage:
    python add_re10k_parser.py --s3po_root /path/to/S3PO-GS

This script:
  1. Adds Re10kParser (identical logic to dl3dvParser) to utils/dataset.py
  2. Adds Re10kDataset class
  3. Registers 're10k' in the load_dataset() dispatcher
  4. Copies sparse config files into S3PO-GS/configs/mono/
"""

import argparse
import os
import shutil
from pathlib import Path

RE10K_PARSER_CODE = '''
class Re10kParser:
    """Parser for Re10k dataset (same cameras.json format as DL3DV)."""
    def __init__(self, input_folder, config):
        self.input_folder = input_folder
        self.begin = config["Dataset"]["begin"]
        self.end = config["Dataset"]["end"]

        self.color_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.n_img = len(self.color_paths)

        self.load_poses(os.path.join(self.input_folder, "cameras.json"))

    def load_poses(self, pose_file):
        self.poses = []
        self.frames = []

        with open(pose_file, "r") as f:
            all_poses = json.load(f)

        selected_poses = all_poses[self.begin:self.end]
        init_trans = np.array(selected_poses[0]["cam_trans"])

        for i, pose in enumerate(selected_poses):
            qx, qy, qz, qw = pose["cam_quat"]
            tx, ty, tz = pose["cam_trans"]

            rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = [tx, ty, tz] - init_trans

            inv_pose = np.linalg.inv(transform_matrix)
            self.poses.append(inv_pose)
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.color_paths[i],
                "mono_depth_path": self.color_paths[i],
                "transform_matrix": transform_matrix.tolist(),
            }
            self.frames.append(frame)
'''

RE10K_DATASET_CODE = '''
class Re10kDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]

        parser = Re10kParser(dataset_path, config)

        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.color_paths
        self.mono_depth_paths = parser.color_paths
        self.poses = parser.poses
'''

LOAD_DATASET_ADDITION = '''    elif config["Dataset"]["type"] == "re10k":
        return Re10kDataset(args, path, config)'''


def patch_dataset_py(s3po_root: Path):
    """Add Re10k support to utils/dataset.py."""
    dataset_py = s3po_root / "utils" / "dataset.py"

    if not dataset_py.exists():
        print(f"ERROR: {dataset_py} not found!")
        return False

    content = dataset_py.read_text(encoding="utf-8")

    if "Re10kParser" in content:
        print("  Re10kParser already exists in dataset.py, skipping.")
        return True

    # Insert Re10kParser after dl3dvParser class
    marker = "class KITTIParser:"
    if marker not in content:
        print(f"ERROR: Could not find '{marker}' in dataset.py")
        return False
    content = content.replace(marker, RE10K_PARSER_CODE + "\n" + marker)

    # Insert Re10kDataset after dl3dvDataset class
    marker2 = "class KITTIDataset(MonocularDataset):"
    if marker2 not in content:
        print(f"ERROR: Could not find '{marker2}' in dataset.py")
        return False
    content = content.replace(marker2, RE10K_DATASET_CODE + "\n" + marker2)

    # Add to load_dataset dispatcher
    dl3dv_dispatch = '    elif config["Dataset"]["type"] == "dl3dv":'
    if dl3dv_dispatch in content and 're10k' not in content.split("load_dataset")[1]:
        dl3dv_block_end = content.find("return dl3dvDataset(args, path, config)")
        if dl3dv_block_end != -1:
            insert_pos = content.find("\n", dl3dv_block_end) + 1
            content = content[:insert_pos] + LOAD_DATASET_ADDITION + "\n" + content[insert_pos:]

    dataset_py.write_text(content, encoding="utf-8")
    print("  Patched utils/dataset.py with Re10k support.")
    return True


def copy_configs(s3po_root: Path, scripts_dir: Path):
    """Copy sparse config files into S3PO-GS configs directory."""
    configs_src = scripts_dir / "configs"

    # Waymo sparse configs
    waymo_dst = s3po_root / "configs" / "mono" / "waymo_sparse"
    waymo_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(configs_src / "waymo" / "base_config.yaml", waymo_dst / "base_config.yaml")
    shutil.copy2(configs_src / "waymo" / "405841_sparse.yaml", waymo_dst / "405841_sparse.yaml")

    # DL3DV sparse configs
    dl3dv_dst = s3po_root / "configs" / "mono" / "dl3dv_sparse"
    dl3dv_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(configs_src / "dl3dv" / "base_config.yaml", dl3dv_dst / "base_config.yaml")
    shutil.copy2(configs_src / "dl3dv" / "2_sparse.yaml", dl3dv_dst / "2_sparse.yaml")

    # Re10k sparse configs
    re10k_dst = s3po_root / "configs" / "mono" / "re10k_sparse"
    re10k_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(configs_src / "re10k" / "base_config.yaml", re10k_dst / "base_config.yaml")
    shutil.copy2(configs_src / "re10k" / "1_sparse.yaml", re10k_dst / "1_sparse.yaml")

    print("  Copied sparse configs to S3PO-GS/configs/mono/")


def main():
    parser = argparse.ArgumentParser(description="Patch S3PO-GS with Re10k support and sparse configs")
    parser.add_argument("--s3po_root", type=str, required=True,
                        help="Path to S3PO-GS repository root")
    args = parser.parse_args()

    s3po_root = Path(args.s3po_root).resolve()
    scripts_dir = Path(__file__).resolve().parent

    print(f"S3PO-GS root: {s3po_root}")
    print("=" * 60)

    print("[1/2] Patching utils/dataset.py ...")
    success = patch_dataset_py(s3po_root)
    if not success:
        print("FAILED. Please check the error above.")
        return

    print("[2/2] Copying sparse config files ...")
    copy_configs(s3po_root, scripts_dir)

    print("=" * 60)
    print("Done! S3PO-GS is now ready for sparse Re10k and sparse configs.")
    print("\nAlso ensure sparse dataset directories are placed at:")
    print(f"  {s3po_root}/datasets/waymo_sparse/405841/FRONT/")
    print(f"  {s3po_root}/datasets/dl3dv_sparse/2/")
    print(f"  {s3po_root}/datasets/re10k_sparse/1/")


if __name__ == "__main__":
    main()
