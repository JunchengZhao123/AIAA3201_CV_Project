#!/usr/bin/env python3
"""
Part 3: Pseudo-View Generation via RIFE + 3DGS + Difix3D.

This script generates pseudo-views between sparse keyframes by:
  1. Loading the trained S3PO-GS Gaussian model (from Part 2)
  2. Interpolating camera poses between adjacent sparse keyframes
  3. Using RIFE video frame interpolation on adjacent real sparse images
     to produce temporally coherent intermediate frames
  4. Rendering 3DGS views at interpolated poses for depth/geometry cues
  5. Blending RIFE interpolations with 3DGS renders using depth confidence
  6. Optionally applying Difix3D diffusion cleanup on the blended result
  7. Outputs pseudo-views + their estimated poses for hybrid training

Usage:
    python generate_pseudo_views.py \
        --model_path  S3PO-GS/results/<dataset>/<timestamp>/point_cloud.ply \
        --sparse_dir  dataset_sparse_p2/waymo/405841/FRONT \
        --dataset_type waymo \
        --output_dir  running_result/5_3/pseudo_views/waymo_405841 \
        --n_interp 2

    python generate_pseudo_views.py --config scripts/scripts_p3/configs/waymo_p3.yaml
"""

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation as Rot, Slerp
from torch.nn import functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =====================================================================
#  RIFE Video Frame Interpolation
# =====================================================================

class RIFEInterpolator:
    """
    Video frame interpolation using Practical-RIFE.

    Given two adjacent real images, produces intermediate frames at arbitrary
    timesteps.  At t close to 0 the result resembles img0; at t close to 1
    it resembles img1 — exactly the behaviour needed for generating pseudo-
    views that "lean toward" their closer sparse keyframe.

    Repository: https://github.com/hzwer/Practical-RIFE
    """

    PAD_DIV = 64  # RIFE requires H, W divisible by 64

    def __init__(self, rife_root, device="cuda"):
        self.device = device
        rife_root = str(rife_root)

        if rife_root not in sys.path:
            sys.path.insert(0, rife_root)

        model_dir = os.path.join(rife_root, "train_log")
        if not os.path.isdir(model_dir):
            raise RuntimeError(
                f"RIFE model weights not found at {model_dir}\n"
                "  Run: cd Practical-RIFE && "
                "wget -O train_log.zip https://drive.google.com/... && unzip ..."
            )

        try:
            from model.RIFE_HDv3 import Model
        except ImportError:
            try:
                from train_log.RIFE_HDv3 import Model
            except ImportError:
                from model.RIFE_HDv2 import Model

        model = Model()
        model.load_model(model_dir, -1)
        model.eval()
        model.device()
        self._model = model
        self._version = getattr(model, "version", 0)
        print(f"  RIFE loaded (version {self._version}) from {rife_root}")

    def _to_tensor(self, img_np):
        """Convert HWC uint8 numpy array to NCHW float tensor on device."""
        t = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
        return t.unsqueeze(0).to(self.device, non_blocking=True)

    @torch.no_grad()
    def interpolate(self, img0_np, img1_np, timestep=0.5):
        """
        Interpolate between two images at the given timestep.

        Args:
            img0_np: np.ndarray (H, W, 3) uint8 — first reference frame
            img1_np: np.ndarray (H, W, 3) uint8 — second reference frame
            timestep: float in (0, 1).  Values < 0.5 produce results closer
                      to img0; values > 0.5 closer to img1.
        Returns:
            result: np.ndarray (H, W, 3) uint8
        """
        h, w = img0_np.shape[:2]
        if img1_np.shape[:2] != (h, w):
            img1_np = cv2.resize(img1_np, (w, h))

        t0 = self._to_tensor(img0_np)
        t1 = self._to_tensor(img1_np)

        ph = ((h - 1) // self.PAD_DIV + 1) * self.PAD_DIV
        pw = ((w - 1) // self.PAD_DIV + 1) * self.PAD_DIV
        pad = (0, pw - w, 0, ph - h)
        t0 = F.pad(t0, pad)
        t1 = F.pad(t1, pad)

        if self._version >= 3.9:
            mid = self._model.inference(t0, t1, timestep)
        else:
            mid = self._model.inference(t0, t1)

        result = (mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        return result

    @torch.no_grad()
    def interpolate_n(self, img0_np, img1_np, n=2):
        """
        Generate *n* evenly-spaced intermediate frames between img0 and img1
        (excluding the endpoints themselves).

        Returns a list of n np.ndarrays (H, W, 3) uint8.
        """
        results = []
        for i in range(1, n + 1):
            t = i / (n + 1)
            results.append(self.interpolate(img0_np, img1_np, timestep=t))
        return results


# =====================================================================
#  Reference-guided blending of RIFE + 3DGS renders
# =====================================================================

def blend_rife_with_gs(rife_frame, gs_render, gs_depth, alpha_t,
                       depth_blend_weight=0.3):
    """
    Blend a RIFE-interpolated frame with a 3DGS render using depth confidence.

    Regions where 3DGS has valid depth (well-covered by Gaussians) receive a
    small contribution from the 3DGS render to preserve geometric accuracy.
    Regions with missing/unreliable depth rely entirely on the RIFE frame.

    Args:
        rife_frame:  np.ndarray (H, W, 3) uint8 — RIFE interpolation result
        gs_render:   np.ndarray (H, W, 3) uint8 — 3DGS rendered image
        gs_depth:    np.ndarray (H, W) float    — 3DGS depth map
        alpha_t:     float — interpolation position (0→img0, 1→img1)
        depth_blend_weight: base weight given to 3DGS in high-confidence regions
    Returns:
        blended: np.ndarray (H, W, 3) uint8
    """
    if rife_frame.shape[:2] != gs_render.shape[:2]:
        gs_render = cv2.resize(gs_render, (rife_frame.shape[1], rife_frame.shape[0]))
        gs_depth = cv2.resize(gs_depth, (rife_frame.shape[1], rife_frame.shape[0]))

    valid_depth = gs_depth > 0.01
    depth_med = np.median(gs_depth[valid_depth]) if valid_depth.any() else 1.0
    confidence = np.clip(gs_depth / (depth_med * 2.0 + 1e-6), 0.0, 1.0)
    confidence[~valid_depth] = 0.0
    confidence = cv2.GaussianBlur(confidence, (15, 15), 0)

    gs_weight = confidence * depth_blend_weight
    gs_weight = gs_weight[:, :, np.newaxis]

    rife_f = rife_frame.astype(np.float32)
    gs_f = gs_render.astype(np.float32)
    blended = (1.0 - gs_weight) * rife_f + gs_weight * gs_f
    return np.clip(blended, 0, 255).astype(np.uint8)


def find_s3po_model(results_base, dataset_key):
    """Locate the latest S3PO-GS point cloud for a dataset.

    S3PO-GS saves models under:
      results/<dataset_key>/<timestamp>/point_cloud/final/point_cloud.ply
    """
    candidates = [
        "point_cloud/final/point_cloud.ply",
        "point_cloud/final_after_opt/point_cloud.ply",
        "point_cloud/iteration_*/point_cloud.ply",
    ]
    results_base = Path(results_base)

    for ds_dir in sorted(results_base.glob(f"*{dataset_key}*"), reverse=True):
        ts_dirs = sorted([d for d in ds_dir.iterdir() if d.is_dir()], reverse=True)
        for ts_dir in ts_dirs:
            for cand in candidates:
                matches = sorted(ts_dir.glob(cand), reverse=True)
                if matches:
                    return str(matches[0]), str(ts_dir)
    return None, None


def interpolate_poses(pose_a, pose_b, n_interp):
    """
    Interpolate n_interp poses between pose_a and pose_b (both 4x4 c2w).
    Uses SLERP for rotation, linear interpolation for translation.
    Returns list of n_interp 4x4 matrices (excluding endpoints).
    """
    R_a = pose_a[:3, :3]
    R_b = pose_b[:3, :3]
    t_a = pose_a[:3, 3]
    t_b = pose_b[:3, 3]

    rot_a = Rot.from_matrix(R_a)
    rot_b = Rot.from_matrix(R_b)
    key_rots = Rot.concatenate([rot_a, rot_b])
    slerp = Slerp([0.0, 1.0], key_rots)

    poses = []
    for i in range(1, n_interp + 1):
        alpha = i / (n_interp + 1)
        R_interp = slerp(alpha).as_matrix()
        t_interp = (1 - alpha) * t_a + alpha * t_b
        T = np.eye(4)
        T[:3, :3] = R_interp
        T[:3, 3] = t_interp
        poses.append(T)
    return poses


def load_sparse_poses_waymo(sparse_dir):
    """Load camera poses from Waymo sparse directory (gt/*.txt = 4x4 matrices)."""
    sparse_dir = Path(sparse_dir)
    gt_files = sorted((sparse_dir / "gt").glob("*.txt"))
    poses = []
    for f in gt_files:
        pose = np.loadtxt(f, delimiter=" ").reshape(4, 4)
        poses.append(pose)

    calib_path = sparse_dir / "calib.txt"
    fx, fy, cx, cy = None, None, None, None
    width, height = 1920, 1280
    if calib_path.exists():
        for line in open(calib_path).readlines():
            if line.startswith("fx:"):
                parts = line.split()
                fx = float(parts[1])
                fy = float(parts[3])
                cx = float(parts[5])
                cy = float(parts[7])
    intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height}
    return poses, intrinsics


def load_sparse_poses_json(sparse_dir, width=256, height=256):
    """Load camera poses from DL3DV/Re10k cameras.json."""
    sparse_dir = Path(sparse_dir)
    with open(sparse_dir / "cameras.json", "r") as f:
        all_cameras = json.load(f)

    poses = []
    init_trans = np.array(all_cameras[0]["cam_trans"])
    for cam_data in all_cameras:
        qx, qy, qz, qw = cam_data["cam_quat"]
        tx, ty, tz = cam_data["cam_trans"]
        rot = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
        T_w2c = np.eye(4)
        T_w2c[:3, :3] = rot
        T_w2c[:3, 3] = np.array([tx, ty, tz]) - init_trans
        T_c2w = np.linalg.inv(T_w2c)
        poses.append(T_c2w)

    intrinsics_data = all_cameras[0]
    fx = intrinsics_data.get("fx", 0.5) * width
    fy = intrinsics_data.get("fy", 0.5) * height
    cx = intrinsics_data.get("cx", 0.5) * width
    cy = intrinsics_data.get("cy", 0.5) * height
    intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height}
    return poses, intrinsics


def load_sparse_images(sparse_dir, dataset_type):
    """Load sparse frame images."""
    sparse_dir = Path(sparse_dir)
    if dataset_type == "waymo":
        img_dir = sparse_dir / "rgb"
    else:
        img_dir = sparse_dir / "rgb"
    imgs = sorted(img_dir.glob("*.png"))
    if not imgs:
        imgs = sorted(img_dir.glob("*.jpg"))
    return imgs


# ---------- Gaussian rendering without S3PO-GS import ----------

def detect_sh_degree_from_ply(ply_path):
    """Auto-detect SH degree from a Gaussian Splatting PLY file."""
    from plyfile import PlyData
    plydata = PlyData.read(ply_path)
    extra_f_names = [p.name for p in plydata.elements[0].properties
                     if p.name.startswith("f_rest_")]
    n_extra = len(extra_f_names)
    # n_extra = 3 * (sh_degree + 1)^2 - 3
    n_coeffs = (n_extra + 3) // 3
    sh_degree = int(round(math.sqrt(n_coeffs))) - 1
    return max(sh_degree, 0)


def render_from_ply(ply_path, pose_c2w, intrinsics, device="cuda"):
    """
    Render a view from a Gaussian Splatting PLY model.
    This imports from S3PO-GS's gaussian_splatting module.
    Must be run inside the S3PO-GS conda environment.
    """
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.gaussian_renderer import render
    from gaussian_splatting.utils.graphics_utils import (
        focal2fov, getWorld2View2, getProjectionMatrix2,
    )
    from munch import munchify

    if not hasattr(render_from_ply, "_gaussians"):
        sh_deg = detect_sh_degree_from_ply(ply_path)
        print(f"  Auto-detected SH degree from PLY: {sh_deg}")
        gaussians = GaussianModel(sh_degree=sh_deg)
        gaussians.load_ply(ply_path)
        render_from_ply._gaussians = gaussians

    gaussians = render_from_ply._gaussians

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

    pipeline = munchify({"convert_SHs_python": False, "compute_cov3D_python": False})
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    with torch.no_grad():
        render_pkg = render(cam, gaussians, pipeline, background)

    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
    depth = render_pkg["depth"]
    return image, depth


def calibrate_exposure(ply_path, poses, sparse_images, intrinsics, device="cuda"):
    """
    Calibrate per-keyframe exposure by rendering at known poses and fitting
    exposure_a, exposure_b via least squares against GT images.

    S3PO-GS color model: output = exp(a) * render + b
    Returns list of (exposure_a, exposure_b) tuples, one per keyframe.
    """
    exposures = []
    print("  Calibrating per-keyframe exposure compensation...")
    for i, (pose, img_path) in enumerate(zip(poses, sparse_images)):
        gt = np.array(Image.open(img_path)).astype(np.float32) / 255.0
        rendered, _ = render_from_ply(ply_path, pose, intrinsics, device)
        rendered_np = rendered.cpu().numpy().transpose(1, 2, 0)

        if gt.shape[:2] != rendered_np.shape[:2]:
            gt = np.array(Image.fromarray(
                (gt * 255).astype(np.uint8)
            ).resize(
                (rendered_np.shape[1], rendered_np.shape[0]), Image.LANCZOS
            )).astype(np.float32) / 255.0

        r_flat = rendered_np.reshape(-1).astype(np.float64)
        g_flat = gt.reshape(-1).astype(np.float64)

        X = np.stack([r_flat, np.ones_like(r_flat)], axis=1)
        result = np.linalg.lstsq(X, g_flat, rcond=None)
        A, b = result[0]
        a = float(np.log(max(A, 1e-6)))

        exposures.append((a, float(b)))
        print(f"    Keyframe {i}: exposure_a={a:.4f}, exposure_b={b:.4f} (scale={A:.4f})")

    return exposures


# ---------- Difix3D pseudo-view completion ----------

def _patch_torch_xpu():
    """Shim torch.xpu for PyTorch < 2.4 so diffusers import doesn't crash."""
    if not hasattr(torch, "xpu"):
        import types
        xpu = types.ModuleType("torch.xpu")
        xpu.is_available = lambda: False
        torch.xpu = xpu

_patch_torch_xpu()


DIFIX_DEFAULT_LOCAL_PATH = (
    "/home/user/.cache/huggingface/hub/"
    "models--nvidia--difix/snapshots/"
    "adcf25f6306f11b2bde4c37ca00f9b10244bb036"
)


class Difix3DEnhancer:
    """
    Wraps the nvidia/difix DifixPipeline for single-step 3DGS artifact removal.

    The model uses a custom VAE with encoder-to-decoder skip connections that
    preserve spatial structure while the UNet removes rendering artifacts in a
    single denoising step (timestep 199).

    Official model: https://huggingface.co/nvidia/difix
    """

    DIFIX_WIDTH = 1024
    DIFIX_HEIGHT = 576

    def __init__(self, difix_model_path=None, device="cuda"):
        self.device = device

        model_dir = difix_model_path or DIFIX_DEFAULT_LOCAL_PATH
        if not os.path.isdir(model_dir):
            raise RuntimeError(
                f"DiFix model directory not found: {model_dir}\n"
                "  Provide --difix_model_path or ensure the default path exists."
            )

        print(f"  Loading DiFix from {model_dir} ...")

        # The DiFix repo ships custom pipeline_difix.py and vae/autoencoder_kl.py
        # that diffusers cannot auto-resolve.  Load components manually.
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        vae_dir = os.path.join(model_dir, "vae")
        if vae_dir not in sys.path:
            sys.path.insert(0, vae_dir)

        # --- Shims for older diffusers versions ---
        import diffusers.loaders as _dl
        if not hasattr(_dl, "FromOriginalVAEMixin"):
            _dl.FromOriginalVAEMixin = type("FromOriginalVAEMixin", (), {})

        from diffusers.models.modeling_utils import ModelMixin as _MM
        if not hasattr(_MM, "add_adapter"):
            def _add_adapter(self, adapter_config, adapter_name="default"):
                from peft import inject_adapter_in_model
                inject_adapter_in_model(adapter_config, self, adapter_name)
            _MM.add_adapter = _add_adapter

        from pipeline_difix import DifixPipeline
        from autoencoder_kl import AutoencoderKL as DifixVAE
        from transformers import CLIPTokenizer, CLIPTextModel
        from diffusers import DDPMScheduler, UNet2DConditionModel

        dtype = torch.bfloat16
        tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            model_dir, subfolder="text_encoder", torch_dtype=dtype,
        )
        scheduler = DDPMScheduler.from_pretrained(model_dir, subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(
            model_dir, subfolder="unet", torch_dtype=dtype,
        )
        vae = DifixVAE.from_pretrained(
            model_dir, subfolder="vae", torch_dtype=dtype,
        )

        pipe = DifixPipeline(
            vae=vae,
            unet=unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
        ).to(device)

        print("  DiFix loaded successfully")
        self.pipe = pipe

    @torch.no_grad()
    def enhance(self, rendered_image):
        """
        Remove 3DGS rendering artifacts from a rendered image.

        Args:
            rendered_image: np.ndarray (H, W, 3) uint8 or PIL.Image
        Returns:
            enhanced: PIL.Image
        """
        if isinstance(rendered_image, np.ndarray):
            rendered_image = Image.fromarray(rendered_image.astype(np.uint8))

        orig_size = rendered_image.size
        target_size = (self.DIFIX_WIDTH, self.DIFIX_HEIGHT)
        input_pil = rendered_image.resize(target_size, Image.LANCZOS)

        result = self.pipe(
            "remove degradation",
            image=input_pil,
            num_inference_steps=1,
            timesteps=[199],
            guidance_scale=0.0,
        ).images[0]

        result = result.resize(orig_size, Image.LANCZOS)
        return result


def generate_pseudo_views_for_scene(
    model_path,
    sparse_dir,
    dataset_type,
    output_dir,
    n_interp=2,
    difix_model_path=None,
    rife_root=None,
    device="cuda",
    skip_difix=False,
    blend_mode="rife_gs",
):
    """
    Main pseudo-view generation pipeline for one scene.

    blend_mode controls how pseudo-views are produced:
      "rife_only"  — pure RIFE interpolation between adjacent sparse images
      "gs_only"    — pure 3DGS rendering at interpolated poses (legacy)
      "rife_gs"    — RIFE interpolation blended with 3DGS render (recommended)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rendered_raw").mkdir(exist_ok=True)
    (output_dir / "rife_interp").mkdir(exist_ok=True)
    (output_dir / "pseudo_views").mkdir(exist_ok=True)
    (output_dir / "poses").mkdir(exist_ok=True)

    # Load sparse data
    print(f"\n  Loading sparse data from: {sparse_dir}")
    if dataset_type == "waymo":
        poses, intrinsics = load_sparse_poses_waymo(sparse_dir)
    else:
        w = 256 if dataset_type == "re10k" else 256
        h = 256 if dataset_type == "re10k" else 256
        poses, intrinsics = load_sparse_poses_json(sparse_dir, width=w, height=h)

    sparse_images = load_sparse_images(sparse_dir, dataset_type)
    n_sparse = len(poses)
    print(f"  Sparse frames: {n_sparse}, interpolating {n_interp} between each pair")
    print(f"  Blend mode: {blend_mode}")

    # --- Load adjacent sparse images into memory ---
    sparse_imgs_np = []
    for img_path in sparse_images:
        img = np.array(Image.open(img_path).convert("RGB"))
        sparse_imgs_np.append(img)
    print(f"  Loaded {len(sparse_imgs_np)} sparse reference images")

    # --- Initialize RIFE interpolator ---
    rife = None
    use_rife = blend_mode in ("rife_only", "rife_gs")
    if use_rife:
        if rife_root is None:
            proj = Path(__file__).resolve().parents[2]
            rife_root = str(proj / "Practical-RIFE")
        print(f"  Initializing RIFE from: {rife_root}")
        try:
            rife = RIFEInterpolator(rife_root, device=device)
        except Exception as e:
            print(f"  WARNING: Could not load RIFE: {e}")
            print("  Falling back to gs_only blend mode.")
            blend_mode = "gs_only"

    # --- Initialize Difix3D enhancer ---
    enhancer = None
    if not skip_difix:
        print("  Initializing Difix3D enhancer...")
        try:
            enhancer = Difix3DEnhancer(difix_model_path=difix_model_path, device=device)
        except Exception as e:
            print(f"  WARNING: Could not load Difix3D: {e}")
            print("  Will skip diffusion cleanup.")

    # Calibrate per-keyframe exposure compensation (for 3DGS renders)
    use_gs = blend_mode in ("gs_only", "rife_gs")
    exposures = None
    if use_gs:
        exposures = calibrate_exposure(
            model_path, poses, sparse_images, intrinsics, device
        )

    all_pseudo_meta = []
    pseudo_idx = 0

    for i in tqdm(range(n_sparse - 1), desc="Generating pseudo-views"):
        pose_a = poses[i]
        pose_b = poses[i + 1]
        img_a = sparse_imgs_np[i]
        img_b = sparse_imgs_np[i + 1]

        interp_poses = interpolate_poses(pose_a, pose_b, n_interp)

        # --- RIFE: interpolate reference images ---
        rife_frames = []
        if rife is not None:
            try:
                rife_frames = rife.interpolate_n(img_a, img_b, n=n_interp)
            except Exception as e:
                print(f"  WARNING: RIFE failed for pair ({i},{i+1}): {e}")
                rife_frames = [None] * n_interp
        else:
            rife_frames = [None] * n_interp

        for j, pose_interp in enumerate(interp_poses):
            alpha = (j + 1) / (n_interp + 1)

            # --- 3DGS render at interpolated pose ---
            rendered_np = None
            depth_np = None
            if use_gs:
                try:
                    rendered, depth = render_from_ply(
                        model_path, pose_interp, intrinsics, device
                    )
                    exp_a_val = (1 - alpha) * exposures[i][0] + alpha * exposures[i + 1][0]
                    exp_b_val = (1 - alpha) * exposures[i][1] + alpha * exposures[i + 1][1]
                    rendered = torch.exp(torch.tensor(exp_a_val, device=device)) * rendered + exp_b_val
                    rendered = torch.clamp(rendered, 0.0, 1.0)

                    rendered_np = (rendered.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    depth_np = depth.cpu().numpy().squeeze()
                except Exception as e:
                    print(f"  WARNING: 3DGS render failed for pair ({i},{i+1}) interp {j}: {e}")

            # Save raw 3DGS render
            if rendered_np is not None:
                raw_path = output_dir / "rendered_raw" / f"{pseudo_idx:06d}.png"
                Image.fromarray(rendered_np).save(raw_path)
            else:
                raw_path = None

            # Save RIFE interpolation
            rife_path = None
            if rife_frames[j] is not None:
                rife_path = output_dir / "rife_interp" / f"{pseudo_idx:06d}.png"
                Image.fromarray(rife_frames[j]).save(rife_path)

            # --- Compose the final pseudo-view ---
            if blend_mode == "rife_gs" and rife_frames[j] is not None and rendered_np is not None:
                pseudo_np = blend_rife_with_gs(
                    rife_frames[j], rendered_np, depth_np, alpha,
                    depth_blend_weight=0.25,
                )
            elif blend_mode == "rife_only" and rife_frames[j] is not None:
                pseudo_np = rife_frames[j]
            elif rendered_np is not None:
                pseudo_np = rendered_np
            elif rife_frames[j] is not None:
                pseudo_np = rife_frames[j]
            else:
                print(f"  WARNING: No output for pair ({i},{i+1}) interp {j}, skipping")
                continue

            # --- Difix3D cleanup ---
            if enhancer is not None:
                try:
                    enhanced = enhancer.enhance(pseudo_np)
                    pseudo_np = np.array(enhanced)
                except Exception as e:
                    print(f"  WARNING: Difix3D enhancement failed: {e}")

            pseudo_path = output_dir / "pseudo_views" / f"{pseudo_idx:06d}.png"
            Image.fromarray(pseudo_np).save(pseudo_path)

            # Save pose
            pose_path = output_dir / "poses" / f"{pseudo_idx:06d}.txt"
            np.savetxt(pose_path, pose_interp.reshape(4, 4))

            # Save depth for confidence computation
            depth_path = None
            if depth_np is not None:
                depth_path = output_dir / "rendered_raw" / f"{pseudo_idx:06d}_depth.npy"
                np.save(depth_path, depth_np)

            meta = {
                "pseudo_idx": pseudo_idx,
                "pair": [i, i + 1],
                "alpha": float(alpha),
                "blend_mode": blend_mode,
                "pose": pose_interp.tolist(),
                "pseudo_path": str(pseudo_path),
                "raw_path": str(raw_path) if raw_path else None,
                "rife_path": str(rife_path) if rife_path else None,
                "depth_path": str(depth_path) if depth_path else None,
            }
            all_pseudo_meta.append(meta)
            pseudo_idx += 1

    # Also save references to original sparse frames
    sparse_meta = []
    for i in range(n_sparse):
        sparse_meta.append({
            "sparse_idx": i,
            "image_path": str(sparse_images[i]),
            "pose": poses[i].tolist(),
        })

    output_meta = {
        "dataset_type": dataset_type,
        "n_sparse": n_sparse,
        "n_pseudo": pseudo_idx,
        "n_interp_per_pair": n_interp,
        "blend_mode": blend_mode,
        "intrinsics": intrinsics,
        "sparse_frames": sparse_meta,
        "pseudo_frames": all_pseudo_meta,
    }
    meta_path = output_dir / "pseudo_meta.json"
    with open(meta_path, "w") as f:
        json.dump(output_meta, f, indent=2)

    print(f"\n  Generated {pseudo_idx} pseudo-views (blend_mode={blend_mode})")
    print(f"  Metadata: {meta_path}")
    return output_meta


def main():
    parser = argparse.ArgumentParser(
        description="Part 3: Generate pseudo-views via Difix3D"
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to point_cloud.ply")
    parser.add_argument("--sparse_dir", type=str, default=None, help="Sparse frames directory")
    parser.add_argument("--dataset_type", type=str, default="waymo",
                        choices=["waymo", "dl3dv", "re10k"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_interp", type=int, default=2,
                        help="Number of interpolated frames per pair (default 2)")
    parser.add_argument("--difix_model_path", type=str, default=None,
                        help="Local path to nvidia/difix model directory")
    parser.add_argument("--rife_root", type=str, default=None,
                        help="Path to Practical-RIFE clone (default: $PROJ/Practical-RIFE)")
    parser.add_argument("--blend_mode", type=str, default="rife_gs",
                        choices=["rife_only", "gs_only", "rife_gs"],
                        help="Pseudo-view generation strategy (default: rife_gs)")
    parser.add_argument("--proj_root", type=str, default=None)
    parser.add_argument("--skip_difix", action="store_true",
                        help="Skip Difix3D diffusion cleanup")
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

    # Auto-detect model path if not provided
    if args.model_path is None:
        # S3PO-GS saves results to its own results/ dir (relative to S3PO-GS root)
        # and also may be copied to running_result/5_2/results/
        search_dirs = [
            proj / "running_result" / "5_2" / "results",
            proj / "S3PO-GS" / "results",
        ]
        ds_key = {
            "waymo": "waymo_sparse_405841",
            "dl3dv": "dl3dv_sparse_2",
            "re10k": "re10k_sparse_1",
        }[args.dataset_type]
        for search_dir in search_dirs:
            if search_dir.exists():
                args.model_path, _ = find_s3po_model(search_dir, ds_key)
                if args.model_path:
                    break
        if args.model_path is None:
            print(f"ERROR: Could not find S3PO-GS model for {args.dataset_type}")
            print(f"  Searched in:")
            for d in search_dirs:
                print(f"    {d}")
            print(f"\n  Expected path pattern: <results_dir>/{ds_key}/<timestamp>/point_cloud/final/point_cloud.ply")
            print(f"  Provide explicitly: --model_path /path/to/point_cloud.ply")
            sys.exit(1)
        print(f"  Auto-detected model: {args.model_path}")

    if args.sparse_dir is None:
        ds_map = {
            "waymo": "waymo/405841/FRONT",
            "dl3dv": "dl3dv/2",
            "re10k": "re10k/1",
        }
        args.sparse_dir = str(proj / "dataset_sparse_p2" / ds_map[args.dataset_type])

    if args.output_dir is None:
        ds_names = {"waymo": "waymo_405841", "dl3dv": "dl3dv_2", "re10k": "re10k_1"}
        args.output_dir = str(
            proj / "running_result" / "5_3" / "pseudo_views" / ds_names[args.dataset_type]
        )

    # Add S3PO-GS to path for rendering
    s3po_root = str(proj / "S3PO-GS")
    if s3po_root not in sys.path:
        sys.path.insert(0, s3po_root)

    generate_pseudo_views_for_scene(
        model_path=args.model_path,
        sparse_dir=args.sparse_dir,
        dataset_type=args.dataset_type,
        output_dir=args.output_dir,
        n_interp=args.n_interp,
        difix_model_path=args.difix_model_path,
        rife_root=args.rife_root,
        device=args.device,
        skip_difix=args.skip_difix,
        blend_mode=args.blend_mode,
    )


if __name__ == "__main__":
    main()
