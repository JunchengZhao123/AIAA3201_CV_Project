"""
VGGT -> COLMAP Export  (Chunked, OOM-safe for A6000 48GB)
==========================================================
Processes ALL frames by splitting into overlapping chunks, running VGGT on
each chunk, aligning coordinate systems via Procrustes on the overlapping
camera centers, and merging into a single COLMAP sparse model.

Memory model (VGGT aggregator, from official benchmark on H100):
    50 fr -> 11.4 GB,  100 fr -> 21.2 GB,  200 fr -> 40.6 GB
    ~0.195 GB/frame + 1.5 GB base.
    Safe per-chunk limit for A6000 48 GB: 100 frames.

Usage:
    python scripts/run_vggt_colmap.py --scene_dir dataset/DL3DV-2
    python scripts/run_vggt_colmap.py --scene_dir dataset/Mip-NeRF/bicycle --output_dir out/bicycle
"""

import os, sys, glob, copy, argparse, random, math, struct, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from PIL import ImageFile

# Mip-NeRF (and other JPEG sets) may contain partially-written files; PIL otherwise
# raises OSError("image file is truncated"). Prefer re-downloading bad files; this
# allows the pipeline to continue with best-effort decoding.
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "vggt"))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import depth_to_cam_coords_points, closed_form_inverse_se3
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues

try:
    from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False

VGGT_RES = 518
CHUNK_SIZE = 100    # frames per forward pass (safe for 48 GB)
OVERLAP = 20        # shared frames between consecutive chunks


# ──────────────────────── Alignment utilities ────────────────────────

def camera_centers_from_extrinsic(extrinsics):
    """(N,4,4) w2c extrinsics -> (N,3) camera centers in world."""
    R = extrinsics[:, :3, :3]
    t = extrinsics[:, :3, 3]
    return -np.einsum("nij,nj->ni", R.transpose(0, 2, 1), t)


def umeyama_alignment(src, dst):
    """Umeyama similarity alignment: find s,R,t so that  s*R@src + t ≈ dst.
    src, dst: (K, 3).  Returns s (scalar), R (3,3), t (3,)."""
    assert src.shape == dst.shape and src.shape[1] == 3
    K = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    var_src = (src_c ** 2).sum() / K
    s = (S * np.array([1, 1, d])).sum() / var_src
    t = mu_d - s * R @ mu_s
    return s, R, t


def transform_extrinsic(extrinsic, s, R_align, t_align):
    """Apply similarity transform to w2c extrinsic (3x4 or 4x4).
    World transform: x_new = s * R_align @ x_old + t_align.
    Returns the same shape as input."""
    R_cam = extrinsic[:3, :3]
    t_cam = extrinsic[:3, 3]
    R_new = R_cam @ R_align.T
    t_new = s * t_cam - R_cam @ R_align.T @ t_align
    out = np.zeros_like(extrinsic)
    out[:3, :3] = R_new
    out[:3, 3] = t_new
    if extrinsic.shape[0] == 4:
        out[3, 3] = 1.0
    return out


def transform_points(pts, s, R_align, t_align):
    """Transform 3D points by similarity: x_new = s * R_align @ x + t_align."""
    return (s * (R_align @ pts.T)).T + t_align


# ──────────────────────── VGGT forward pass ────────────────────────

def vggt_forward(model, images, dtype):
    """Run VGGT at 518x518. Returns extrinsic(N,4,4), intrinsic(N,3,3), depth(N,H,W), conf(N,H,W)."""
    imgs_518 = F.interpolate(images, size=(VGGT_RES, VGGT_RES),
                             mode="bilinear", align_corners=False)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            agg, ps_idx = model.aggregator(imgs_518[None])
        pose_enc = model.camera_head(agg)[-1]
        ext, intr = pose_encoding_to_extri_intri(pose_enc, imgs_518[None].shape[-2:])
        depth, dconf = model.depth_head(agg, imgs_518[None], ps_idx)

    ext_np = ext.squeeze(0).cpu().numpy()    # (N, 3, 4)
    intr_np = intr.squeeze(0).cpu().numpy()  # (N, 3, 3)
    depth_np = depth.squeeze(0).cpu().numpy()
    dconf_np = dconf.squeeze(0).cpu().numpy()

    # Keep extrinsics as (N, 3, 4) -- VGGT utilities expect this format

    # Ensure depth is (N, H, W) -- VGGT may return (N, H, W, 1)
    if depth_np.ndim == 4 and depth_np.shape[-1] == 1:
        depth_np = depth_np[..., 0]

    return ext_np, intr_np, depth_np, dconf_np


# ──────────────────────── Chunking + merge ────────────────────────

def compute_chunks(total, chunk_size, overlap):
    """Return list of (start, end) ranges with the given overlap."""
    if total <= chunk_size:
        return [(0, total)]
    chunks = []
    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        chunks.append((start, end))
        if end == total:
            break
        start = end - overlap
    return chunks


def run_chunked_vggt(model, all_image_paths, device, dtype):
    """Run VGGT on overlapping chunks, align, and merge.
    Returns merged extrinsics(N,4,4), intrinsics(N,3,3), depth(N,H,W), conf(N,H,W)."""
    N = len(all_image_paths)
    chunks = compute_chunks(N, CHUNK_SIZE, OVERLAP)
    print(f"  Chunking {N} frames into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={OVERLAP})")
    for i, (s, e) in enumerate(chunks):
        print(f"    Chunk {i}: frames [{s}..{e}) ({e - s} frames)")

    # Storage: one entry per frame; filled as chunks are processed
    all_ext = [None] * N
    all_intr = [None] * N
    all_depth = [None] * N
    all_dconf = [None] * N
    # For alignment: camera centers per chunk, indexed by global frame id
    chunk_centers = []

    pbar = tqdm(chunks, desc="Chunks", unit="chunk")
    for ci, (start, end) in enumerate(pbar):
        chunk_paths = all_image_paths[start:end]
        n_chunk = len(chunk_paths)
        pbar.set_postfix(frames=f"{start}-{end-1}", n=n_chunk)

        imgs, orig_coords = load_and_preprocess_images_square(chunk_paths, VGGT_RES)
        imgs = imgs.to(device)

        ext, intr, depth, dconf = vggt_forward(model, imgs, dtype)
        torch.cuda.empty_cache()
        print(f"  Chunk {ci} output shapes: ext={ext.shape}, intr={intr.shape}, "
              f"depth={depth.shape}, dconf={dconf.shape}", flush=True)

        # Save per-frame results
        centers = camera_centers_from_extrinsic(ext)  # (n_chunk, 3)

        if ci == 0:
            # First chunk: defines the global coordinate system
            for j in range(n_chunk):
                gid = start + j
                all_ext[gid] = ext[j]
                all_intr[gid] = intr[j]
                all_depth[gid] = depth[j]
                all_dconf[gid] = dconf[j]
            chunk_centers.append({"start": start, "end": end, "centers": centers})
        else:
            # Align to global frame using overlapping camera centers
            prev = chunk_centers[-1]
            # Overlapping global frame indices
            ovlp_start = start  # first frame of current chunk
            ovlp_end = prev["end"]  # last frame of previous chunk
            ovlp_global = list(range(ovlp_start, ovlp_end))

            if len(ovlp_global) < 3:
                print(f"  WARNING: only {len(ovlp_global)} overlap frames; alignment may be poor")

            # Gather reference positions (already aligned to global frame)
            ref_positions = np.array([
                camera_centers_from_extrinsic(all_ext[gid][None])[0]
                for gid in ovlp_global
            ])
            # Corresponding positions in current chunk's local frame
            local_indices = [gid - start for gid in ovlp_global]
            local_positions = centers[local_indices]

            s_align, R_align, t_align = umeyama_alignment(local_positions, ref_positions)
            residual = np.linalg.norm(
                transform_points(local_positions, s_align, R_align, t_align) - ref_positions,
                axis=1
            ).mean()
            print(f"  Alignment: scale={s_align:.4f}, mean residual={residual:.6f}")

            # Apply transform and store (skip frames already filled by previous chunk)
            for j in range(n_chunk):
                gid = start + j
                if all_ext[gid] is not None:
                    continue  # already have a result from a previous chunk
                all_ext[gid] = transform_extrinsic(ext[j], s_align, R_align, t_align)
                all_intr[gid] = intr[j]  # intrinsics are frame-local, unaffected
                all_depth[gid] = depth[j] * s_align  # scale depth by similarity scale
                all_dconf[gid] = dconf[j]

            aligned_centers = transform_points(centers, s_align, R_align, t_align)
            chunk_centers.append({"start": start, "end": end, "centers": aligned_centers})

    # Validate: check for unfilled frames
    missing = [i for i in range(N) if all_ext[i] is None]
    if missing:
        print(f"  ERROR: {len(missing)} frames have no pose: {missing[:10]}...")
        sys.exit(1)

    # Debug: print shapes of first and last frame to diagnose np.stack errors
    print(f"  Frame 0 shapes: ext={all_ext[0].shape}, intr={all_intr[0].shape}, "
          f"depth={all_depth[0].shape}, dconf={all_dconf[0].shape}", flush=True)
    print(f"  Frame {N-1} shapes: ext={all_ext[N-1].shape}, intr={all_intr[N-1].shape}, "
          f"depth={all_depth[N-1].shape}, dconf={all_dconf[N-1].shape}", flush=True)

    # Find any shape mismatches
    ref_shapes = {
        "ext": all_ext[0].shape, "intr": all_intr[0].shape,
        "depth": all_depth[0].shape, "dconf": all_dconf[0].shape,
    }
    for i in range(N):
        for name, arr_list in [("ext", all_ext), ("intr", all_intr),
                               ("depth", all_depth), ("dconf", all_dconf)]:
            if arr_list[i].shape != ref_shapes[name]:
                print(f"  MISMATCH frame {i} {name}: {arr_list[i].shape} vs {ref_shapes[name]}",
                      flush=True)

    # Depth maps may differ in spatial resolution across chunks;
    # resize all to the resolution of the first frame.
    target_shape = all_depth[0].shape
    for i in range(N):
        if all_depth[i].shape != target_shape:
            all_depth[i] = np.array(
                PILImage.fromarray(all_depth[i].astype(np.float32)).resize(
                    (target_shape[1], target_shape[0]), PILImage.BILINEAR))
            all_dconf[i] = np.array(
                PILImage.fromarray(all_dconf[i].astype(np.float32)).resize(
                    (target_shape[1], target_shape[0]), PILImage.BILINEAR))

    # Stack individually with error reporting
    try:
        stacked_ext = np.stack(all_ext)
    except ValueError as e:
        shapes = set(tuple(a.shape) for a in all_ext)
        print(f"  STACK FAIL ext: unique shapes = {shapes}")
        raise
    try:
        stacked_intr = np.stack(all_intr)
    except ValueError as e:
        shapes = set(tuple(a.shape) for a in all_intr)
        print(f"  STACK FAIL intr: unique shapes = {shapes}")
        raise
    try:
        stacked_depth = np.stack(all_depth)
    except ValueError as e:
        shapes = set(tuple(a.shape) for a in all_depth)
        print(f"  STACK FAIL depth: unique shapes = {shapes}")
        raise
    try:
        stacked_dconf = np.stack(all_dconf)
    except ValueError as e:
        shapes = set(tuple(a.shape) for a in all_dconf)
        print(f"  STACK FAIL dconf: unique shapes = {shapes}")
        raise

    return stacked_ext, stacked_intr, stacked_depth, stacked_dconf


# ──────────────────────── COLMAP export ────────────────────────

def build_and_write_colmap(extrinsic, intrinsic, depth_map, depth_conf,
                           images_on_gpu, image_names, original_coords,
                           conf_threshold, max_points, output_dir):
    """Build pycolmap reconstruction and write to disk."""
    if not HAS_PYCOLMAP:
        print("ERROR: pycolmap not installed. Cannot write COLMAP model.")
        print("Install: pip install pycolmap==3.10.0 pyceres==2.3")
        sys.exit(1)

    N = extrinsic.shape[0]
    H = W = VGGT_RES
    image_size = np.array([VGGT_RES, VGGT_RES])

    # --- Subsample BEFORE unprojecting to avoid huge memory allocation ---
    # Instead of unprojecting all 199*518*518 points then filtering,
    # first pick which pixels pass the confidence threshold, then unproject only those.
    conf_mask = depth_conf >= conf_threshold
    n_valid = conf_mask.sum()
    print(f"  Confident points: {n_valid:,} / {N*H*W:,} (threshold={conf_threshold})", flush=True)
    conf_mask = randomly_limit_trues(conf_mask, max_points)
    n_selected = conf_mask.sum()
    print(f"  Selected for export: {n_selected:,} points", flush=True)

    # Gather selected pixel coords (frame_idx, row, col)
    sel_frames, sel_rows, sel_cols = np.where(conf_mask)

    # Unproject only selected points (one per selected pixel)
    print("  Unprojecting selected points to 3D...", flush=True)
    all_pts3d = []
    all_rgb = []
    all_xyf = []

    imgs_518 = F.interpolate(images_on_gpu, size=(VGGT_RES, VGGT_RES),
                             mode="bilinear", align_corners=False)
    imgs_np = (imgs_518.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
    del imgs_518
    torch.cuda.empty_cache()

    for fi in tqdm(range(N), desc="  Unprojecting", leave=False):
        mask_fi = sel_frames == fi
        if not mask_fi.any():
            continue
        rows_fi = sel_rows[mask_fi]
        cols_fi = sel_cols[mask_fi]

        # Unproject this single frame
        cam_pts = depth_to_cam_coords_points(depth_map[fi], intrinsic[fi])  # (H,W,3)
        c2w = closed_form_inverse_se3(extrinsic[fi][None])[0]  # (4,4)
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        world_pts = cam_pts[rows_fi, cols_fi] @ R_c2w.T + t_c2w  # (K, 3)

        all_pts3d.append(world_pts)
        all_rgb.append(imgs_np[fi, rows_fi, cols_fi])  # (K, 3)
        all_xyf.append(np.stack([cols_fi.astype(np.float32),
                                 rows_fi.astype(np.float32),
                                 np.full(len(rows_fi), fi, dtype=np.float32)], axis=1))

    all_pts3d = np.concatenate(all_pts3d)
    all_rgb = np.concatenate(all_rgb)
    all_xyf = np.concatenate(all_xyf)
    del imgs_np

    print(f"  Total 3D points: {len(all_pts3d):,}", flush=True)

    print("  Building pycolmap reconstruction...", flush=True)
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        all_pts3d, all_xyf, all_rgb,
        extrinsic, intrinsic, image_size,
        shared_camera=False, camera_type="PINHOLE",
    )
    print(f"  Reconstruction built: {len(reconstruction.cameras)} cameras, "
          f"{len(reconstruction.points3D)} points", flush=True)
    del all_pts3d, all_rgb, all_xyf

    for pyimgid in tqdm(reconstruction.images, desc="Rescaling cameras"):
        pyimg = reconstruction.images[pyimgid]
        pycam = reconstruction.cameras[pyimg.camera_id]
        pyimg.name = image_names[pyimgid - 1]

        params = copy.deepcopy(pycam.params)
        real_size = original_coords[pyimgid - 1, -2:]
        ratio = max(real_size) / VGGT_RES
        params = params * ratio
        params[-2:] = real_size / 2
        pycam.params = params
        pycam.width = real_size[0]
        pycam.height = real_size[1]

        top_left = original_coords[pyimgid - 1, :2]
        for pt2d in pyimg.points2D:
            pt2d.xy = (pt2d.xy - top_left) * ratio

    print("  Writing COLMAP model to disk...", flush=True)
    for subdir in ["sparse", os.path.join("sparse", "0")]:
        out = os.path.join(output_dir, subdir)
        os.makedirs(out, exist_ok=True)
        reconstruction.write(out)

    import trimesh
    vis_pts = np.array([reconstruction.points3D[pid].xyz for pid in reconstruction.points3D])
    if len(vis_pts) > max_points:
        vis_pts = vis_pts[np.random.choice(len(vis_pts), max_points, replace=False)]
    trimesh.PointCloud(vis_pts).export(os.path.join(output_dir, "sparse", "points.ply"))
    print(f"  Exported {len(vis_pts)} points to points.ply", flush=True)

    n_c = len(reconstruction.cameras)
    n_i = len(reconstruction.images)
    n_p = len(reconstruction.points3D)
    return n_c, n_i, n_p


# ──────────────────────── Main ────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="VGGT -> COLMAP (chunked, all frames)")
    p.add_argument("--scene_dir", type=str, required=True)
    p.add_argument("--images_subdir", type=str, default="images")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    p.add_argument("--overlap", type=int, default=OVERLAP)
    p.add_argument("--conf_threshold", type=float, default=5.0)
    p.add_argument("--max_points", type=int, default=100000)
    p.add_argument("--model_path", type=str, default=None,
                    help="Local path to VGGT model.pt (skips download)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    output_dir = args.output_dir or args.scene_dir

    # Discover images
    img_dir = os.path.join(args.scene_dir, args.images_subdir)
    paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    paths = [p for p in paths if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    N = len(paths)
    if N == 0:
        print(f"ERROR: no images in {img_dir}"); sys.exit(1)

    print(f"Scene: {args.scene_dir}")
    print(f"Total frames: {N}")
    print(f"Chunk size: {args.chunk_size}, Overlap: {args.overlap}")
    n_chunks = len(compute_chunks(N, args.chunk_size, args.overlap))
    print(f"Estimated chunks: {n_chunks}")
    print(f"Per-chunk VRAM: ~{1.5 + args.chunk_size * 0.195:.1f} GB\n")

    # Load model
    print("Loading VGGT model...")
    model = VGGT()
    if args.model_path:
        print(f"  Loading from local: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=False))
    else:
        # Search common cache locations
        hub_dir = torch.hub.get_dir()
        candidates = [
            os.path.join(hub_dir, "checkpoints", "model.pt"),
            os.path.join(hub_dir, "checkpoints", "vggt", "model.pt"),
        ]
        cache_path = None
        for c in candidates:
            if os.path.isfile(c):
                cache_path = c
                break
        if cache_path:
            print(f"  Loading from cache: {cache_path}")
            model.load_state_dict(torch.load(cache_path, map_location="cpu", weights_only=False))
        else:
            print("  Downloading from HuggingFace...")
            url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            model.load_state_dict(torch.hub.load_state_dict_from_url(url))
    model.eval().to(device)
    torch.cuda.empty_cache()
    print("Model loaded.\n")

    # Run chunked inference + alignment
    global CHUNK_SIZE, OVERLAP
    CHUNK_SIZE = args.chunk_size
    OVERLAP = args.overlap
    extrinsic, intrinsic, depth, dconf = run_chunked_vggt(model, paths, device, dtype)
    print(f"\nMerged: {extrinsic.shape[0]} frames with poses")

    # Reload all images at 518 for color extraction
    print("Loading all images for COLMAP export...")
    print(f"  Loading {N} images at {VGGT_RES}x{VGGT_RES}...")
    images, orig_coords = load_and_preprocess_images_square(tqdm(paths, desc="Loading images"), VGGT_RES)
    images = images.to(device)
    orig_coords_np = orig_coords.numpy()
    base_names = [os.path.basename(p) for p in paths]

    # Build and write COLMAP
    print("Building COLMAP reconstruction...")
    nc, ni, np_ = build_and_write_colmap(
        extrinsic, intrinsic, depth, dconf,
        images, base_names, orig_coords_np,
        args.conf_threshold, args.max_points, output_dir,
    )

    info_path = os.path.join(output_dir, "sparse", "info.txt")
    with open(info_path, "w") as f:
        f.write(f"vggt_chunked_N{N}_chunk{args.chunk_size}_ovlp{args.overlap}_"
                f"conf{args.conf_threshold}_pts{args.max_points}\n")

    print(f"\nSaved to: {os.path.join(output_dir, 'sparse')}")
    print(f"  Cameras: {nc}, Images: {ni}, Points3D: {np_}")
    print("Done.")


if __name__ == "__main__":
    with torch.no_grad():
        main()
