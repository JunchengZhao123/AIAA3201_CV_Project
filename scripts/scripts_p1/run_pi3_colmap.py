"""
Pi3 -> COLMAP Export  (Chunked, OOM-safe for A6000 48GB)
========================================================
Processes ALL frames by splitting into overlapping chunks, running Pi3 on
each chunk, aligning coordinate systems via Procrustes on overlapping
camera centers, and merging into a single COLMAP sparse model.

Usage:
    python scripts/run_pi3_colmap.py --scene_dir dataset/DL3DV-2
    python scripts/run_pi3_colmap.py --scene_dir dataset/Mip-NeRF/bicycle --output_dir out/bicycle
"""

import os, sys, glob, argparse, random, math, struct, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision import transforms as TF
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "Pi3"))

from pi3.utils.geometry import depth_edge, recover_intrinsic_from_rays_d
from pi3.models.pi3 import Pi3

CHUNK_SIZE = 100
OVERLAP = 20


# ──────────── Alignment (shared with VGGT script) ────────────

def umeyama_alignment(src, dst):
    """src, dst: (K,3).  Returns s, R(3,3), t(3,)  s.t.  s*R@src + t ≈ dst."""
    K = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    H = sc.T @ dc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    s = (S * np.array([1.0, 1.0, d])).sum() / ((sc ** 2).sum() / K)
    t = mu_d - s * R @ mu_s
    return s, R, t


def transform_c2w(c2w, s, R_a, t_a):
    """Apply similarity (s,R_a,t_a) to a 4x4 camera-to-world matrix."""
    out = np.eye(4, dtype=c2w.dtype)
    out[:3, :3] = R_a @ c2w[:3, :3]
    out[:3, 3] = s * R_a @ c2w[:3, 3] + t_a
    return out


def transform_points(pts, s, R, t):
    return (s * (R @ pts.T)).T + t


# ──────────── COLMAP binary writers ────────────

def rotmat_to_qvec(R):
    Rxx, Ryx, Rzx = R[0,0], R[1,0], R[2,0]
    Rxy, Ryy, Rzy = R[0,1], R[1,1], R[2,1]
    Rxz, Ryz, Rzz = R[0,2], R[1,2], R[2,2]
    K = np.array([
        [Rxx-Ryy-Rzz, 0, 0, 0],
        [Ryx+Rxy, Ryy-Rxx-Rzz, 0, 0],
        [Rzx+Rxz, Rzy+Ryz, Rzz-Rxx-Ryy, 0],
        [Ryz-Rzy, Rzx-Rxz, Rxy-Ryx, Rxx+Ryy+Rzz],
    ]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3,0,1,2], np.argmax(vals)]
    if q[0] < 0: q *= -1
    return q


def write_cameras_bin(cameras, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cid, c in cameras.items():
            f.write(struct.pack("<I", cid))
            f.write(struct.pack("<i", c["model"]))
            f.write(struct.pack("<Q", c["w"]))
            f.write(struct.pack("<Q", c["h"]))
            for p in c["params"]:
                f.write(struct.pack("<d", p))


def write_images_bin(images, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for iid, im in images.items():
            f.write(struct.pack("<I", iid))
            for q in im["qvec"]: f.write(struct.pack("<d", q))
            for t in im["tvec"]: f.write(struct.pack("<d", t))
            f.write(struct.pack("<I", im["cam_id"]))
            f.write(im["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # no 2D points


def write_points3d_bin(pts, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for pid, p in pts.items():
            f.write(struct.pack("<Q", pid))
            for v in p["xyz"]: f.write(struct.pack("<d", v))
            for c in p["rgb"]: f.write(struct.pack("<B", c))
            f.write(struct.pack("<d", 0.0))
            f.write(struct.pack("<Q", len(p["track"])))
            for im, idx in p["track"]:
                f.write(struct.pack("<II", im, idx))


def write_ply(pts, rgb, path):
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {len(pts)}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "end_header\n")
    with open(path, "wb") as f:
        f.write(hdr.encode("ascii"))
        for i in range(len(pts)):
            f.write(struct.pack("<fff", *pts[i].astype(np.float32)))
            f.write(struct.pack("<BBB", *rgb[i].astype(np.uint8)))


# ──────────── Chunking ────────────

def compute_chunks(total, chunk_size, overlap):
    if total <= chunk_size:
        return [(0, total)]
    chunks, start = [], 0
    while start < total:
        end = min(start + chunk_size, total)
        chunks.append((start, end))
        if end == total: break
        start = end - overlap
    return chunks


def load_images_batch(paths, device, show_progress=False):
    """Load images, resize to the first image's dimensions, return (N,3,H,W) tensor."""
    to_tensor = TF.ToTensor()
    it = tqdm(paths, desc="  Loading", leave=False) if show_progress else paths
    tensors = [to_tensor(Image.open(p).convert("RGB")) for p in it]
    H0, W0 = tensors[0].shape[1], tensors[0].shape[2]
    out = []
    for t in tensors:
        if t.shape[1] != H0 or t.shape[2] != W0:
            t = F.interpolate(t.unsqueeze(0), size=(H0, W0),
                              mode="bilinear", align_corners=False).squeeze(0)
        out.append(t)
    return torch.stack(out).to(device), H0, W0


def run_pi3_chunk(model, imgs, dtype):
    """Run Pi3 on (N,3,H,W).  Returns c2w(N,4,4), local_pts(N,H,W,3), global_pts(N,H,W,3), conf(N,H,W)."""
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            res = model(imgs[None])
    c2w = res["camera_poses"][0].float().cpu().numpy()
    lp = res["local_points"]
    gp = res["points"][0].float().cpu().numpy()
    conf = torch.sigmoid(res["conf"][0, ..., 0]).cpu().numpy()
    non_edge = (~depth_edge(lp[..., 2], rtol=0.03))[0].cpu().numpy()

    rays_d = F.normalize(lp, dim=-1)
    K = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)
    K = K[0].float().cpu().numpy()

    return c2w, gp, conf & (conf > 0) & non_edge, K  # valid_mask is boolean


def run_chunked_pi3(model, all_paths, device, dtype, chunk_size, overlap, conf_thr):
    N = len(all_paths)
    chunks = compute_chunks(N, chunk_size, overlap)
    print(f"  Chunking {N} frames into {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")

    all_c2w = [None] * N
    all_gpts = [None] * N
    all_valid = [None] * N
    all_K = [None] * N

    prev_centers = None

    pbar = tqdm(chunks, desc="Chunks", unit="chunk")
    for ci, (start, end) in enumerate(pbar):
        n = end - start
        pbar.set_postfix(frames=f"{start}-{end-1}", n=n)

        imgs, H, W = load_images_batch(all_paths[start:end], device, show_progress=True)
        c2w, gpts, valid, K = run_pi3_chunk(model, imgs, dtype)
        valid = valid & (torch.sigmoid(torch.tensor(0.0)).item() > -1)  # ensure bool
        # Recompute valid with conf threshold
        # valid was already computed in run_pi3_chunk; we keep it as-is (conf > 0 & non-edge)
        torch.cuda.empty_cache()

        centers = c2w[:, :3, 3]  # camera centers from c2w

        if ci == 0:
            for j in range(n):
                all_c2w[start + j] = c2w[j]
                all_gpts[start + j] = gpts[j]
                all_valid[start + j] = valid[j]
                all_K[start + j] = K[j]
            prev_centers = {start + j: centers[j] for j in range(n)}
        else:
            ovlp_gids = [gid for gid in range(start, min(end, start + overlap + (end - start - overlap)))
                         if all_c2w[gid] is not None]
            ovlp_gids = [gid for gid in range(start, end) if all_c2w[gid] is not None]

            if len(ovlp_gids) < 3:
                print(f"  WARNING: only {len(ovlp_gids)} overlap frames")

            ref_pos = np.array([prev_centers[gid] for gid in ovlp_gids])
            loc_pos = centers[[gid - start for gid in ovlp_gids]]

            s, R, t = umeyama_alignment(loc_pos, ref_pos)
            residual = np.linalg.norm(transform_points(loc_pos, s, R, t) - ref_pos, axis=1).mean()
            print(f"  Alignment: scale={s:.4f}, mean_residual={residual:.6f}")

            for j in range(n):
                gid = start + j
                if all_c2w[gid] is not None:
                    continue
                all_c2w[gid] = transform_c2w(c2w[j], s, R, t)
                all_gpts[gid] = transform_points(
                    gpts[j].reshape(-1, 3), s, R, t
                ).reshape(gpts[j].shape)
                all_valid[gid] = valid[j]
                all_K[gid] = K[j]
                prev_centers[gid] = all_c2w[gid][:3, 3]

    return all_c2w, all_gpts, all_valid, all_K


# ──────────── Main ────────────

def parse_args():
    p = argparse.ArgumentParser(description="Pi3 -> COLMAP (chunked, all frames)")
    p.add_argument("--scene_dir", type=str, required=True)
    p.add_argument("--images_subdir", type=str, default="images")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    p.add_argument("--overlap", type=int, default=OVERLAP)
    p.add_argument("--conf_threshold", type=float, default=0.1)
    p.add_argument("--max_points", type=int, default=100000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir or args.scene_dir

    img_dir = os.path.join(args.scene_dir, args.images_subdir)
    paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    paths = [p for p in paths if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    N = len(paths)
    if N == 0:
        print(f"ERROR: no images in {img_dir}"); sys.exit(1)

    print(f"Scene: {args.scene_dir}  |  Total: {N} frames")
    print(f"Chunk: {args.chunk_size}, Overlap: {args.overlap}\n")

    print("Loading Pi3 model...")
    if args.ckpt:
        model = Pi3().to(device).eval()
        if args.ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file
            w = load_file(args.ckpt, device=str(device))
        else:
            w = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(w)
        print(f"  Loaded from: {args.ckpt}")
    else:
        hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface",
                                "hub", "models--yyfz233--Pi3")
        if os.path.isdir(hf_cache):
            print(f"  Loading from HF cache: {hf_cache}")
        else:
            print("  Downloading from HuggingFace...")
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    torch.cuda.empty_cache()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print("Model loaded.\n")

    all_c2w, all_gpts, all_valid, all_K = run_chunked_pi3(
        model, paths, device, dtype, args.chunk_size, args.overlap, args.conf_threshold
    )

    # Determine image size from first image
    first_img = Image.open(paths[0]).convert("RGB")
    W_orig, H_orig = first_img.size
    names = [os.path.basename(p) for p in paths]

    print(f"\nBuilding COLMAP model ({N} frames, {W_orig}x{H_orig})...")

    cameras, images_d, pts3d = {}, {}, {}
    all_pts_list, all_rgb_list, all_fid_list = [], [], []
    to_tensor = TF.ToTensor()

    for i in tqdm(range(N), desc="Building COLMAP entries"):
        cid = i + 1
        K = all_K[i]
        cameras[cid] = {"model": 1, "w": W_orig, "h": H_orig,
                         "params": [K[0,0], K[1,1], K[0,2], K[1,2]]}

        w2c = np.linalg.inv(all_c2w[i])
        images_d[cid] = {"qvec": rotmat_to_qvec(w2c[:3,:3]).tolist(),
                          "tvec": w2c[:3,3].tolist(),
                          "cam_id": cid, "name": names[i]}

        gp = all_gpts[i]
        vm = all_valid[i]
        img_np = (to_tensor(Image.open(paths[i]).convert("RGB")).numpy() * 255).astype(np.uint8)
        img_np = img_np.transpose(1, 2, 0)  # (H,W,3)
        # Resize valid mask if needed
        if vm.shape != (img_np.shape[0], img_np.shape[1]):
            from scipy.ndimage import zoom
            vm = zoom(vm.astype(float), (img_np.shape[0]/vm.shape[0], img_np.shape[1]/vm.shape[1]), order=0) > 0.5
            gp_flat = gp.reshape(-1, 3)
        if vm.any():
            all_pts_list.append(gp[vm])
            all_rgb_list.append(img_np[vm[:img_np.shape[0], :img_np.shape[1]]] if vm.shape == img_np.shape[:2] else img_np.reshape(-1,3)[vm.flatten()])
            all_fid_list.append(np.full(vm.sum(), i))

    all_pts = np.concatenate(all_pts_list)
    all_rgb = np.concatenate(all_rgb_list)
    all_fid = np.concatenate(all_fid_list)

    if len(all_pts) > args.max_points:
        idx = np.random.choice(len(all_pts), args.max_points, replace=False)
        all_pts, all_rgb, all_fid = all_pts[idx], all_rgb[idx], all_fid[idx]

    for j in range(len(all_pts)):
        pts3d[j+1] = {"xyz": all_pts[j].tolist(), "rgb": all_rgb[j].tolist(),
                       "track": [(int(all_fid[j])+1, 0)]}

    print(f"  Cameras: {len(cameras)}, Images: {len(images_d)}, Points3D: {len(pts3d)}")

    for sub in ["sparse", os.path.join("sparse", "0")]:
        d = os.path.join(output_dir, sub)
        os.makedirs(d, exist_ok=True)
        write_cameras_bin(cameras, os.path.join(d, "cameras.bin"))
        write_images_bin(images_d, os.path.join(d, "images.bin"))
        write_points3d_bin(pts3d, os.path.join(d, "points3D.bin"))

    write_ply(all_pts, all_rgb, os.path.join(output_dir, "sparse", "0", "points3D.ply"))

    with open(os.path.join(output_dir, "sparse", "info.txt"), "w") as f:
        f.write(f"pi3_chunked_N{N}_chunk{args.chunk_size}_ovlp{args.overlap}\n")

    print(f"\nSaved to: {os.path.join(output_dir, 'sparse')}")
    print("Done.")


if __name__ == "__main__":
    main()
