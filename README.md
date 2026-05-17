# AIAA3201 CV Project: Generative Sparse-View 3D Reconstruction

A progressive pipeline for sparse-view 3D reconstruction, covering initialization analysis, unposed monocular SLAM, and generative enhancement with confidence-aware optimization.

**Course:** AIAA3201 Computer Vision, HKUST(GZ)  

---

## Project Structure

```
├── scripts/
│   ├── scripts_p1/          # Part 1: Initialization & 3DGS training
│   │   ├── run_planB_all.sh         # VGGT/Pi3 pose estimation
│   │   ├── run_3dgs_training.sh     # 3DGS training (Plan A & B)
│   │   ├── run_vggt_colmap.py       # VGGT → COLMAP format converter
│   │   ├── run_pi3_colmap.py        # Pi3 → COLMAP format converter
│   │   ├── analyze_part1.py         # Metrics analysis & convergence plots
│   │   ├── undistort_colmap.py      # COLMAP undistortion
│   │   └── verify_colmap_outputs.py # Output verification
│   ├── scripts_p2/          # Part 2: Sparse SLAM (S3PO-GS)
│   │   ├── run_s3po_all.sh          # Run S3PO-GS on all datasets
│   │   ├── subsample_sparse.py      # Create sparse frame subsets
│   │   ├── evaluate_p2.py           # Pose & rendering evaluation
│   │   ├── render_test_views_s3po.py# Novel test view rendering
│   │   ├── add_re10k_parser.py      # Patch S3PO-GS for Re10k
│   │   ├── downscale_waymo.py       # Waymo image downscaling
│   │   └── configs/                 # Per-dataset YAML configs
│   └── scripts_p3/          # Part 3: Generative enhancement
│       ├── run_full_pipeline_p3.sh   # End-to-end pipeline
│       ├── generate_pseudo_views.py  # RIFE + 3DGS pseudo-view synthesis
│       ├── compute_confidence_masks.py # Multi-signal confidence masks
│       ├── hybrid_train_3dgs.py      # Confidence-weighted hybrid training
│       ├── evaluate_p3.py            # Ablation evaluation & comparison
│       ├── setup_p3.sh              # Dependency setup
│       └── configs/                  # Per-dataset YAML configs
├── running_result/
│   ├── 5_1/                 # Part 1 results
│   │   ├── 5_1_1_Pose & Point Cloud Initialization/
│   │   │   ├── PlanA_COLMAP_output/   # COLMAP poses & point clouds
│   │   │   └── PlanB_VGGT_output/     # VGGT poses & point clouds
│   │   └── 5_1_2_3DGS_Optimization/
│   │       ├── experiments/           # Per-scene 3DGS results (renders, metrics)
│   │       └── analysis/              # Aggregated metrics & convergence plots
│   ├── 5_2/                 # Part 2 results (S3PO-GS SLAM)
│   │   └── results/                   # Per-dataset SLAM outputs & metrics
│   └── 5_3/                 # Part 3 results
│       ├── pseudo_views/              # Generated pseudo-views
│       ├── confidence_masks/          # Per-pixel confidence masks
│       ├── hybrid_train/              # Fine-tuned models (3 modes)
│       └── evaluation/                # Comparison renders & metrics
├── report/
│   ├── main.tex             # CVPR-format paper
│   ├── main.bib             # Bibliography
│   ├── background.md        # Technical background
│   ├── reference.md         # Reference papers
│   ├── requirements.md      # Report requirements
│   └── experiments/         # Detailed experiment documentation
│       ├── p1.md            # Part 1 experiments
│       ├── p2.md            # Part 2 experiments
│       └── p3.md            # Part 3 experiments
├── gaussian-splatting/      # External 3DGS codebase (clone separately)
├── vggt/                    # External VGGT codebase (clone separately)
├── S3PO-GS/                 # External S3PO-GS codebase (clone separately)
├── Practical-RIFE/          # External RIFE codebase (clone separately)
├── Difix3D/ or difix/       # External Difix3D / nvidia-difix files
├── checkpoints/             # Model checkpoints and HuggingFace downloads
└── dataset/                 # Raw datasets (not included in repo)
```

---

## Setup From the GitHub Submission

The submitted GitHub repository is mainly intended to provide the project scripts and documentation. Large datasets, generated results, and third-party research codebases are not stored in this repo. Reproduce the working tree by cloning this repo first, then cloning the required external projects beside `scripts/`.

### Clone This Project / Scripts

Full clone:

```bash
git clone https://github.com/JunchengZhao123/AIAA3201_CV_Project.git
cd AIAA3201_CV_Project
```

If you only want the submitted `scripts/` directory and README:

```bash
git clone --filter=blob:none --sparse https://github.com/JunchengZhao123/AIAA3201_CV_Project.git
cd AIAA3201_CV_Project
git sparse-checkout set scripts README.md
```

Expected root layout after external dependencies are cloned:

```bash
AIAA3201_CV_Project/
├── scripts/
├── gaussian-splatting/
├── vggt/
├── S3PO-GS/
├── Practical-RIFE/
├── Difix3D/                 # or difix/ if using a HuggingFace snapshot manually
├── checkpoints/
├── dataset/
└── running_result/
```

The scripts assume this root layout by default. If your paths differ, set environment variables such as `CV_PROJ_ROOT`, `S3PO_ROOT`, `RIFE_ROOT`, and `DIFIX3D_ROOT` before running the shell scripts.

---

## External Codebases and Environments

### Hardware
- GPU: NVIDIA A6000 (48 GB) or equivalent (minimum 24 GB for most operations)
- RAM: 32 GB+
- Storage: ~100 GB for datasets + results

### Software

**Base environment:**
- Ubuntu 20.04+ / Linux
- CUDA 12.1+
- Conda (Miniconda or Anaconda)

### Part 1A: 3D Gaussian Splatting

Clone the official 3DGS repository recursively because it uses CUDA submodules:

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive

conda create -n gaussian_splatting python=3.8 -y
conda activate gaussian_splatting
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
cd gaussian-splatting && pip install -e . && cd ..
pip install plyfile tqdm
```

This environment is used by `scripts/scripts_p1/run_3dgs_training.sh`.

### Part 1B: VGGT Initialization

Clone VGGT beside `scripts/`:

```bash
git clone https://github.com/facebookresearch/vggt.git

conda create -n vggt python=3.10 -y
conda activate vggt
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
cd vggt && pip install -e . && cd ..
pip install trimesh pycolmap==3.10.0 pyceres==2.3 tqdm
```

VGGT weights are downloaded automatically by the VGGT code if not already cached.

### Part 2: S3PO-GS SLAM

Recommended automated setup:

```bash
bash scripts/scripts_p2/setup_s3po.sh
```

Manual equivalent:

```bash
git clone https://github.com/3DAgentWorld/S3PO-GS.git --recursive S3PO-GS
cd S3PO-GS && conda env create -f environment.yml
conda activate S3PO-GS
pip install -U "setuptools>=65,<82" wheel ninja
pip install submodules/simple-knn --no-build-isolation
pip install submodules/diff-gaussian-rasterization --no-build-isolation
cd croco/models/curope/ && python setup.py build_ext --inplace && cd ../../../
pip install evo scipy scikit-image torchmetrics lpips
```

The setup script also patches S3PO-GS with the Re10k parser and sparse configs used in this project.

### Part 3A: RIFE Frame Interpolation

Part 3 uses Practical-RIFE for intermediate frame synthesis. The automated Part 3 setup clones it, but the manual steps are:

```bash
git clone https://github.com/hzwer/Practical-RIFE.git Practical-RIFE
conda activate S3PO-GS
cd Practical-RIFE
pip install -r requirements.txt
cd ..
```

Download a RIFE model and place the files under `Practical-RIFE/train_log/`. The project was configured for the Practical-RIFE `train_log` layout expected by `generate_pseudo_views.py`, including files such as `flownet.pkl`, `contextnet.pkl`, and `unet.pkl`.

### Part 3B: Difix3D / nvidia-difix Cleanup

Part 3 explicitly uses the Difix/Difix3D model for single-step diffusion cleanup of generated pseudo-views. This is important because `generate_pseudo_views.py` loads the HuggingFace `nvidia/difix` model with custom files such as `pipeline_difix.py` and `autoencoder_kl.py`.

Clone the Difix3D code if available:

```bash
git clone https://github.com/nv-tlabs/Difix3D.git Difix3D
```

Install diffusion dependencies into the reused `S3PO-GS` environment:

```bash
conda activate S3PO-GS
pip install "numpy<2"
pip install "diffusers>=0.25.0,<0.31.0" \
            "transformers>=4.36.0,<4.46.0" \
            "accelerate>=0.25.0,<0.34.0" \
            safetensors peft huggingface_hub matplotlib opencv-python-headless
```

Download the `nvidia/difix` model. If HuggingFace access requires authentication, run `huggingface-cli login` first:

```bash
huggingface-cli download nvidia/difix \
    --local-dir checkpoints/difix3d/nvidia_difix \
    --local-dir-use-symlinks False
```

Then pass this path to pseudo-view generation when needed:

```bash
python scripts/scripts_p3/generate_pseudo_views.py \
    --config scripts/scripts_p3/configs/dl3dv_p3.yaml \
    --difix_model_path checkpoints/difix3d/nvidia_difix
```

For convenience, the project setup script attempts to prepare RIFE, Difix3D-related dependencies, and checkpoint folders:

```bash
conda activate S3PO-GS
bash scripts/scripts_p3/setup_p3.sh
```

If Difix/Difix3D is unavailable on a machine, Part 3 can still be run with `--skip_difix`, but that does not reproduce the reported Part 3 pipeline because the submitted experiments used Difix-style cleanup.

### Evaluation Utilities

```bash
pip install evo lpips torchmetrics scikit-image matplotlib
```

---

## Usage

### Datasets

Download and organize datasets under `dataset/`:
```
dataset/
├── 405841/FRONT/images/    # Waymo-405841
├── DL3DV-2/images/         # DL3DV scene 2
└── Re10k-1/images/         # RealEstate10K scene 1
```

### Part 1: Initialization Analysis & 3DGS Training

```bash
# Step 1: COLMAP pose estimation (Plan A) — assumes COLMAP is installed
# Outputs stored in running_result/5_1/5_1_1_Pose & Point Cloud Initialization/PlanA_COLMAP_output/

# Step 2: VGGT pose estimation (Plan B)
conda activate vggt
bash scripts/scripts_p1/run_planB_all.sh vggt

# Step 3: 3DGS training (both plans)
conda activate gaussian_splatting
bash scripts/scripts_p1/run_3dgs_training.sh 3dgs planA
bash scripts/scripts_p1/run_3dgs_training.sh 3dgs planB

# Step 4: Analysis
python scripts/scripts_p1/analyze_part1.py \
    --exp_dir running_result/5_1/5_1_2_3DGS_Optimization/experiments
```

### Part 2: Unposed Sparse Reconstruction

```bash
conda activate S3PO-GS

# Step 1: Create sparse frame subsets
python scripts/scripts_p2/subsample_sparse.py

# Step 2: Run S3PO-GS SLAM
bash scripts/scripts_p2/run_s3po_all.sh

# Step 3: Evaluate
python scripts/scripts_p2/evaluate_p2.py
```

### Part 3: Generative Enhancement

```bash
conda activate S3PO-GS

# Run full pipeline (all datasets, all stages)
bash scripts/scripts_p3/run_full_pipeline_p3.sh

# Or run individual stages:
bash scripts/scripts_p3/run_full_pipeline_p3.sh --dataset waymo
bash scripts/scripts_p3/run_full_pipeline_p3.sh --skip_pseudo --skip_confidence  # train only
bash scripts/scripts_p3/run_full_pipeline_p3.sh --skip_train --skip_eval         # generate only

# If your nvidia/difix snapshot is not in the default path used by the script,
# run pseudo-view generation directly and provide the model path:
python scripts/scripts_p3/generate_pseudo_views.py \
    --config scripts/scripts_p3/configs/dl3dv_p3.yaml \
    --difix_model_path checkpoints/difix3d/nvidia_difix

# Custom parameters:
bash scripts/scripts_p3/run_full_pipeline_p3.sh \
    --iterations 20000 --beta_max 0.3 --n_interp 3
```

---

## Model Weights / Checkpoints

| Model | Source | Path |
|-------|--------|------|
| VGGT | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | `checkpoints/vggt_model.pt` or auto-downloaded to torch hub cache |
| S3PO-GS | [3DAgentWorld/S3PO-GS](https://github.com/3DAgentWorld/S3PO-GS) | Built-in (trained from scratch per scene) |
| Difix3D / Difix | [nvidia/difix](https://huggingface.co/nvidia/difix), [nv-tlabs/Difix3D](https://github.com/nv-tlabs/Difix3D) | `checkpoints/difix3d/nvidia_difix/` or a HuggingFace snapshot path passed via `--difix_model_path` |
| RIFE | [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE) | `Practical-RIFE/train_log/` |
| 3DGS (trained) | This project | `running_result/5_1/5_1_2_3DGS_Optimization/experiments/*/point_cloud/iteration_30000/` |
| S3PO-GS (trained) | This project | `running_result/5_3/hybrid_train/*/sparse_pseudo_confidence/point_cloud/final/` |

Place pre-trained model weights in the `checkpoints/` directory when possible. VGGT weights are automatically downloaded on first use if not found locally. Difix/Difix3D is the most important manual checkpoint for Part 3 because the pseudo-view generator loads the custom `nvidia/difix` pipeline files directly.

---

## Visual Results

### Part 1: COLMAP vs VGGT Initialization

| Scene | Plan A (COLMAP) | Plan B (VGGT) | Ground Truth |
|-------|----------------|---------------|--------------|
| Waymo-405841 | 31.22 dB PSNR | 25.83 dB PSNR | — |
| DL3DV-2 | 31.46 dB PSNR | 19.79 dB PSNR | — |
| Re10k-1 | 25.94 dB PSNR | 21.87 dB PSNR | — |

Rendered test views: `running_result/5_1/5_1_2_3DGS_Optimization/experiments/plan{A,B}_{scene}_3dgs_iter30000/test/ours_30000/renders/`

Convergence plots: `running_result/5_1/5_1_2_3DGS_Optimization/analysis/convergence_*.png`

### Part 2: Sparse-View SLAM Reconstruction

| Dataset | Sparsity | PSNR | SSIM | LPIPS |
|---------|----------|------|------|-------|
| Re10k-1 | 1/30 | 19.02 | 0.782 | 0.105 |
| DL3DV-2 | 1/30 | 16.81 | 0.520 | 0.363 |
| Waymo-405841 | 1/10 | 12.89 | 0.585 | 0.686 |

Trajectory and rendering outputs: `running_result/5_2/results/`

### Part 3: Generative Enhancement Ablation

| Method | PSNR | SSIM | LPIPS |
|--------|------|------|-------|
| P2 Baseline | 8.25 | 0.151 | 0.747 |
| Retrain | 7.23 | 0.172 | 0.861 |
| +Pseudo | 8.16 | 0.243 | 0.908 |
| +Pseudo+Conf (Full) | **8.96** | **0.267** | 0.933 |

These numbers are an ablation result rather than a claim of full visual success: the report discusses that pseudo-view supervision improves selected metrics on DL3DV-2 but can still produce darker and blurrier qualitative renderings.

Comparison visualizations: `running_result/5_3/evaluation/*/comparisons/`  
Pseudo-views: `running_result/5_3/pseudo_views/*/pseudo_views/`

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{zhao2026gensparse3d,
  title={Generative Sparse-View 3D Reconstruction: Initialization, SLAM, and Confidence-Aware Enhancement},
  author={Zhao, Juncheng},
  year={2026},
  howpublished={\url{https://github.com/JunchengZhao123/AIAA3201_CV_Project}}
}
```

## Acknowledgements

This project builds upon the following open-source works:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) (Kerbl et al., 2023)
- [S3PO-GS](https://github.com/3DAgentWorld/S3PO-GS) (Cheng et al., 2025)
- [VGGT](https://github.com/facebookresearch/vggt) (Wang et al., 2025)
- [Difix3D / Difix](https://github.com/nv-tlabs/Difix3D) and [nvidia/difix](https://huggingface.co/nvidia/difix) (Wu et al., CVPR 2025 Oral)
- [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) / RIFE (Huang et al., 2022)
- [COLMAP](https://colmap.github.io/) (Schonberger & Frahm, 2016)
