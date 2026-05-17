#!/usr/bin/env bash
set -euo pipefail

# ================================================================
#  Part 3: Setup Script - RIFE + Difix3D + Dependencies
# ================================================================
#
#  Installs the Practical-RIFE frame interpolation model, the
#  Difix3D diffusion model, and required dependencies for
#  pseudo-view generation and confidence-weighted hybrid training.
#
#  Reuses the S3PO-GS conda environment from Part 2 and adds
#  diffusion model dependencies on top.
#
#  Usage:
#    bash scripts/scripts_p3/setup_p3.sh
#
#  Prerequisites:
#    - S3PO-GS environment from Part 2 already set up
#      (conda activate S3PO-GS)
#
# ================================================================

SCRIPTS_P3="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "${CV_PROJ_ROOT:-"$SCRIPTS_P3/../.."}" && pwd)"
S3PO_ROOT="${S3PO_ROOT:-$PROJ/S3PO-GS}"
DIFIX3D_ROOT="${DIFIX3D_ROOT:-$PROJ/Difix3D}"
RIFE_ROOT="${RIFE_ROOT:-$PROJ/Practical-RIFE}"

echo "================================================================"
echo "  Part 3: Environment Setup"
echo "================================================================"
echo "  Project root:  $PROJ"
echo "  S3PO-GS:      $S3PO_ROOT"
echo "  Difix3D:      $DIFIX3D_ROOT"
echo "  RIFE:         $RIFE_ROOT"
echo ""

# ============================================================
# Step 0: Clone and set up Practical-RIFE
# ============================================================
echo "[0/4] Setting up Practical-RIFE (video frame interpolation)..."
if [ ! -d "$RIFE_ROOT" ]; then
    echo "  Cloning Practical-RIFE..."
    git clone https://github.com/hzwer/Practical-RIFE.git "$RIFE_ROOT"
else
    echo "  Practical-RIFE already present at $RIFE_ROOT"
fi

if [ ! -d "$RIFE_ROOT/train_log" ]; then
    echo "  Downloading RIFE v4.26 model weights..."
    mkdir -p "$RIFE_ROOT/train_log"
    # Model weights hosted on GitHub releases
    RIFE_MODEL_URL="https://github.com/hzwer/Practical-RIFE/raw/main/train_log"
    for f in flownet.pkl contextnet.pkl unet.pkl; do
        if [ ! -f "$RIFE_ROOT/train_log/$f" ]; then
            wget -q -O "$RIFE_ROOT/train_log/$f" "$RIFE_MODEL_URL/$f" 2>/dev/null || \
            curl -sL -o "$RIFE_ROOT/train_log/$f" "$RIFE_MODEL_URL/$f" 2>/dev/null || \
            echo "    WARNING: Could not download $f. Download manually from"
            echo "      https://github.com/hzwer/Practical-RIFE"
        fi
    done
    echo "  RIFE model weights downloaded."
else
    echo "  RIFE model weights already present."
fi

# ============================================================
# Step 1: Clone Difix3D (if available)
# ============================================================
echo "[1/4] Setting up Difix3D..."
if [ ! -d "$DIFIX3D_ROOT" ]; then
    echo "  Cloning Difix3D repository..."
    git clone https://github.com/JZhangjie/Difix3D.git "$DIFIX3D_ROOT" 2>/dev/null || {
        echo "  NOTE: Official Difix3D repo may not be public yet."
        echo "  Creating placeholder structure for Difix3D..."
        mkdir -p "$DIFIX3D_ROOT"
        echo "  You may need to manually obtain Difix3D code from:"
        echo "    https://research.nvidia.com/labs/toronto-ai/difix3d"
        echo "    or the pipeline will fall back to SD-Turbo img2img."
    }
else
    echo "  Difix3D already present at $DIFIX3D_ROOT"
fi

# ============================================================
# Step 2: Activate S3PO-GS environment and add dependencies
# ============================================================
echo "[2/4] Installing Part 3 dependencies into S3PO-GS environment..."

eval "$(conda shell.bash hook)"

if ! conda env list 2>/dev/null | grep -q "S3PO-GS"; then
    echo "  ERROR: S3PO-GS environment not found."
    echo "  Run Part 2 setup first: bash scripts/scripts_p2/setup_s3po.sh"
    exit 1
fi

conda activate S3PO-GS

# Pin numpy<2 to avoid ABI mismatch with PyTorch 2.1 compiled against numpy 1.x
pip install "numpy<2"

pip install \
    "diffusers>=0.25.0,<0.31.0" \
    "transformers>=4.36.0,<4.46.0" \
    "accelerate>=0.25.0,<0.34.0" \
    safetensors \
    peft \
    matplotlib

echo "  Dependencies installed into S3PO-GS environment."

# ============================================================
# Step 3: Download SD-Turbo checkpoint
# ============================================================
echo "[3/4] Downloading model checkpoints..."

CKPT_DIR="$PROJ/checkpoints/difix3d"
mkdir -p "$CKPT_DIR"

if [ ! -f "$CKPT_DIR/sd_turbo_downloaded" ]; then
    echo "  Downloading SD-Turbo base model (for Difix3D)..."
    python -c "
from diffusers import AutoPipelineForImage2Image
pipe = AutoPipelineForImage2Image.from_pretrained(
    'stabilityai/sd-turbo',
    torch_dtype='auto',
    variant='fp16',
)
pipe.save_pretrained('$CKPT_DIR/sd-turbo')
print('SD-Turbo downloaded successfully.')
" && touch "$CKPT_DIR/sd_turbo_downloaded" || {
        echo "  WARNING: Could not auto-download SD-Turbo."
        echo "  Please manually download from: https://huggingface.co/stabilityai/sd-turbo"
        echo "  Or use --skip_difix flag to skip Difix3D enhancement."
    }
else
    echo "  SD-Turbo checkpoint already downloaded."
fi

# ============================================================
# Step 4: Verify S3PO-GS and Part 2 results
# ============================================================
echo "[4/4] Verifying prerequisites..."

if [ -f "$S3PO_ROOT/slam.py" ]; then
    echo "  S3PO-GS found at $S3PO_ROOT"
else
    echo "  WARNING: S3PO-GS not found. Run Part 2 setup first."
fi

RESULTS_DIR="$PROJ/running_result/5_2/results"
if [ -d "$RESULTS_DIR" ]; then
    n_results=$(find "$RESULTS_DIR" -name "point_cloud.ply" 2>/dev/null | wc -l)
    echo "  Found $n_results trained models in Part 2 results."
else
    echo "  WARNING: Part 2 results not found at $RESULTS_DIR"
    echo "  Run Part 2 first: bash scripts/scripts_p2/run_full_pipeline.sh"
fi

echo ""
echo "================================================================"
echo "  Part 3 Setup Complete!"
echo "================================================================"
echo ""
echo "  Environment:  conda activate S3PO-GS"
echo "  Difix3D:     $DIFIX3D_ROOT"
echo "  Checkpoints: $CKPT_DIR"
echo ""
echo "  Next steps:"
echo "    conda activate S3PO-GS"
echo "    bash scripts/scripts_p3/run_full_pipeline_p3.sh"
echo "================================================================"
