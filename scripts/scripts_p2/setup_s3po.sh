#!/usr/bin/env bash
set -euo pipefail

# ================================================================
#  S3PO-GS Setup Script
# ================================================================
#
#  Automates the installation of S3PO-GS and its dependencies.
#
#  Usage:
#    bash setup_s3po.sh [S3PO_ROOT]
#
#  Layout (defaults):
#    Project root PROJ is the parent of scripts/ (script is scripts/scripts_p2/).
#    Raw dense data lives at  $PROJ/dataset/  (Waymo 405841, DL3DV-2, Re10k-1).
#
#  Environment:
#    CV_PROJ_ROOT=...           Override PROJ when paths differ on your machine
#
#  Prerequisites:
#    - CUDA 11.8+ with nvcc
#    - conda (miniconda/anaconda)
#    - git
#
# ================================================================

SCRIPTS_P2="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "${CV_PROJ_ROOT:-"$SCRIPTS_P2/../.."}" && pwd)"

# Optional: bash setup_s3po.sh [/path/to/S3PO-GS]
S3PO_ROOT="${1:-$PROJ/S3PO-GS}"

echo "================================================================"
echo "  S3PO-GS Installation"
echo "================================================================"
echo "  Install path: $S3PO_ROOT"
echo ""

# Step 1: Clone repository
if [ ! -d "$S3PO_ROOT" ]; then
    echo "[1/6] Cloning S3PO-GS repository..."
    git clone https://github.com/3DAgentWorld/S3PO-GS.git --recursive "$S3PO_ROOT"
else
    echo "[1/6] S3PO-GS already cloned at $S3PO_ROOT"
fi

cd "$S3PO_ROOT"

# Step 2: Create conda environment
echo "[2/6] Creating conda environment..."
if conda env list | grep -q "S3PO-GS"; then
    echo "  Environment 'S3PO-GS' already exists. Skipping creation."
else
    conda env create -f environment.yml
fi

# Step 3: Install submodules
echo "[3/6] Installing submodules (simple-knn, diff-gaussian-rasterization)..."
eval "$(conda shell.bash hook)"
conda activate S3PO-GS

# Ensure CUDA 11.8 headers are present (cuda_runtime.h): older / partial envs may only have cuda-nvcc
_need_headers=1
for _inc in "${CONDA_PREFIX}/include/cuda_runtime.h" "${CONDA_PREFIX}/targets/x86_64-linux/include/cuda_runtime.h"; do
    if [ -f "$_inc" ]; then _need_headers=0; break; fi
done
if [ "$_need_headers" -eq 1 ]; then
    echo "  Installing CUDA 11.8 dev packages (cuda-cudart headers) — needed for simple-knn / diff-gaussian builds..."
    conda install -y -c nvidia "cuda-cudart-dev=11.8*" "cuda-cccl=11.8*" || {
        echo "  ERROR: conda could not install cuda-cudart-dev. Add to environment.yml and recreate the env, or use a system CUDA 11.x toolkit."
        exit 1
    }
fi

# torch.utils.cpp_extension imports pkg_resources (setuptools); setuptools>=82 removed it
pip install -U 'setuptools>=65,<82' wheel

# PyTorch was built with CUDA 11.8 (pytorch-cuda=11.8); extension builds must use matching nvcc.
# If the host has /usr/local/cuda = 12.x, torch.utils.cpp_extension will error unless we prefer conda's toolkit.
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/nvcc" ]; then
    export CUDA_HOME="${CONDA_PREFIX}"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    # Conda CUDA layout often puts headers under targets/.../include
    for _inc in "${CONDA_PREFIX}/include" "${CONDA_PREFIX}/targets/x86_64-linux/include"; do
        if [ -d "$_inc" ]; then
            export CPATH="${_inc}${CPATH:+:${CPATH}}"
        fi
    done
    echo "  CUDA_HOME=$CUDA_HOME (conda nvcc: $(nvcc --version 2>/dev/null | head -1 || true))"
elif [ -n "${CUDA_HOME:-}" ]; then
    echo "  CUDA_HOME=$CUDA_HOME (existing)"
else
    _tc=$(python -c "import torch; print(torch.version.cuda or '?')")
    echo "  WARNING: No nvcc in \$CONDA_PREFIX/bin. Install cuda-nvcc (see environment.yml),"
    echo "           or set CUDA_HOME to a CUDA ${_tc} toolkit before building."
fi

pip install ninja  # speeds up ninja-based extension builds when used

pip install submodules/simple-knn --no-build-isolation
pip install submodules/diff-gaussian-rasterization --no-build-isolation

# Step 4: Build CRoCo curope extension
echo "[4/6] Building CRoCo curope extension..."
cd croco/models/curope/
python setup.py build_ext --inplace
cd "$S3PO_ROOT"

# Step 5: Install additional dependencies
echo "[5/6] Installing additional dependencies..."
pip install evo scipy scikit-image torchmetrics lpips

# Step 6: Patch S3PO-GS with Re10k support and sparse configs
echo "[6/6] Patching S3PO-GS with sparse configs and Re10k parser..."
python "$SCRIPTS_P2/add_re10k_parser.py" --s3po_root "$S3PO_ROOT"

# Create datasets directory
mkdir -p "$S3PO_ROOT/datasets"

echo ""
echo "================================================================"
echo "  S3PO-GS Setup Complete!"
echo "================================================================"
echo ""
echo "  Activation: conda activate S3PO-GS"
echo ""
echo "  Next steps:"
echo "    1. python \"$SCRIPTS_P2/subsample_sparse.py\" --proj_root \"$PROJ\""
echo "       (Reads dense sequences from \"$PROJ/dataset/\"; writes \"$PROJ/dataset_sparse_p2/\" unless you pass --output_root.)"
echo "    2. Run: bash \"$SCRIPTS_P2/run_s3po_all.sh\"  (or set DATASET_SPARSE_P2 if sparse output is elsewhere)"
echo ""
echo "  Or run the full pipeline:"
echo "    bash \"$SCRIPTS_P2/run_full_pipeline.sh\""
echo "================================================================"
