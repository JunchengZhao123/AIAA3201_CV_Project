#!/usr/bin/env bash
set -euo pipefail

# ================================================================
#  Part 1: 3DGS + Scaffold-GS Training (Plan A & Plan B)
# ================================================================
#
#  Usage:
#    bash scripts/run_3dgs_training.sh                         # all (3DGS + Scaffold-GS, Plan A+B)
#    bash scripts/run_3dgs_training.sh 3dgs                    # standard 3DGS only
#    bash scripts/run_3dgs_training.sh scaffold                # Scaffold-GS only
#    bash scripts/run_3dgs_training.sh 3dgs planA              # 3DGS, Plan A only
#    bash scripts/run_3dgs_training.sh 3dgs planB              # 3DGS, Plan B only
#    bash scripts/run_3dgs_training.sh scaffold planB mipnerf  # Scaffold-GS, Plan B, Mip-NeRF only
#
#  Environment variables:
#    ITERATIONS=30000      Training iterations (default 30000)
#    FORCE_RERUN=0         Set to 1 to overwrite existing results
#
#  Prerequisites:
#    - conda env "gaussian_splatting" with 3DGS dependencies
#    - conda env "scaffold_gs" with Scaffold-GS dependencies
#    - Scaffold-GS cloned at $PROJ/Scaffold-GS (see setup below)
#
#  Scaffold-GS setup (one-time):
#    git clone https://github.com/city-super/Scaffold-GS.git --recursive
#    cd Scaffold-GS
#    conda env create --file environment.yml
#    conda activate scaffold_gs
#    pip install submodules/simple-knn
# ================================================================

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
DS="$PROJ/dataset"

# Standard 3DGS
GS_DIR="$PROJ/gaussian-splatting"
GS_ENV="gaussian_splatting"

# Scaffold-GS
SGS_DIR="$PROJ/Scaffold-GS"
SGS_ENV="scaffold_gs"

# Outputs
OUT_BASE="$PROJ/running_result/5_1/5_1_2_3DGS_Optimization"

# Plan A (COLMAP) source roots
COLMAP_BASE="$PROJ/running_result/5_1/5_1_1_Pose & Point Cloud Initialization/PlanA_COLMAP_output/output"
# Plan B (VGGT) source roots
VGGT_BASE="$PROJ/running_result/5_1/5_1_1_Pose & Point Cloud Initialization/PlanB_VGGT_output/dataset_VGGT"

ITERATIONS="${ITERATIONS:-30000}"
FORCE_RERUN="${FORCE_RERUN:-0}"

MODEL="${1:-all}"       # "all", "3dgs", "scaffold"
PLAN="${2:-all}"        # "all", "planA", "planB"
SUBSET="${3:-all}"      # "all", "mandatory", "mipnerf"

# Convergence checkpoints
if [[ "$ITERATIONS" -ge 30000 ]]; then
    TEST_ITERS="2000 5000 7000 10000 15000 20000 30000"
elif [[ "$ITERATIONS" -ge 10000 ]]; then
    TEST_ITERS="1000 3000 5000 7000 10000"
else
    TEST_ITERS="1000 $ITERATIONS"
fi

# ──────────── Data layout helpers ────────────

prepare_planA() {
    local name="$1"
    local colmap_sparse="$2"   # path containing sparse/0/*.bin
    local images_dir="$3"      # path to images/ folder
    local dst="$OUT_BASE/datasets_planA/$name"
    mkdir -p "$dst/sparse/0"

    # Symlink images
    [ ! -e "$dst/images" ] && [ -d "$images_dir" ] && ln -sfn "$images_dir" "$dst/images"

    # Symlink sparse .bin files
    for f in cameras.bin images.bin points3D.bin; do
        [ ! -e "$dst/sparse/0/$f" ] && [ -f "$colmap_sparse/sparse/0/$f" ] && \
            ln -sfn "$colmap_sparse/sparse/0/$f" "$dst/sparse/0/$f"
    done

    # Verify
    local ok=1
    [ ! -d "$dst/images" ] && echo "  WARN: $dst/images missing" && ok=0
    [ ! -f "$dst/sparse/0/cameras.bin" ] && echo "  WARN: $dst/sparse/0/cameras.bin missing" && ok=0
    [ ! -f "$dst/sparse/0/images.bin" ] && echo "  WARN: $dst/sparse/0/images.bin missing" && ok=0
    [ ! -f "$dst/sparse/0/points3D.bin" ] && echo "  WARN: $dst/sparse/0/points3D.bin missing" && ok=0
    [ "$ok" -eq 0 ] && echo "  ERROR: Plan A dataset incomplete for $name" && return 1

    echo "$dst"
}

prepare_planA_mipnerf() {
    local scene="$1"
    local src="$DS/Mip-NeRF/$scene"
    local dst="$OUT_BASE/datasets_planA/Mip-NeRF_$scene"
    mkdir -p "$dst/sparse/0"
    # Mip-NeRF 360 comes with pre-built COLMAP
    [ ! -e "$dst/images" ] && [ -d "$src/images" ] && ln -sfn "$src/images" "$dst/images"
    for f in cameras.bin images.bin points3D.bin; do
        [ ! -e "$dst/sparse/0/$f" ] && [ -f "$src/sparse/0/$f" ] && \
            ln -sfn "$src/sparse/0/$f" "$dst/sparse/0/$f"
    done
    echo "$dst"
}

# ──────────── Training runners ────────────

run_3dgs() {
    local tag="$1"       # e.g. "planA_405841"
    local src_path="$2"
    local out_dir="$OUT_BASE/experiments/${tag}_3dgs_iter${ITERATIONS}"

    if [[ -f "$out_dir/results.json" && "$FORCE_RERUN" != "1" ]]; then
        echo "[SKIP] 3DGS $tag — results exist: $out_dir/results.json"
        return
    fi

    echo ""
    echo "======== 3DGS: $tag ========"
    echo "  source: $src_path"
    echo "  output: $out_dir"
    mkdir -p "$out_dir"

    conda run -n "$GS_ENV" python "$GS_DIR/train.py" \
        -s "$src_path" \
        -m "$out_dir" \
        --eval \
        --iterations "$ITERATIONS" \
        --test_iterations $TEST_ITERS \
        --save_iterations "$ITERATIONS" \
        2>&1 | tee "$out_dir/train.log"

    conda run -n "$GS_ENV" python "$GS_DIR/render.py" -m "$out_dir" \
        2>&1 | tee "$out_dir/render.log"

    conda run -n "$GS_ENV" python "$GS_DIR/metrics.py" -m "$out_dir" \
        2>&1 | tee "$out_dir/metrics.log"

    echo "  Done: $tag"
}

run_scaffold() {
    local tag="$1"
    local src_path="$2"
    local voxel_size="${3:-0}"
    local out_dir="$OUT_BASE/experiments/${tag}_scaffold_iter${ITERATIONS}"

    if ! [ -d "$SGS_DIR" ]; then
        echo "[ERROR] Scaffold-GS not found at $SGS_DIR"
        echo "        Clone it: git clone https://github.com/city-super/Scaffold-GS.git --recursive"
        return 1
    fi

    if [[ -f "$out_dir/results.json" && "$FORCE_RERUN" != "1" ]]; then
        echo "[SKIP] Scaffold-GS $tag — results exist: $out_dir/results.json"
        return
    fi

    echo ""
    echo "======== Scaffold-GS: $tag (voxel_size=$voxel_size) ========"
    echo "  source: $src_path"
    echo "  output: $out_dir"
    mkdir -p "$out_dir"

    conda run -n "$SGS_ENV" python "$SGS_DIR/train.py" \
        -s "$src_path" \
        -m "$out_dir" \
        --eval \
        --iterations "$ITERATIONS" \
        --test_iterations $TEST_ITERS \
        --save_iterations "$ITERATIONS" \
        --voxel_size "$voxel_size" \
        --update_init_factor 16 \
        2>&1 | tee "$out_dir/train.log"

    conda run -n "$SGS_ENV" python "$SGS_DIR/render.py" -m "$out_dir" \
        2>&1 | tee "$out_dir/render.log"

    conda run -n "$SGS_ENV" python "$SGS_DIR/metrics.py" -m "$out_dir" \
        2>&1 | tee "$out_dir/metrics.log"

    echo "  Done: $tag"
}

# ──────────── Scene definitions ────────────

# Mandatory scenes: [name, COLMAP_sparse_root, images_dir]
SCENES_MANDATORY=(405841 DL3DV-2 Re10k-1)
COLMAP_DIRS_MANDATORY=(
    "$COLMAP_BASE/405841"
    "$COLMAP_BASE/DL3DV-2"
    "$COLMAP_BASE/Re10k-1"
)
IMAGES_DIRS_MANDATORY=(
    "$DS/405841/FRONT/images"
    "$DS/DL3DV-2/images"
    "$DS/Re10k-1/images"
)
VGGT_DIRS_MANDATORY=(
    "$VGGT_BASE/405841/FRONT"
    "$VGGT_BASE/DL3DV-2"
    "$VGGT_BASE/Re10k-1"
)
# Voxel sizes for Scaffold-GS per mandatory scene
VOXEL_MANDATORY=(0 0 0.005)

# Mip-NeRF scenes
SCENES_MIPNERF=(bicycle bonsai counter garden kitchen room stump)
VOXEL_MIPNERF=0.005

# ──────────── Execute ────────────

run_for_scenes() {
    local model_type="$1"  # "3dgs" or "scaffold"

    # Mandatory scenes
    if [[ "$SUBSET" == "all" || "$SUBSET" == "mandatory" ]]; then
        for i in "${!SCENES_MANDATORY[@]}"; do
            local scene="${SCENES_MANDATORY[$i]}"
            local voxel="${VOXEL_MANDATORY[$i]}"

            if [[ "$PLAN" == "all" || "$PLAN" == "planA" ]]; then
                local pa_dir
                pa_dir="$(prepare_planA "$scene" "${COLMAP_DIRS_MANDATORY[$i]}" "${IMAGES_DIRS_MANDATORY[$i]}")"
                if [ "$model_type" == "3dgs" ]; then
                    run_3dgs "planA_${scene}" "$pa_dir"
                else
                    run_scaffold "planA_${scene}" "$pa_dir" "$voxel"
                fi
            fi

            if [[ "$PLAN" == "all" || "$PLAN" == "planB" ]]; then
                local pb_dir="${VGGT_DIRS_MANDATORY[$i]}"
                if [ "$model_type" == "3dgs" ]; then
                    run_3dgs "planB_${scene}" "$pb_dir"
                else
                    run_scaffold "planB_${scene}" "$pb_dir" "$voxel"
                fi
            fi
        done
    fi

    # Mip-NeRF 360 scenes
    if [[ "$SUBSET" == "all" || "$SUBSET" == "mipnerf" ]]; then
        for scene in "${SCENES_MIPNERF[@]}"; do
            if [[ "$PLAN" == "all" || "$PLAN" == "planA" ]]; then
                if [ -d "$DS/Mip-NeRF/$scene/sparse/0" ]; then
                    local pa_dir
                    pa_dir="$(prepare_planA_mipnerf "$scene")"
                    if [ "$model_type" == "3dgs" ]; then
                        run_3dgs "planA_MipNeRF-${scene}" "$pa_dir"
                    else
                        run_scaffold "planA_MipNeRF-${scene}" "$pa_dir" "$VOXEL_MIPNERF"
                    fi
                else
                    echo "[SKIP] Mip-NeRF/$scene — no COLMAP sparse/0 found"
                fi
            fi

            if [[ "$PLAN" == "all" || "$PLAN" == "planB" ]]; then
                local pb_dir="$VGGT_BASE/Mip-NeRF/$scene"
                if [ -d "$pb_dir/sparse/0" ]; then
                    if [ "$model_type" == "3dgs" ]; then
                        run_3dgs "planB_MipNeRF-${scene}" "$pb_dir"
                    else
                        run_scaffold "planB_MipNeRF-${scene}" "$pb_dir" "$VOXEL_MIPNERF"
                    fi
                else
                    echo "[SKIP] VGGT Mip-NeRF/$scene — no sparse/0 found at $pb_dir"
                fi
            fi
        done
    fi
}

# ──────────── Main dispatch ────────────

echo "============================================================"
echo "  Part 1: 3DGS Training"
echo "  Model:  $MODEL | Plan: $PLAN | Subset: $SUBSET"
echo "  Iterations: $ITERATIONS"
echo "============================================================"

if [[ "$MODEL" == "all" || "$MODEL" == "3dgs" ]]; then
    echo ""
    echo ">>> Standard 3DGS <<<"
    run_for_scenes "3dgs"
fi

if [[ "$MODEL" == "all" || "$MODEL" == "scaffold" ]]; then
    echo ""
    echo ">>> Scaffold-GS <<<"
    if ! [ -d "$SGS_DIR" ]; then
        echo ""
        echo "ERROR: Scaffold-GS not found at $SGS_DIR"
        echo "Set it up first:"
        echo "  git clone https://github.com/city-super/Scaffold-GS.git --recursive"
        echo "  cd Scaffold-GS && conda env create --file environment.yml"
        echo "  conda activate scaffold_gs && pip install submodules/simple-knn"
        exit 1
    fi
    run_for_scenes "scaffold"
fi

echo ""
echo "============================================================"
echo "  All requested training runs complete."
echo "  Results: $OUT_BASE/experiments/"
echo "============================================================"
