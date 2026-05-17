#!/bin/bash
# ================================================================
#  Part 1 Plan B: VGGT & Pi3 -> COLMAP (chunked, ALL frames)
#  OOM-safe for A6000 48GB via overlapping-chunk processing.
#
#  Each model processes 100 frames per chunk with 20-frame overlap.
#  Coordinate systems are aligned across chunks via Procrustes,
#  so every frame in the dataset gets a pose estimate.
# ================================================================
#
#  Usage (arg2 defaults to "all" if omitted = mandatory + Mip-NeRF):
#    cd /path/to/Third_Spring_CVproj
#    bash scripts/run_planB_all.sh                    # VGGT + Pi3, all datasets
#    bash scripts/run_planB_all.sh vggt               # VGGT only, all datasets
#    bash scripts/run_planB_all.sh vggt mipnerf       # VGGT only, Mip-NeRF scenes
#    bash scripts/run_planB_all.sh pi3                # Pi3 only, all datasets
#    bash scripts/run_planB_all.sh pi3 mipnerf       # Pi3 only, Mip-NeRF scenes
#
#  Environment setup (one-time):
#
#    conda create -n vggt python=3.10 -y && conda activate vggt
#    pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
#    cd vggt && pip install -e . && cd ..
#    pip install trimesh pycolmap==3.10.0 pyceres==2.3 tqdm
#
#    conda create -n pi3 python=3.10 -y && conda activate pi3
#    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
#    pip install -r Pi3/requirements.txt
#    pip install scipy tqdm   # scipy for mask resizing, tqdm for progress bars
#
# ================================================================

set -e

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
DS="$PROJ/dataset"
VGGT_OUT="$PROJ/running_result/5_1/5_1_1_Pose & Point Cloud Initialization/PlanB_VGGT_output/dataset_VGGT"
PI3_OUT="$PROJ/running_result/5_1/5_1_1_Pose & Point Cloud Initialization/PlanB_Pi3_output/dataset_Pi3"

CHUNK=100
OVLP=20
MODE="${1:-all}"       # "all", "vggt", or "pi3"
SUBSET="${2:-all}"     # "all" (default) or "mipnerf" / "mip"

# Local model weights (set these to skip download on offline servers)
# Search order: env var > project checkpoints dir > torch hub cache
VGGT_CKPT="${VGGT_CKPT:-}"
if [ -z "$VGGT_CKPT" ]; then
    for _p in "$PROJ/checkpoints/vggt_model.pt" \
              "$HOME/.cache/torch/hub/checkpoints/vggt/model.pt" \
              "$HOME/.cache/torch/hub/checkpoints/model.pt"; do
        [ -f "$_p" ] && VGGT_CKPT="$_p" && break
    done
fi
PI3_CKPT="${PI3_CKPT:-$PROJ/checkpoints/pi3_model.safetensors}"

link_images() {
    local src="$1" dst="$2"
    mkdir -p "$dst"
    [ -e "$dst/images" ] || ln -s "$src" "$dst/images"
}

# ──────────── VGGT ────────────

run_vggt() {
    echo -e "\n======== VGGT: $1 ========"
    local src_images="$2" out="$3"
    link_images "$src_images" "$out"
    local ckpt_flag=""
    [ -f "$VGGT_CKPT" ] && ckpt_flag="--model_path $VGGT_CKPT"
    conda run -n vggt python "$PROJ/scripts/run_vggt_colmap.py" \
        --scene_dir "$out" --output_dir "$out" \
        --chunk_size $CHUNK --overlap $OVLP \
        --conf_threshold 5.0 --max_points 100000 \
        $ckpt_flag
}

if [ "$MODE" = "all" ] || [ "$MODE" = "vggt" ]; then
    if [ "$SUBSET" = "all" ]; then
        run_vggt "405841"  "$DS/405841/FRONT/images" "$VGGT_OUT/405841/FRONT"
        run_vggt "DL3DV-2" "$DS/DL3DV-2/images"      "$VGGT_OUT/DL3DV-2"
        run_vggt "Re10k-1" "$DS/Re10k-1/images"       "$VGGT_OUT/Re10k-1"
    fi
    if [ "$SUBSET" = "all" ] || [ "$SUBSET" = "mipnerf" ] || [ "$SUBSET" = "mip" ]; then
        for S in bicycle bonsai counter garden kitchen room stump; do
            [ -d "$DS/Mip-NeRF/$S/images" ] && \
                run_vggt "Mip-NeRF/$S" "$DS/Mip-NeRF/$S/images" "$VGGT_OUT/Mip-NeRF/$S"
        done
    fi
    echo -e "\n======== VGGT complete ========"
fi

# ──────────── Pi3 ────────────

run_pi3() {
    echo -e "\n======== Pi3: $1 ========"
    local src_images="$2" out="$3"
    link_images "$src_images" "$out"
    local ckpt_flag=""
    [ -f "$PI3_CKPT" ] && ckpt_flag="--ckpt $PI3_CKPT"
    conda run -n pi3 python "$PROJ/scripts/run_pi3_colmap.py" \
        --scene_dir "$out" --output_dir "$out" \
        --chunk_size $CHUNK --overlap $OVLP \
        --conf_threshold 0.1 --max_points 100000 \
        $ckpt_flag
}

if [ "$MODE" = "all" ] || [ "$MODE" = "pi3" ]; then
    if [ "$SUBSET" = "all" ]; then
        run_pi3 "405841"  "$DS/405841/FRONT/images" "$PI3_OUT/405841/FRONT"
        run_pi3 "DL3DV-2" "$DS/DL3DV-2/images"      "$PI3_OUT/DL3DV-2"
        run_pi3 "Re10k-1" "$DS/Re10k-1/images"       "$PI3_OUT/Re10k-1"
    fi
    if [ "$SUBSET" = "all" ] || [ "$SUBSET" = "mipnerf" ] || [ "$SUBSET" = "mip" ]; then
        for S in bicycle bonsai counter garden kitchen room stump; do
            [ -d "$DS/Mip-NeRF/$S/images" ] && \
                run_pi3 "Mip-NeRF/$S" "$DS/Mip-NeRF/$S/images" "$PI3_OUT/Mip-NeRF/$S"
        done
    fi
    echo -e "\n======== Pi3 complete ========"
fi

echo -e "\nAll Plan B runs complete."
echo "VGGT: $VGGT_OUT"
echo "Pi3:  $PI3_OUT"
echo ""
echo "Verify:  python scripts/verify_colmap_outputs.py --all"
