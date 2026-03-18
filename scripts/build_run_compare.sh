#!/bin/bash
# Build loger_infer, run inference, then launch side-by-side visualizer.
#
# Usage (from loger_cpp/):
#   bash scripts/build_run_compare.sh [OPTIONS]
#
# Options:
#   --input PATH        Input image folder  (default: ../data/examples/office)
#   --model PATH        Checkpoint .pt      (default: ../ckpts/LoGeR/latest.pt)
#   --config PATH       Config .yaml        (default: ../ckpts/LoGeR/original_config.yaml)
#   --ref PATH          Reference .pt       (default: reference/examples_office_0_50_1.pt)
#   --output_ply PATH   Output PLY          (default: output.ply)
#   --output_traj PATH  Output trajectory   (default: trajectory.txt)
#   --window_size N     (default: 32)
#   --overlap_size N    (default: 3)
#   --no_ttt            Disable TTT
#   --no_swa            Disable SWA
#   --skip_build        Skip cmake build step
#   --skip_infer        Skip inference step (just visualize existing outputs)
#   --cpp_port N        Viser port for C++ output  (default: 8080)
#   --ref_port N        Viser port for reference   (default: 8081)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON=/home/giulianowf/Documents/models/venv_gguf/bin/python

# --- Defaults ---
INPUT="$ROOT/../data/examples/office"
MODEL="$ROOT/../ckpts/LoGeR/latest.pt"
CONFIG="$ROOT/../ckpts/LoGeR/original_config.yaml"
REF="$ROOT/reference/examples_office_0_50_1.pt"
OUTPUT_PLY="$ROOT/output.ply"
OUTPUT_TRAJ="$ROOT/trajectory.txt"
OUTPUT_PT="$ROOT/output.pt"
WINDOW_SIZE=32
OVERLAP_SIZE=3
EXTRA_FLAGS=""
SKIP_BUILD=0
SKIP_INFER=0
CPP_PORT=8080
REF_PORT=8081

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)        INPUT="$2";        shift 2 ;;
        --model)        MODEL="$2";        shift 2 ;;
        --config)       CONFIG="$2";       shift 2 ;;
        --ref)          REF="$2";          shift 2 ;;
        --output_ply)   OUTPUT_PLY="$2";   shift 2 ;;
        --output_traj)  OUTPUT_TRAJ="$2";  shift 2 ;;
        --window_size)  WINDOW_SIZE="$2";  shift 2 ;;
        --overlap_size) OVERLAP_SIZE="$2"; shift 2 ;;
        --no_ttt)       EXTRA_FLAGS="$EXTRA_FLAGS --no_ttt"; shift ;;
        --no_swa)       EXTRA_FLAGS="$EXTRA_FLAGS --no_swa"; shift ;;
        --skip_build)   SKIP_BUILD=1;      shift ;;
        --skip_infer)   SKIP_INFER=1;      shift ;;
        --cpp_port)     CPP_PORT="$2";     shift 2 ;;
        --ref_port)     REF_PORT="$2";     shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Build
# ---------------------------------------------------------------------------
if [[ $SKIP_BUILD -eq 0 ]]; then
    echo "=== Building loger_infer ==="
    cmake -S "$ROOT" -B "$ROOT/build" \
        -DCMAKE_PREFIX_PATH="$ROOT/third_party/libtorch" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        --log-level=ERROR
    cmake --build "$ROOT/build" --parallel "$(nproc)" --target loger_infer
    echo "Build OK"
else
    echo "=== Skipping build ==="
fi

# ---------------------------------------------------------------------------
# 2. Inference
# ---------------------------------------------------------------------------
if [[ $SKIP_INFER -eq 0 ]]; then
    echo ""
    echo "=== Running inference ==="
    echo "  input  : $INPUT"
    echo "  model  : $MODEL"
    echo "  config : $CONFIG"
    echo "  output : $OUTPUT_PLY / $OUTPUT_TRAJ"
    echo ""

    "$ROOT/build/loger_infer" \
        --input        "$INPUT" \
        --model_name   "$MODEL" \
        --config       "$CONFIG" \
        --output_ply   "$OUTPUT_PLY" \
        --output_traj  "$OUTPUT_TRAJ" \
        --output_pt    "$OUTPUT_PT" \
        --window_size  "$WINDOW_SIZE" \
        --overlap_size "$OVERLAP_SIZE" \
        $EXTRA_FLAGS

    echo "Inference OK"
else
    echo "=== Skipping inference ==="
fi

# ---------------------------------------------------------------------------
# 3. Numerical comparison
# ---------------------------------------------------------------------------
echo ""
echo "=== Numerical comparison ==="
"$PYTHON" "$SCRIPT_DIR/compare_output.py" \
    --ref  "$REF" \
    --ply  "$OUTPUT_PLY" \
    --traj "$OUTPUT_TRAJ"

# ---------------------------------------------------------------------------
# 4. Side-by-side visualizer
# ---------------------------------------------------------------------------
echo ""
echo "=== Launching visualizers ==="
echo "  C++ output  → http://localhost:$CPP_PORT"
echo "  Python ref  → http://localhost:$REF_PORT"
echo ""

"$PYTHON" "$SCRIPT_DIR/compare_vis.py" \
    --ref      "$REF" \
    --pt       "$OUTPUT_PT" \
    --cpp_port "$CPP_PORT" \
    --ref_port "$REF_PORT"
