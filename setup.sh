#!/bin/bash
# setup.sh — Download libtorch and LoGeR checkpoints, then configure the build.
# Run from the loger_cpp/ directory.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[setup]${NC} $*"; }
error()   { echo -e "${RED}[setup]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Detect PyTorch version from the active Python environment
# ---------------------------------------------------------------------------
info "Detecting PyTorch version..."

TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")

if [ -z "$TORCH_VERSION" ]; then
    warn "Could not detect PyTorch version from 'python3'. Falling back to default."
    TORCH_VERSION="2.7.0"
fi

info "Detected PyTorch version: $TORCH_VERSION"

# ---------------------------------------------------------------------------
# 2. Determine libtorch download URL
#    RTX 5090 (sm_120/Blackwell) requires CUDA 12.8.
#    PyTorch >= 2.6 has experimental sm_120 support in cu128 builds.
# ---------------------------------------------------------------------------
CUDA_TAG="cu128"
LIBTORCH_DIR="$SCRIPT_DIR/third_party/libtorch"

# Encode '+' as '%2B' for URL
TORCH_VERSION_URL=$(echo "$TORCH_VERSION" | sed 's/\+/%2B/g')
LIBTORCH_ZIP="libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION_URL}%2B${CUDA_TAG}.zip"
LIBTORCH_URL="https://download.pytorch.org/libtorch/${CUDA_TAG}/${LIBTORCH_ZIP}"

# ---------------------------------------------------------------------------
# 3. Download libtorch (skip if already present)
# ---------------------------------------------------------------------------
if [ -d "$LIBTORCH_DIR/lib" ]; then
    info "libtorch already found at $LIBTORCH_DIR — skipping download."
else
    info "Downloading libtorch $TORCH_VERSION + $CUDA_TAG..."
    info "URL: $LIBTORCH_URL"
    mkdir -p "$SCRIPT_DIR/third_party"

    # Try the detected version first; fall back to known Blackwell-compatible version
    if ! wget --spider "$LIBTORCH_URL" 2>/dev/null; then
        warn "URL not found for version $TORCH_VERSION. Trying 2.7.0..."
        TORCH_VERSION="2.7.0"
        LIBTORCH_ZIP="libtorch-cxx11-abi-shared-with-deps-2.7.0%2B${CUDA_TAG}.zip"
        LIBTORCH_URL="https://download.pytorch.org/libtorch/${CUDA_TAG}/${LIBTORCH_ZIP}"
        if ! wget --spider "$LIBTORCH_URL" 2>/dev/null; then
            warn "------------------------------------------------------"
            warn "Could not find a cu128 libtorch build automatically."
            warn "Please download it manually from:"
            warn "  https://pytorch.org/get-started/locally/"
            warn "Select: LibTorch | C++/Java | CUDA 12.8 | cxx11 ABI"
            warn "Then unzip to: $LIBTORCH_DIR"
            warn "------------------------------------------------------"
            error "Automatic download failed. See instructions above."
        fi
    fi

    TMP_ZIP="$SCRIPT_DIR/third_party/libtorch.zip"
    wget -O "$TMP_ZIP" "$LIBTORCH_URL" || error "Download failed."

    info "Extracting libtorch..."
    cd "$SCRIPT_DIR/third_party"
    unzip -q "$TMP_ZIP"
    rm "$TMP_ZIP"

    # The zip extracts to a directory called 'libtorch'
    if [ ! -d "$LIBTORCH_DIR/lib" ]; then
        error "Extraction failed — expected $LIBTORCH_DIR/lib to exist."
    fi
    info "libtorch extracted to $LIBTORCH_DIR"
fi

# ---------------------------------------------------------------------------
# 4. Download LoGeR checkpoints (skip if already present)
# ---------------------------------------------------------------------------
HF_BASE="https://huggingface.co/Junyi42/LoGeR/resolve/main"

download_ckpt() {
    local name="$1"      # e.g. "LoGeR" or "LoGeR_star"
    local dest_dir="$REPO_ROOT/ckpts/$name"
    mkdir -p "$dest_dir"

    # Checkpoint weights
    if [ -f "$dest_dir/latest.pt" ]; then
        local size
        size=$(stat -c%s "$dest_dir/latest.pt" 2>/dev/null || echo 0)
        if [ "$size" -gt 1000000 ]; then
            info "Checkpoint $name/latest.pt already exists ($(( size / 1024 / 1024 )) MB) — skipping."
        else
            warn "Checkpoint $name/latest.pt seems incomplete ($size bytes). Re-downloading..."
            rm "$dest_dir/latest.pt"
        fi
    fi

    if [ ! -f "$dest_dir/latest.pt" ]; then
        info "Downloading $name/latest.pt (~4.7 GB)..."
        wget --show-progress \
             -O "$dest_dir/latest.pt" \
             "${HF_BASE}/${name}/latest.pt?download=true" \
            || error "Failed to download $name checkpoint."
        info "$name checkpoint downloaded."
    fi

    # Config YAML
    if [ ! -f "$dest_dir/original_config.yaml" ]; then
        info "Downloading $name/original_config.yaml..."
        wget -q -O "$dest_dir/original_config.yaml" \
             "${HF_BASE}/${name}/original_config.yaml?download=true" \
            || warn "Could not download $name config (may not exist on HF yet)."
    fi
}

info "Checking LoGeR checkpoints..."
download_ckpt "LoGeR"
download_ckpt "LoGeR_star"

# ---------------------------------------------------------------------------
# 5a. Convert checkpoints to old-style pickle format (required by libtorch
#     torch::pickle_load which cannot read the new zip-format .pt files).
# ---------------------------------------------------------------------------
convert_ckpt() {
    local ckpt="$REPO_ROOT/ckpts/$1/latest.pt"
    if [ -f "$ckpt" ]; then
        info "Converting $1/latest.pt to old-format pickle..."
        python3 "$SCRIPT_DIR/scripts/convert_checkpoint.py" "$ckpt" \
            || warn "Checkpoint conversion failed for $1 — inference may crash."
    fi
}
convert_ckpt "LoGeR"
convert_ckpt "LoGeR_star"

# ---------------------------------------------------------------------------
# 5. Configure CMake build
# ---------------------------------------------------------------------------
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"

info "Configuring CMake..."
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    || error "CMake configuration failed."

# Symlink compile_commands.json for IDE/clangd support
ln -sf "$BUILD_DIR/compile_commands.json" "$SCRIPT_DIR/compile_commands.json"
info "Symlinked compile_commands.json for IDE support."

# ---------------------------------------------------------------------------
# 6. Build
# ---------------------------------------------------------------------------
info "Building (this may take a few minutes)..."
cmake --build "$BUILD_DIR" --parallel "$(nproc)" \
    || error "Build failed."

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Run inference:"
echo "  $BUILD_DIR/loger_infer \\"
echo "    --input $REPO_ROOT/data/examples/office \\"
echo "    --model_name $REPO_ROOT/ckpts/LoGeR/latest.pt \\"
echo "    --config $REPO_ROOT/ckpts/LoGeR/original_config.yaml \\"
echo "    --output_ply output.ply \\"
echo "    --output_traj trajectory.txt"
echo ""
echo "Run unit tests:"
echo "  cd $BUILD_DIR && ctest --output-on-failure"
