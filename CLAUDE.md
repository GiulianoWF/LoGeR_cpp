# LoGeR C++ (loger_cpp)

> **Note:** This file must always be kept up to date. When making changes to the codebase — new features, bug fixes, architectural changes, or accuracy findings — update the relevant sections here so future sessions have accurate context.

Full C++ libtorch reimplementation of the LoGeR 3D reconstruction model (Pi3). Produces a standalone binary that takes image sequences and outputs PLY point clouds + camera trajectories.

## Setup & Build

The easiest path is `setup.sh` which downloads libtorch, checkpoints, and builds:

```bash
sudo apt install libopencv-dev   # one-time dependency
./setup.sh
```

Or manually:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=third_party/libtorch \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
# Symlink compile_commands.json for clangd IDE support
ln -sf build/compile_commands.json compile_commands.json
```

**Dependencies:** `libopencv-dev` (apt), libtorch cu128 (bundled under `third_party/libtorch/`).
**libtorch:** Downloaded automatically by `setup.sh` to `third_party/libtorch/`. Must be cu128 build for RTX 5090 (sm_120/Blackwell) support.

### Known libtorch C++ API quirks (already fixed in source)

- `tensor.dtype()` returns `caffe2::TypeMeta` — use `tensor.scalar_type()` when passing to functions expecting `torch::Dtype`.
- `c10::Device` has no `operator<` — compare via `.type()` and `.index()` separately.
- `torch::nn::RMSNorm` is not in the libtorch C++ API — a custom `loger::RMSNorm` is defined in `include/loger/model/ttt.hpp`.
- **Checkpoints must be pre-converted**: `torch::pickle_load` cannot deserialize `collections.OrderedDict` (uses GLOBAL/REDUCE opcodes). Run `python scripts/convert_checkpoint.py ckpts/LoGeR/latest.pt` once to re-save as a plain dict. `setup.sh` should do this automatically.
- `torch::linalg::svd/det` namespace doesn't exist in C++ — use `torch::linalg_svd` / `torch::linalg_det` (flat namespace).
- `ModuleList::at<T>()` requires the Impl type — use `ptr<FooImpl>(i)` to get a `shared_ptr` that constructs into a holder.
- `IntArrayRef` has no `.str()` — stream through `std::ostringstream`.
- **All `ModuleList` members must be `register_module`-d before `push_back`**: `model->to(device)` only traverses registered modules. Unregistered `ModuleList`s (e.g. `decoder_`, `blocks_`, `res_conv_`) will keep their parameters on CPU even after `model->to(cuda)`. Register first, then fill: `register_module("blocks", blocks_); blocks_->push_back(...)`.
- **SVD requires float32 on CUDA** — `torch::linalg_svd` does not support bfloat16; cast to float32 before calling and cast the result back.
- **`lr_fc` runs in model dtype** — unlike Python's `autocast(enabled=False)` pattern, in C++ run `lr_fc_->forward(x)` in the model's dtype (bf16) and `.to(torch::kFloat32)` the output for the learning rate computation.
- **TTT `pre_norm` is `RMSNorm`, not `LayerNorm`** — no bias, use custom `loger::RMSNorm`.
- **`TransformerDecoder` uses standard self-attention `BlockRope`**, not cross-attention — `qk_norm=false`, `init_values=0.0`, `use_swiglu=false` (GELU MLP).
- **LayerNorm must run in float32** — Python autocast promotes LayerNorm to float32, but C++ `model->to(bf16)` runs it in bf16. Use `torch::layer_norm()` functional API with explicit weight/input casting to float32 (see `ln_f32()` helper in `block_rope.cpp`, `encoder.cpp`, `attention.cpp`).
- **QK-norm must be applied in `compute_kv()`** — Python's `compute_kv` applies both `q_norm` and `k_norm` before caching K. The C++ `compute_kv` must also apply `k_norm` (via `ln_f32(k_norm_, k)`) to match.

## Run

```bash
./build/loger_infer \
  --input ../data/examples/office \
  --model_name ../ckpts/LoGeR/latest.pt \
  --config ../ckpts/LoGeR/original_config.yaml \
  --output_ply output.ply \
  --output_traj trajectory.txt \
  --output_pt output.pt \
  --window_size 32 --overlap_size 3
```

`--output_pt` writes 4 numpy files (`output_points.npy`, `output_conf.npy`, `output_images.npy`, `output_poses.npy`) used by the per-frame visualizer.

Key flags: `--no_ttt`, `--no_swa`, `--no_se3`, `--cpu`, `--conf_threshold`.

## Visualize Output

```bash
/home/giulianowf/Documents/models/venv_gguf/bin/python scripts/visualize.py \
  --ply output.ply --traj trajectory.txt
```

Then open **http://localhost:8080** in a browser. Shows point cloud + per-frame camera axes.
The GUI has a **Frame slider** and **Play/Stop** buttons to animate camera poses over time.
Options: `--max_points 200000` to subsample, `--port 8081` if 8080 is busy.

### Side-by-side comparison with Python reference

```bash
/home/giulianowf/Documents/models/venv_gguf/bin/python scripts/compare_vis.py
```

Spawns two viser servers:
- **http://localhost:8080** — C++ output
- **http://localhost:8081** — Python reference (`reference/examples_office_0_50_1.pt`)

Options: `--cpp_port`, `--ref_port`, `--conf 0.2` (confidence threshold), `--max_points`.

### Numerical comparison

```bash
/home/giulianowf/Documents/models/venv_gguf/bin/python scripts/compare_output.py
```

Reports ATE, RPE (translation + rotation), scale ratio, and point cloud nearest-neighbour distance
against `reference/examples_office_0_50_1.pt`.

**Numerical accuracy vs Python reference (as of 2026-03-18):** The C++ port is functionally correct
with no remaining bugs. All divergence is accumulated bf16 precision differences through 36 decoder layers.

| Output | Cosine Similarity | Status |
|--------|-------------------|--------|
| encoder_out | 0.999 | Excellent |
| decoder_concat | 0.993 (mean) | Good — ~0.001/layer bf16 drift over 36 layers |
| local_points | 0.9998 | Excellent |
| camera_poses | allclose(atol=0.05) PASS | Excellent |
| conf_raw | 0.86 | Poor — amplified decoder error through 5 conf_decoder blocks |

The conf_raw divergence is NOT a bug: feeding C++ decoder_concat through the Python conf_decoder
gives cosine 0.999985 vs C++ output, proving the TransformerDecoder is functionally identical.
The 0.993 decoder_concat input simply gets amplified through 5 self-attention blocks.

**Breakdown of decoder divergence:**
- Block forward: ~0.001 cosine loss per layer (bf16 matmul precision)
- TTT layers: negligible additional error (< 0.001 per layer)
- SWA adapters: small contribution at layer 26 where gate is positive (+0.24 mean)
- Without TTT/SWA: final layer cosine 0.985; with TTT/SWA: 0.993 (TTT/SWA actually help via gating)

Unit tests:

```bash
cd build && ctest --output-on-failure
# or individually:
./test_rope2d
./test_encoder ../../ckpts/LoGeR/latest.pt
./test_ttt
./test_e2e ../../ckpts/LoGeR/latest.pt ../../ckpts/LoGeR/original_config.yaml
```

## Project Structure

```
include/loger/
  model/      pi3.hpp, encoder.hpp, block_rope.hpp, ttt.hpp, task_heads.hpp
  ops/        rope2d.hpp, attention.hpp, mlp.hpp
  io/         weight_loader.hpp, image_loader.hpp, output_writer.hpp
  utils/      geometry.hpp, windowing.hpp
src/          mirrors include/ — all .cpp implementations
app/main.cpp  CLI binary
tests/        test_rope2d, test_encoder, test_ttt, test_e2e
```

## Architecture

The model mirrors the Python LoGeR (`../loger/models/pi3.py`) exactly:

1. **DINOv2 ViT-L/14 encoder** (`src/model/encoder.cpp`): 24 blocks, dim=1024, 4 register tokens. Patch size 14. Outputs `x_norm_patchtokens` shape `(B*N, hw, 1024)`.

2. **36-layer BlockRope decoder** (`src/model/block_rope.cpp`): Alternates frame-level (even layers, reshape to `B*N, hw, dim`) and global (odd layers, reshape to `B, N*hw, dim`) attention. Uses RoPE2D positional encoding, QK-norm, LayerScale (init=0.01), standard GELU MLP (`ffn_layer=Mlp`, NOT SwiGLU).

3. **TTT layers** (`src/model/ttt.cpp`): 18 `FastWeightGluMLPMultihead` inserted after even decoder layers [0,2,...,34]. Fast weights `w0,w1,w2` updated per-window via Newton-Schulz (MUON). State persists across windows in `WindowState.ttt`. `lr_fc` always runs in float32.

4. **SWA adapters** (`src/model/pi3.cpp`): 4 `BlockRope` adapters after layers [10,18,26,34]. KV cache persists across windows in `WindowState.swa_kv`.

5. **Task heads** (`src/model/task_heads.cpp`):
   - `TransformerDecoder` (5 BlockRope self-attention blocks, `in=2048→1024→out`) for points, confidence, camera
   - `LinearPts3d`: `pixel_shuffle(14)` to upsample patch tokens to full resolution
   - `CameraHead`: global avg pool → ResConvBlocks → SVD-orthogonalized SO(3) → 4×4 SE(3)

6. **Windowed inference** (`src/model/pi3.cpp`, `src/utils/windowing.cpp`): Slice N frames into windows of `window_size` with `overlap_size` overlap. SE3 alignment between overlap regions stitches windows together.

## Key Implementation Details

- **PATCH_START = 6**: 5 register tokens + 1 PE token prepended before patch tokens. Positions for these are zeros; patch positions are shifted +1.
- **Weight loading**: `TensorStore` uses `torch::pickle_load()` on raw `.pt` bytes. Handles `model_state_dict` wrapper and `module.` DDP prefix.
- **Task decoder blocks**: Use `qk_norm=false`, no LayerScale — unlike main decoder blocks which have `qk_norm=true`, `init_values=0.01`.
- **local_points format**: `[x*z, y*z, z]` where `z = exp(z_raw.clamp_max(15))`.
- **Global points**: `einsum("bnij,bnhwj->bnhwi", camera_poses, homogenize(local_points))[..., :3]`.
- **Confidence post-processing**: sigmoid + zero out depth-edge pixels (max_pool-based discontinuity detection).
- **dtype**: bfloat16 on CUDA (RTX 5090 supports it natively); Newton-Schulz casts to bfloat16 internally.
- **LayerNorm in float32**: All LayerNorm ops use `ln_f32()` helper — casts input and weights to float32, runs `torch::layer_norm()`, casts back. Matches Python autocast behavior.

## Python Reference

The original Python implementation is one directory up at `../loger/models/`:
- `pi3.py` — main model, windowing, SE3 merge logic
- `ttt.py` — TTT fast weight update math, Newton-Schulz
- `layers/pos_embed.py` — RoPE2D
- `layers/attention.py` — FlashAttentionRope, KV cache
- `layers/camera_head.py` — CameraHead, SVD orthogonalize
- `layers/transformer_head.py` — TransformerDecoder, LinearPts3d

## Verification

Run unit tests: `cd build && ctest --output-on-failure`.

### Debug tensor comparison tooling

Environment variables control debug tensor saves:
- `LOGER_DEBUG=1` — saves key intermediate tensors (encoder_out, decoder_concat, conf/point decoder outputs, final outputs) to `debug_cpp/`
- `LOGER_DEBUG_LAYERS=1` — saves `hidden` after each of the 36 decoder layers to `debug_cpp/decoder_layer_{0..35}.pt`
- `LOGER_DEBUG_COMPONENTS=1` — saves post_block, post_ttt, post_swa, and their sub-components (gate, raw output) per layer
- `LOGER_DEBUG_TD=1` — saves per-block intermediates inside TransformerDecoder (projects, block0..4)

Python reference tensors:
```bash
# Generate Python reference (saves to debug_ref/)
LOGER_DEBUG_LAYERS=1 /home/giulianowf/Documents/models/venv_gguf/bin/python scripts/debug_tensors.py

# Generate C++ tensors (saves to debug_cpp/)
LOGER_DEBUG=1 LOGER_DEBUG_LAYERS=1 ./build/loger_infer --input ../data/examples/office \
  --model_name ../ckpts/LoGeR/latest.pt --config ../ckpts/LoGeR/original_config.yaml \
  --window_size 3 --overlap_size 0 --no_se3
```

Comparison scripts:
- `scripts/compare_debug.py` — compares all stages (encoder, decoder_concat, points, conf, cameras)
- `scripts/compare_decoder_layers.py` — per-layer decoder comparison (36 layers)
- `scripts/test_conf_decoder_isolation.py` — feeds C++ decoder_concat through Python conf_decoder to isolate TransformerDecoder correctness
