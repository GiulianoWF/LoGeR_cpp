#!/usr/bin/env python3
"""
Compare intermediate tensors saved by the Python reference (debug_ref/)
and the C++ port (debug_cpp/).

Usage (from loger_cpp/):
  /home/giulianowf/Documents/models/venv_gguf/bin/python scripts/compare_debug.py
"""
import sys, os, io, struct
from pathlib import Path
import torch
import numpy as np

REF_DIR = Path("debug_ref")
CPP_DIR = Path("debug_cpp")

STAGES = [
    "encoder_out",
    "decoder_concat",
    "point_head_raw",
    "local_points",
    "conf_raw",
    "camera_poses",
]

def load(path: Path) -> torch.Tensor:
    return torch.load(str(path), map_location="cpu", weights_only=False).float()

def stats(t: torch.Tensor):
    return (f"shape={tuple(t.shape)}  "
            f"min={t.min():.5g}  max={t.max():.5g}  "
            f"mean={t.mean():.5g}  std={t.std():.5g}")

def compare(ref: torch.Tensor, cpp: torch.Tensor, name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  REF : {stats(ref)}")
    print(f"  C++ : {stats(cpp)}")

    if ref.shape != cpp.shape:
        print(f"  !! SHAPE MISMATCH: ref={ref.shape} vs cpp={cpp.shape}")
        return

    diff = (ref - cpp).abs()
    rel  = diff / (ref.abs() + 1e-8)
    print(f"  diff: max={diff.max():.5g}  mean={diff.mean():.5g}  "
          f"rel_max={rel.max():.5g}  rel_mean={rel.mean():.5g}")

    # Cosine similarity along last dim (when rank > 1)
    if ref.dim() >= 2:
        r_flat = ref.flatten(0, -2)   # (N, D)
        c_flat = cpp.flatten(0, -2)
        cos = torch.nn.functional.cosine_similarity(r_flat, c_flat, dim=-1)
        print(f"  cosine sim: min={cos.min():.5g}  mean={cos.mean():.5g}")

    atol = 0.05
    close = torch.allclose(ref, cpp, atol=atol, rtol=1e-3)
    print(f"  allclose(atol={atol}): {'PASS ✓' if close else 'FAIL ✗'}")

for stage in STAGES:
    ref_path = REF_DIR / f"{stage}.pt"
    cpp_path = CPP_DIR / f"{stage}.pt"

    if not ref_path.exists():
        print(f"\n[skip] {stage}: no Python reference at {ref_path}")
        continue
    if not cpp_path.exists():
        print(f"\n[skip] {stage}: no C++ output at {cpp_path}")
        continue

    ref_t = load(ref_path)
    cpp_t = load(cpp_path)
    compare(ref_t, cpp_t, stage)

# Per-layer decoder comparison (LOGER_DEBUG_LAYERS)
has_layers = (REF_DIR / "decoder_layer_0.pt").exists() and (CPP_DIR / "decoder_layer_0.pt").exists()
if has_layers:
    print(f"\n{'='*60}")
    print("  PER-LAYER DECODER COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Layer':>5s}  {'Cosine':>8s}  {'MaxDiff':>10s}  {'MeanDiff':>10s}")
    print(f"  {'-'*5:>5s}  {'-'*8:>8s}  {'-'*10:>10s}  {'-'*10:>10s}")
    for i in range(36):
        ref_path = REF_DIR / f"decoder_layer_{i}.pt"
        cpp_path = CPP_DIR / f"decoder_layer_{i}.pt"
        if not ref_path.exists() or not cpp_path.exists():
            break
        ref_t = load(ref_path)
        cpp_t = load(cpp_path)
        if ref_t.shape != cpp_t.shape:
            # Odd layers: Python saves (B, N*hw, dim), C++ saves (BN, hw, dim) — reshape to match
            if ref_t.numel() == cpp_t.numel():
                ref_t = ref_t.reshape(cpp_t.shape)
            else:
                print(f"  {i:5d}  SHAPE MISMATCH ref={ref_t.shape} cpp={cpp_t.shape}")
                continue
        diff = (ref_t - cpp_t).abs()
        r_flat = ref_t.flatten(0, -2)
        c_flat = cpp_t.flatten(0, -2)
        cos = torch.nn.functional.cosine_similarity(r_flat, c_flat, dim=-1).mean()
        print(f"  {i:5d}  {cos:.6f}  {diff.max():.6f}  {diff.mean():.6f}")

# Component-level comparison (LOGER_DEBUG_COMPONENTS)
swa_layers = [10, 18, 26, 34]
has_components = (REF_DIR / "layer_0_post_block.pt").exists() and (CPP_DIR / "layer_0_post_block.pt").exists()
if has_components:
    print(f"\n{'='*60}")
    print("  COMPONENT-LEVEL COMPARISON")
    print(f"{'='*60}")
    for i in range(36):
        for suffix in ["post_block", "post_ttt", "ttt_out", "ttt_gate",
                        "post_swa", "swa_out", "swa_gate"]:
            ref_path = REF_DIR / f"layer_{i}_{suffix}.pt"
            cpp_path = CPP_DIR / f"layer_{i}_{suffix}.pt"
            if not ref_path.exists() or not cpp_path.exists():
                continue
            ref_t = load(ref_path)
            cpp_t = load(cpp_path)
            if ref_t.numel() == cpp_t.numel():
                ref_t = ref_t.reshape(cpp_t.shape)
            elif ref_t.shape != cpp_t.shape:
                print(f"  layer_{i}_{suffix}: SHAPE MISMATCH ref={ref_t.shape} cpp={cpp_t.shape}")
                continue
            diff = (ref_t - cpp_t).abs()
            r_flat = ref_t.flatten(0, -2)
            c_flat = cpp_t.flatten(0, -2)
            cos = torch.nn.functional.cosine_similarity(r_flat, c_flat, dim=-1).mean()
            print(f"  layer_{i:2d}_{suffix:12s}  cos={cos:.6f}  maxdiff={diff.max():.6f}  meandiff={diff.mean():.6f}")

# Extra: print camera_poses side-by-side
for tag, path in [("REF", REF_DIR / "camera_poses.pt"), ("C++", CPP_DIR / "camera_poses.pt")]:
    if path.exists():
        t = load(path)
        print(f"\n--- {tag} camera_poses (B,N,4,4) ---")
        for n in range(t.shape[1] if t.dim() >= 2 else 1):
            idx = (0, n) if t.dim() == 4 else (n,)
            mat = t[idx] if t.dim() == 4 else t[n]
            tx, ty, tz = mat[0,3].item(), mat[1,3].item(), mat[2,3].item()
            print(f"  frame {n}: t=({tx:.4f}, {ty:.4f}, {tz:.4f})")

# Extra: z-depth stats per frame for local_points
for tag, path in [("REF", REF_DIR / "local_points.pt"), ("C++", CPP_DIR / "local_points.pt")]:
    if path.exists():
        lp = load(path)  # (B,N,H,W,3)
        print(f"\n--- {tag} local_points z-depth per frame ---")
        for n in range(lp.shape[1]):
            z = lp[0, n, ..., 2]
            print(f"  frame {n}: z min={z.min():.4f} max={z.max():.4f} mean={z.mean():.4f}")
        xy = lp[0, :, ..., :2]
        print(f"  xy range: min={xy.min():.4f} max={xy.max():.4f}")
