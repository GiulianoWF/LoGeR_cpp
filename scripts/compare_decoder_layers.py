#!/usr/bin/env python3
"""Compare per-layer decoder outputs between Python and C++."""
import torch
from pathlib import Path

def cosine_sim(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return (torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm())).item()

def per_sample_cosine(a, b):
    """Per-sample (dim 0) cosine similarities."""
    cos_vals = []
    for i in range(a.shape[0]):
        cos_vals.append(cosine_sim(a[i], b[i]))
    return cos_vals

print(f"{'Layer':>8} {'Mean cos':>10} {'Min cos':>10} {'Max diff':>10}")
print("-" * 45)

for i in range(36):
    ref_path = f"debug_ref/decoder_layer_{i}.pt"
    cpp_path = f"debug_cpp/decoder_layer_{i}.pt"
    if not Path(ref_path).exists() or not Path(cpp_path).exists():
        print(f"  layer {i}: missing files")
        continue
    ref = torch.load(ref_path, map_location="cpu")
    cpp = torch.load(cpp_path, map_location="cpu")

    # Shapes may differ (frame vs global reshape) — just flatten
    ref_flat = ref.flatten().float()
    cpp_flat = cpp.flatten().float()
    if ref_flat.numel() != cpp_flat.numel():
        print(f"{i:>8} SHAPE MISMATCH: ref={ref.shape} cpp={cpp.shape}")
        continue
    cos = cosine_sim(ref, cpp)
    max_diff = (ref_flat - cpp_flat).abs().max().item()
    # Per-sample: reshape to match
    if ref.shape == cpp.shape:
        per_sample = per_sample_cosine(ref, cpp)
        min_cos = min(per_sample)
    else:
        min_cos = cos

    marker = ""
    if cos < 0.999:
        marker = " <<<" if cos < 0.99 else " <"
    print(f"{i:>8} {cos:>10.6f} {min_cos:>10.6f} {max_diff:>10.4f}{marker}")
