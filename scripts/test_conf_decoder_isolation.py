#!/usr/bin/env python3
"""
Isolate whether conf_decoder divergence is due to:
  A) The TransformerDecoder code itself (functional bug)
  B) Amplified decoder_concat input differences

Loads both Python and C++ decoder_concat, runs the Python conf_decoder
on BOTH inputs, and compares:
  1. py_input → py_conf_decoder → A   (reference)
  2. cpp_input → py_conf_decoder → B   (tests input sensitivity)
  3. cpp_input → cpp_conf_decoder → C  (already saved)

If B ≈ C: no TransformerDecoder bug, just input amplification
If B ≠ C: real bug in C++ TransformerDecoder
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import yaml
import numpy as np
from pathlib import Path
from copy import deepcopy

from loger.models.pi3 import Pi3

# ── config ────────────────────────────────────────────────────────────────────
CKPT      = "../ckpts/LoGeR/latest.pt"
CFG_YAML  = "../ckpts/LoGeR/original_config.yaml"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ── load model ────────────────────────────────────────────────────────────────
with open(CFG_YAML) as f:
    cfg = yaml.safe_load(f)
if "model" in cfg and isinstance(cfg["model"], dict):
    cfg = cfg["model"]

model = Pi3(
    pos_type=cfg.get("pos_type", "rope100"),
    decoder_size=cfg.get("decoder_size", "large"),
    ttt_insert_after=cfg.get("ttt_insert_after", []),
    ttt_head_dim=cfg.get("ttt_head_dim", 512),
    ttt_inter_multi=cfg.get("ttt_inter_multi", 2),
    num_muon_update_steps=cfg.get("num_muon_update_steps", 5),
    use_momentum=cfg.get("use_momentum", False),
    ttt_update_steps=cfg.get("ttt_update_steps", 1),
    conf=cfg.get("conf", True),
    attn_insert_after=cfg.get("attn_insert_after", None),
    ttt_pre_norm=cfg.get("ttt_pre_norm", False),
)
state = torch.load(CKPT, map_location="cpu", weights_only=False)
if "model_state_dict" in state:
    state = state["model_state_dict"]
state = {k.replace("module.", ""): v for k, v in state.items()}
model.load_state_dict(state, strict=False)
model.eval().to(DEVICE)
print("Model loaded.")

# ── load debug tensors ────────────────────────────────────────────────────────
py_decoder_concat = torch.load("debug_ref/decoder_concat.pt", map_location=DEVICE)
cpp_decoder_concat = torch.load("debug_cpp/decoder_concat.pt", map_location=DEVICE)
cpp_conf_out = torch.load("debug_cpp/conf_decoder_out.pt", map_location=DEVICE)
py_conf_out = torch.load("debug_ref/conf_decoder_out.pt", map_location=DEVICE)

# Also load positions (needed for RoPE in the TransformerDecoder)
py_decoder_pos = torch.load("debug_ref/decoder_pos.pt", map_location=DEVICE)

print(f"py_decoder_concat:  shape={py_decoder_concat.shape}, dtype={py_decoder_concat.dtype}")
print(f"cpp_decoder_concat: shape={cpp_decoder_concat.shape}, dtype={cpp_decoder_concat.dtype}")
print(f"py_conf_out:        shape={py_conf_out.shape}, dtype={py_conf_out.dtype}")
print(f"cpp_conf_out:       shape={cpp_conf_out.shape}, dtype={cpp_conf_out.dtype}")
print(f"py_decoder_pos:     shape={py_decoder_pos.shape}, dtype={py_decoder_pos.dtype}")

def cosine_sim(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return (torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm())).item()

print(f"\n--- Input comparison ---")
print(f"decoder_concat py vs cpp: cos={cosine_sim(py_decoder_concat, cpp_decoder_concat):.6f}")

# ── Run Python conf_decoder on both inputs ────────────────────────────────────
conf_decoder = model.conf_decoder
print(f"\npatch_start_idx = {model.patch_start_idx}")

with torch.no_grad():
    with torch.amp.autocast(device_type='cuda', enabled=True):
        # A: Python input → Python conf_decoder (should match py_conf_out)
        out_A = conf_decoder(py_decoder_concat, xpos=py_decoder_pos)

        # B: C++ input → Python conf_decoder
        out_B = conf_decoder(cpp_decoder_concat, xpos=py_decoder_pos)

out_A = out_A.float()
out_B = out_B.float()

print(f"\n--- Results ---")
print(f"A (py_input → py_conf_dec):  shape={out_A.shape}")
print(f"B (cpp_input → py_conf_dec): shape={out_B.shape}")

print(f"\nA vs py_conf_out (sanity):     cos={cosine_sim(out_A, py_conf_out):.6f}")
print(f"B vs cpp_conf_out:             cos={cosine_sim(out_B, cpp_conf_out):.6f}")
print(f"A vs B (input sensitivity):    cos={cosine_sim(out_A, out_B):.6f}")
print(f"py_conf_out vs cpp_conf_out:   cos={cosine_sim(py_conf_out, cpp_conf_out):.6f}")

print(f"\n--- Interpretation ---")
b_vs_cpp = cosine_sim(out_B, cpp_conf_out)
if b_vs_cpp > 0.99:
    print("B ≈ C++ output → NO TransformerDecoder bug, just input amplification")
else:
    print(f"B ≠ C++ output (cos={b_vs_cpp:.6f}) → REAL BUG in C++ TransformerDecoder")
    # Do per-block analysis
    print("\n--- Per-block analysis (C++ input through Python conf_decoder) ---")
    x = conf_decoder.projects(cpp_decoder_concat)
    x_proj = x.float().cpu()
    cpp_proj = torch.load("debug_cpp/conf_decoder_projects.pt", map_location="cpu")
    print(f"  projects: cos={cosine_sim(x_proj, cpp_proj):.6f}")

    for i, blk in enumerate(conf_decoder.blocks):
        x = blk(x, xpos=py_decoder_pos)
        x_blk = x.float().cpu()
        cpp_blk_path = f"debug_cpp/conf_decoder_block{i}.pt"
        if os.path.exists(cpp_blk_path):
            cpp_blk = torch.load(cpp_blk_path, map_location="cpu")
            print(f"  block{i}:   cos={cosine_sim(x_blk, cpp_blk):.6f}")
        else:
            print(f"  block{i}:   (no C++ reference)")
