#!/usr/bin/env python3
"""
Run Pi3 on the first 3 frames of the office scene and save intermediate tensors
for comparison against the C++ port.

Saves to debug_ref/ (relative to loger_cpp/):
  encoder_out.pt        (BN, hw, 1024)     after DINOv2 encoder
  decoder_concat.pt     (BN, PATCH_START+hw, 2048)  last 2 decoder layer concat
  point_head_raw.pt     (BN, H, W, 3)     raw LinearPts3d output (before exp)
  local_points.pt       (B, N, H, W, 3)   [x*z, y*z, z]
  camera_poses.pt       (B, N, 4, 4)      SE(3) camera poses
  conf_raw.pt           (BN, H, W, 1)     raw confidence (before sigmoid)

Usage (from loger_cpp/):
  /home/giulianowf/Documents/models/venv_gguf/bin/python scripts/debug_tensors.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

from loger.models.pi3 import Pi3

# ── config ────────────────────────────────────────────────────────────────────
CKPT      = "../ckpts/LoGeR/latest.pt"
CFG_YAML  = "../ckpts/LoGeR/original_config.yaml"
IMG_DIR   = "../data/examples/office"
N_FRAMES  = 3          # number of frames in the single window
IMG_W     = 588        # must match C++ ImageLoader output (588x434 for office)
IMG_H     = 434        # run C++ first with LOGER_DEBUG=1 to see what it prints
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR   = Path("debug_ref")
OUT_DIR.mkdir(exist_ok=True)

# ── load model ────────────────────────────────────────────────────────────────
with open(CFG_YAML) as f:
    cfg = yaml.safe_load(f)
# Config may be nested under "model:" key
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
# strip DDP "module." prefix
state = {k.replace("module.", ""): v for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
if unexpected:
    print(f"  [warn] {len(unexpected)} unexpected keys (ignored)")
if missing:
    print(f"  [warn] {len(missing)} missing keys: {missing[:5]}...")
model.eval().to(DEVICE)
# Use float32 for Python reference (Python normally uses autocast; run in fp32 here)
# model = model.to(torch.float32)  # uncomment to disable autocast
print("Model loaded.")

# ── load images ───────────────────────────────────────────────────────────────
from natsort import natsorted
imgs_paths = natsorted(Path(IMG_DIR).glob("*.png"))[:N_FRAMES]
assert len(imgs_paths) == N_FRAMES, f"Need {N_FRAMES} images, found {len(imgs_paths)}"

tf = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),  # H first, then W
    transforms.ToTensor(),  # → [0,1]
])
imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in imgs_paths])  # (N,3,H,W)
imgs = imgs.unsqueeze(0).to(DEVICE)  # (1,N,3,H,W)
print(f"Input shape: {imgs.shape}")

# ── hook storage ──────────────────────────────────────────────────────────────
saved = {}

def hook_encoder(module, inp, out):
    if isinstance(out, dict):
        saved["encoder_out"] = out["x_norm_patchtokens"].detach().cpu().clone()
    else:
        saved["encoder_out"] = out.detach().cpu().clone()

# Hook after decoder (patch the decode method to capture)
_orig_decode = model.decode
def patched_decode(hidden, N, H, W, **kw):
    result = _orig_decode(hidden, N, H, W, **kw)
    # result[0] = concat of last 2 decoder outputs (BN, full_seq, 2*dim)
    saved["decoder_concat"] = result[0].detach().float().cpu().clone()
    saved["decoder_pos"]    = result[1].detach().cpu().clone() if result[1] is not None else None
    return result
model.decode = patched_decode

# Hook point_head forward to capture raw output
_orig_point_head_fwd = model.point_head.forward
def patched_point_head(decout, img_shape):
    out = _orig_point_head_fwd(decout, img_shape)
    saved["point_head_raw"] = out.detach().float().cpu().clone()
    return out
model.point_head.forward = patched_point_head

# Hook conf_decoder and point_decoder to capture intermediate outputs (including per-block)
import os as _os
def _make_td_patched_forward(decoder, tag, saved_dict):
    _orig_fwd = decoder.forward
    def patched_fwd(hidden, **kw):
        if _os.environ.get("LOGER_DEBUG_TD"):
            # Save per-block intermediates
            x = decoder.projects(hidden)
            torch.save(x.detach().float().cpu().clone(), OUT_DIR / f"{tag}_projects.pt")
            for i, blk in enumerate(decoder.blocks):
                x = blk(x, **kw)
                torch.save(x.detach().float().cpu().clone(), OUT_DIR / f"{tag}_block{i}.pt")
            out = decoder.linear_out(x)
        else:
            out = _orig_fwd(hidden, **kw)
        saved_dict[f"{tag}_out"] = out.detach().float().cpu().clone()
        return out
    return patched_fwd

model.conf_decoder.forward = _make_td_patched_forward(model.conf_decoder, "conf_decoder", saved)
model.point_decoder.forward = _make_td_patched_forward(model.point_decoder, "point_decoder", saved)

# Hook conf_head
_orig_conf_head_fwd = model.conf_head.forward
def patched_conf_head(decout, img_shape):
    out = _orig_conf_head_fwd(decout, img_shape)
    saved["conf_raw"] = out.detach().float().cpu().clone()
    return out
model.conf_head.forward = patched_conf_head

# Hook camera_head
_orig_cam_head_fwd = model.camera_head.forward
def patched_cam_head(feat, patch_h, patch_w):
    out = _orig_cam_head_fwd(feat, patch_h, patch_w)
    saved["camera_poses_raw"] = out.detach().float().cpu().clone()
    return out
model.camera_head.forward = patched_cam_head

# Register encoder hook
model.encoder.register_forward_hook(hook_encoder)

# ── run inference ─────────────────────────────────────────────────────────────
TURN_OFF_TTT = False   # set to True to match C++ --no_ttt flag
TURN_OFF_SWA = False   # set to True to match C++ --no_swa flag

with torch.no_grad():
    with torch.amp.autocast(device_type='cuda', enabled=True):
        result = model(
            imgs,
            window_size=N_FRAMES,
            overlap_size=0,
            se3=False,
            no_detach=True,
            turn_off_ttt=TURN_OFF_TTT,
            turn_off_swa=TURN_OFF_SWA,
        )

# Capture final outputs
saved["local_points"]  = result["local_points"].detach().float().cpu()
saved["camera_poses"]  = result["camera_poses"].detach().float().cpu()
saved["conf"]          = result.get("conf", None)
if saved["conf"] is not None:
    saved["conf"] = saved["conf"].detach().float().cpu()
saved["points"]        = result["points"].detach().float().cpu()

# ── save ──────────────────────────────────────────────────────────────────────
for key, val in saved.items():
    if val is not None:
        path = OUT_DIR / f"{key}.pt"
        torch.save(val, path)
        if val.is_floating_point():
            print(f"  {key:25s}: shape={tuple(val.shape)}, "
                  f"min={val.min().item():.4f}, max={val.max().item():.4f}, "
                  f"mean={val.mean().item():.4f}")
        else:
            print(f"  {key:25s}: shape={tuple(val.shape)}, dtype={val.dtype}")
    else:
        print(f"  {key:25s}: None")

print(f"\nSaved to {OUT_DIR.resolve()}/")
print("\n--- camera_poses (first window) ---")
print(saved["camera_poses"])
print("\n--- local_points stats per frame ---")
lp = saved["local_points"]  # (B,N,H,W,3)
for n in range(lp.shape[1]):
    z = lp[0, n, ..., 2]
    print(f"  frame {n}: z min={z.min():.4f} max={z.max():.4f} mean={z.mean():.4f}")
