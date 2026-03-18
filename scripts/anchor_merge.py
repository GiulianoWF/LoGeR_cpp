#!/usr/bin/env python3
"""Anchor-frame merge: use sparse keyframes for global consistency.

Instead of chaining SE3 across 172 windows (drift accumulates), this:
1. Runs inference on sparse anchor frames (every ~100 frames) → global skeleton
2. Aligns each dense window to its nearest anchor frame (1 hop, no drift)

Usage:
    # Step 1: Run anchor inference (only needed once per video)
    python scripts/anchor_merge.py --step anchor \
        --video casa_vo.mp4 --anchor_stride 100 \
        --anchor_dir anchor_windows/

    # Step 2: Merge dense windows using anchors
    python scripts/anchor_merge.py --step merge \
        --windows_dir windows/ --anchor_dir anchor_windows/ \
        --output_ply anchored.ply --output_traj anchored_traj.txt

    # Or both steps at once:
    python scripts/anchor_merge.py \
        --video casa_vo.mp4 --windows_dir windows/ \
        --anchor_stride 100 --anchor_dir anchor_windows/ \
        --output_ply anchored.ply --output_traj anchored_traj.txt
"""
import argparse
import os
import sys
import subprocess
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from merge_windows import (load_meta, load_window, apply_se3, homogenize,
                           conf_postprocess, write_ply, write_traj)


def run_anchor_inference(video_path, anchor_stride, anchor_dir,
                         model_name, config, window_size=32, overlap_size=3):
    """Run C++ binary on sparse anchor frames."""
    os.makedirs(anchor_dir, exist_ok=True)

    binary = os.path.join(_SCRIPT_DIR, "..", "build", "loger_infer")
    if not os.path.exists(binary):
        raise FileNotFoundError(f"C++ binary not found: {binary}")

    cmd = [
        binary,
        "--input", video_path,
        "--stride", str(anchor_stride),
        "--model_name", model_name,
        "--config", config,
        "--window_size", str(window_size),
        "--overlap_size", str(overlap_size),
        "--save_windows", anchor_dir,
    ]
    print(f"Running anchor inference with stride={anchor_stride}:")
    print(f"  {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("Anchor inference complete.")


def load_anchor_poses(anchor_dir, anchor_stride, total_frames):
    """Load anchor poses and build global_frame_idx -> pose mapping."""
    meta = load_meta(anchor_dir)
    n_anchor_windows = meta["n_windows"]
    anchor_overlap = meta["overlap_size"]

    # Reconstruct which global frame each anchor corresponds to
    anchor_global_indices = list(range(0, total_frames, anchor_stride))
    n_anchors = len(anchor_global_indices)
    print(f"  {n_anchors} anchor frames, {n_anchor_windows} anchor windows")

    # Load and merge anchor windows (short chain, 1-3 windows, negligible drift)
    all_poses = []
    prev_poses = None
    for wi in range(n_anchor_windows):
        w = load_window(anchor_dir, wi)
        poses = w["poses"].astype(np.float32)  # (B, N_w, 4, 4)

        if wi > 0 and anchor_overlap > 0 and prev_poses is not None:
            # Align to previous anchor window
            R, t = _estimate_se3_single(prev_poses, poses, anchor_overlap)
            T = np.tile(np.eye(4, dtype=np.float32), (poses.shape[0], 1, 1))
            T[:, :3, :3] = R
            T[:, :3, 3] = t
            poses = T[:, None] @ poses

        skip = 0 if wi == 0 else anchor_overlap
        all_poses.append(poses[:, skip:])
        prev_poses = poses

    merged_poses = np.concatenate(all_poses, axis=1)  # (B, n_anchors, 4, 4)
    assert merged_poses.shape[1] == n_anchors, \
        f"Expected {n_anchors} anchor poses, got {merged_poses.shape[1]}"

    # Build mapping: global_frame_idx -> pose (4x4)
    anchor_map = {}
    for i, gf in enumerate(anchor_global_indices):
        anchor_map[gf] = merged_poses[0, i]  # drop batch dim

    return anchor_map, anchor_global_indices


def _estimate_se3_single(prev_poses, curr_poses, overlap_size):
    """SE3 from first overlap frame (same as Python reference)."""
    prev_ov = prev_poses[:, -overlap_size:]
    curr_ov = curr_poses[:, :overlap_size]
    prev_R = prev_ov[:, 0, :3, :3]
    curr_R = curr_ov[:, 0, :3, :3]
    R_rel = prev_R @ curr_R.transpose(0, 2, 1)
    prev_t0 = prev_ov[:, 0, :3, 3]
    curr_t0 = curr_ov[:, 0, :3, 3]
    t_rel = prev_t0 - (R_rel @ curr_t0[..., None]).squeeze(-1)
    return R_rel, t_rel


def find_window_anchor(wi_start, window_size, anchor_global_indices):
    """Find anchor frame(s) that fall within a dense window's frame range.

    Returns list of (anchor_global_idx, local_frame_idx_in_window).
    """
    wi_end = wi_start + window_size
    matches = []
    for gf in anchor_global_indices:
        if wi_start <= gf < wi_end:
            matches.append((gf, gf - wi_start))
    return matches


def align_window_to_anchor(window_data, local_anchor_idx, anchor_pose):
    """Align a dense window so that its anchor frame matches the anchor pose.

    Computes T such that T @ dense_pose[anchor_idx] = anchor_pose,
    then applies T to all frames.
    """
    dense_pose = window_data["poses"][0, local_anchor_idx]  # (4, 4)

    # T = anchor_pose @ inv(dense_pose)
    T = anchor_pose @ np.linalg.inv(dense_pose)

    R = T[:3, :3][None]  # (1, 3, 3)
    t = T[:3, 3][None]   # (1, 3)
    return apply_se3(window_data, R, t)


def anchor_merge(windows_dir, anchor_map, anchor_global_indices,
                 conf_threshold=0.1, max_points=2_000_000):
    """Merge all dense windows using anchor-based alignment."""
    meta = load_meta(windows_dir)
    n_windows = meta["n_windows"]
    window_size = meta["window_size"]
    overlap_size = meta["overlap_size"]
    starts = meta["starts"]

    print(f"\nMerging {n_windows} windows using {len(anchor_global_indices)} anchors...")

    # Phase 1: Directly align windows that contain an anchor frame
    aligned = {}  # wi -> aligned window data
    anchor_window_map = {}  # wi -> True if directly anchored

    for wi in range(n_windows):
        wi_start = starts[wi]
        matches = find_window_anchor(wi_start, window_size, anchor_global_indices)
        if matches:
            # Use the first anchor in this window
            anchor_gf, local_idx = matches[0]
            anchor_pose = anchor_map[anchor_gf]

            w = load_window(windows_dir, wi)
            for k in w:
                w[k] = w[k].astype(np.float32)

            aligned[wi] = align_window_to_anchor(w, local_idx, anchor_pose)
            if "images" in w:
                aligned[wi]["images"] = w["images"]
            anchor_window_map[wi] = True

    print(f"  Phase 1: {len(aligned)}/{n_windows} windows directly anchored")

    # Phase 2: Align remaining windows via short chains to nearest anchored window
    unanchored = [wi for wi in range(n_windows) if wi not in aligned]
    if unanchored:
        print(f"  Phase 2: bridging {len(unanchored)} unanchored windows...")

    for wi in unanchored:
        # Find nearest anchored window
        best_dist = float("inf")
        best_neighbor = None
        for awi in aligned:
            d = abs(wi - awi)
            if d < best_dist:
                best_dist = d
                best_neighbor = awi

        # Chain from wi to best_neighbor through overlap
        w = load_window(windows_dir, wi)
        for k in w:
            w[k] = w[k].astype(np.float32)

        if best_neighbor is not None and best_dist <= 4:
            # Short chain: align wi to neighbor via pairwise SE3
            chain = _build_chain(wi, best_neighbor, windows_dir, aligned,
                                 overlap_size)
            aligned[wi] = chain
            if "images" in w:
                aligned[wi]["images"] = w["images"]
        else:
            # Fallback: just use raw (shouldn't happen with anchor_stride <= 100)
            print(f"    WARNING: window {wi} has no nearby anchor (dist={best_dist})")
            aligned[wi] = w

    # Phase 3: Concatenate, dropping overlaps
    print("  Phase 3: concatenating...")
    all_pts, all_rgb, all_conf, all_poses, all_local = [], [], [], [], []

    for wi in range(n_windows):
        w = aligned[wi]
        skip = 0 if wi == 0 else overlap_size

        all_pts.append(w["points"][:, skip:])
        all_local.append(w["local_points"][:, skip:])
        all_conf.append(w["conf"][:, skip:])
        all_poses.append(w["poses"][:, skip:])
        if "images" in w:
            all_rgb.append(w["images"][:, skip:] if "images" in w else None)

    merged = {
        "points": np.concatenate(all_pts, axis=1),
        "local_points": np.concatenate(all_local, axis=1),
        "conf": np.concatenate(all_conf, axis=1),
        "poses": np.concatenate(all_poses, axis=1),
    }
    if all_rgb and all_rgb[0] is not None:
        merged["images"] = np.concatenate(all_rgb, axis=1)

    # Confidence post-processing
    merged["conf"] = conf_postprocess(merged["conf"], merged["local_points"])

    return merged


def _build_chain(wi, target_wi, windows_dir, aligned, overlap_size):
    """Chain SE3 from window wi to an already-aligned window target_wi."""
    # Direction: step toward target
    step = 1 if target_wi > wi else -1
    chain_wis = list(range(wi, target_wi, step))

    # Start with raw window wi
    w = load_window(windows_dir, wi)
    for k in w:
        w[k] = w[k].astype(np.float32)

    current = w
    for i, cwi in enumerate(chain_wis):
        next_wi = cwi + step
        if next_wi in aligned:
            # Align current to the already-aligned neighbor
            neighbor = aligned[next_wi]
        else:
            # Load raw neighbor (shouldn't happen in normal flow)
            neighbor = load_window(windows_dir, next_wi)
            for k in neighbor:
                neighbor[k] = neighbor[k].astype(np.float32)

        # Estimate SE3 between current and neighbor
        if step > 0:
            # current's last frames overlap with neighbor's first frames
            R, t = _estimate_se3_single(neighbor["poses"], current["poses"],
                                        overlap_size)
            # We want: T @ current = aligned to neighbor's space
            # neighbor's first overlap = current's last overlap
            prev_ov = neighbor["poses"][:, :overlap_size]
            curr_ov = current["poses"][:, -overlap_size:]
        else:
            prev_ov = neighbor["poses"][:, -overlap_size:]
            curr_ov = current["poses"][:, :overlap_size]

        prev_R = prev_ov[:, 0, :3, :3]
        curr_R = curr_ov[:, 0, :3, :3]
        R_rel = prev_R @ curr_R.transpose(0, 2, 1)
        prev_t0 = prev_ov[:, 0, :3, 3]
        curr_t0 = curr_ov[:, 0, :3, 3]
        t_rel = prev_t0 - (R_rel @ curr_t0[..., None]).squeeze(-1)

        current = apply_se3(current, R_rel, t_rel)

    return current


def main():
    ap = argparse.ArgumentParser(description="Anchor-frame merge")
    ap.add_argument("--step", choices=["anchor", "merge", "both"], default="both",
                    help="Run anchor inference, merge, or both")
    # Anchor inference
    ap.add_argument("--video", default=None, help="Video file for anchor inference")
    ap.add_argument("--anchor_stride", type=int, default=100,
                    help="Pick one anchor every N video frames")
    ap.add_argument("--anchor_dir", default="anchor_windows/",
                    help="Directory for anchor window outputs")
    ap.add_argument("--model_name", default="../ckpts/LoGeR/latest.pt")
    ap.add_argument("--config", default="../ckpts/LoGeR/original_config.yaml")
    ap.add_argument("--anchor_window_size", type=int, default=32)
    ap.add_argument("--anchor_overlap_size", type=int, default=3)
    # Dense merge
    ap.add_argument("--windows_dir", default="windows/",
                    help="Directory with dense window predictions")
    ap.add_argument("--output_ply", default="anchored.ply")
    ap.add_argument("--output_traj", default="anchored_traj.txt")
    ap.add_argument("--conf_threshold", type=float, default=0.1)
    ap.add_argument("--max_points", type=int, default=2_000_000)
    args = ap.parse_args()

    # Step 1: Anchor inference
    if args.step in ("anchor", "both"):
        if not args.video:
            ap.error("--video required for anchor inference")
        run_anchor_inference(
            args.video, args.anchor_stride, args.anchor_dir,
            args.model_name, args.config,
            args.anchor_window_size, args.anchor_overlap_size)

    # Step 2: Merge
    if args.step in ("merge", "both"):
        dense_meta = load_meta(args.windows_dir)
        total_frames = dense_meta["total_frames"]

        anchor_map, anchor_indices = load_anchor_poses(
            args.anchor_dir, args.anchor_stride, total_frames)

        merged = anchor_merge(
            args.windows_dir, anchor_map, anchor_indices,
            args.conf_threshold, args.max_points)

        # Write outputs
        points = merged["points"][0]
        conf = merged["conf"][0]
        poses = merged["poses"][0]
        N, H, W = points.shape[:3]

        if "images" in merged:
            colors = merged["images"][0]
        else:
            colors = np.ones((N, H, W, 3), dtype=np.float32) * 0.7

        write_ply(args.output_ply, points, colors, conf,
                  args.conf_threshold)
        write_traj(args.output_traj, poses)
        print(f"\nWrote {args.output_ply} and {args.output_traj}")
        print(f"Visualize: python scripts/visualize.py "
              f"--ply {args.output_ply} --traj {args.output_traj}")


if __name__ == "__main__":
    main()
