#!/usr/bin/env python3
"""Merge saved per-window predictions from loger_infer --save_windows.

Loads raw window .npy files, applies SE3 alignment, writes PLY + trajectory.
Iterate on merge strategies here without re-running inference.

Usage:
    python scripts/merge_windows.py --windows_dir windows/ \
        --output_ply output.ply --output_traj trajectory.txt
"""
import argparse
import os
import numpy as np
from pathlib import Path


def load_meta(windows_dir):
    meta = {}
    starts = {}
    with open(os.path.join(windows_dir, "meta.txt")) as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] == "start":
                starts[int(parts[1])] = int(parts[2])
            else:
                meta[parts[0]] = int(parts[1])
    meta["starts"] = starts
    return meta


def load_window(windows_dir, idx):
    prefix = os.path.join(windows_dir, f"window_{idx}")
    w = {
        "points": np.load(f"{prefix}_points.npy"),
        "local_points": np.load(f"{prefix}_local_points.npy"),
        "conf": np.load(f"{prefix}_conf.npy"),
        "poses": np.load(f"{prefix}_poses.npy"),
    }
    images_path = f"{prefix}_images.npy"
    if os.path.exists(images_path):
        w["images"] = np.load(images_path)  # (B, N_w, H, W, 3) float32 [0,1]
    return w


def homogenize(pts):
    """(..., 3) -> (..., 4) with last coord = 1."""
    return np.concatenate([pts, np.ones_like(pts[..., :1])], axis=-1)


def estimate_se3(prev_poses, curr_poses, overlap_size):
    """Estimate rigid transform aligning curr overlap to prev overlap.

    prev_poses: (B, N_prev, 4, 4) - already aligned
    curr_poses: (B, N_curr, 4, 4) - raw from current window
    Returns (R, t) where R is (B, 3, 3) and t is (B, 3).
    """
    # Use last overlap_size of prev, first overlap_size of curr
    prev_ov = prev_poses[:, -overlap_size:]
    curr_ov = curr_poses[:, :overlap_size]

    # Rotation from first overlap frame pair
    prev_R = prev_ov[:, 0, :3, :3]  # (B, 3, 3)
    curr_R = curr_ov[:, 0, :3, :3]
    R_rel = prev_R @ curr_R.transpose(0, 2, 1)  # (B, 3, 3)

    # Translation
    prev_t0 = prev_ov[:, 0, :3, 3]  # (B, 3)
    curr_t0 = curr_ov[:, 0, :3, 3]
    t_rel = prev_t0 - (R_rel @ curr_t0[..., None]).squeeze(-1)

    return R_rel, t_rel


def apply_se3(window, R, t):
    """Apply rigid transform to window predictions."""
    B, N = window["poses"].shape[:2]

    # Build 4x4 transform
    T = np.tile(np.eye(4, dtype=np.float32), (B, 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = t

    # Transform camera poses: T @ pose for each frame
    T_exp = T[:, None]  # (B, 1, 4, 4)
    new_poses = T_exp @ window["poses"]  # (B, N, 4, 4)

    # Re-derive global points from transformed poses and local points
    hom = homogenize(window["local_points"])  # (B, N, H, W, 4)
    new_points = np.einsum("bnij,bnhwj->bnhwi", new_poses, hom)[..., :3]

    return {
        "points": new_points,
        "local_points": window["local_points"],
        "conf": window["conf"],
        "poses": new_poses,
    }


def depth_edge(depth, rtol=0.02):
    """Detect depth discontinuities via max_pool."""
    from scipy.ndimage import maximum_filter
    # depth: (B, N, H, W)
    abs_diff = np.abs(maximum_filter(depth, size=(1, 1, 3, 3)) - depth)
    return abs_diff > (rtol * np.abs(depth) + 1e-6)


def conf_postprocess(conf, local_points):
    """Sigmoid + edge masking."""
    from scipy.special import expit
    # conf: (B, N, H, W, 1), local_points: (B, N, H, W, 3)
    z = local_points[..., 2]  # (B, N, H, W)
    edges = depth_edge(z)
    conf_sig = expit(conf[..., 0])  # (B, N, H, W)
    conf_sig[edges] = 0.0
    return conf_sig[..., None]  # (B, N, H, W, 1)


def merge_windows(windows_dir, overlap_size, use_se3=True, verbose=True,
                  max_windows=None):
    meta = load_meta(windows_dir)
    n_windows = meta["n_windows"]
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    all_points = []
    all_conf = []
    all_poses = []
    all_local = []
    all_images = []

    prev_aligned_poses = None

    for wi in range(n_windows):
        if verbose:
            print(f"\r  Merging window {wi+1}/{n_windows}", end="", flush=True)

        w = load_window(windows_dir, wi)
        # Cast to float32 for merge math
        for k in w:
            w[k] = w[k].astype(np.float32)

        if wi > 0 and use_se3 and overlap_size > 0:
            R, t = estimate_se3(prev_aligned_poses, w["poses"], overlap_size)
            w = apply_se3(w, R, t)
            if verbose:
                t_mag = np.linalg.norm(t)
                print(f"  (SE3 |t|={t_mag:.3f})", end="", flush=True)

        skip = 0 if wi == 0 else overlap_size
        all_points.append(w["points"][:, skip:])
        all_local.append(w["local_points"][:, skip:])
        all_conf.append(w["conf"][:, skip:])
        all_poses.append(w["poses"][:, skip:])
        if "images" in w:
            all_images.append(w["images"][:, skip:])

        # Keep full aligned window poses for next iteration's overlap ref
        prev_aligned_poses = w["poses"]

    if verbose:
        print()

    merged = {
        "points": np.concatenate(all_points, axis=1),
        "local_points": np.concatenate(all_local, axis=1),
        "conf": np.concatenate(all_conf, axis=1),
        "poses": np.concatenate(all_poses, axis=1),
    }
    if all_images:
        merged["images"] = np.concatenate(all_images, axis=1)

    # Confidence post-processing
    merged["conf"] = conf_postprocess(merged["conf"], merged["local_points"])

    return merged


def write_ply(path, points, colors, conf, threshold=0.1):
    """Write binary PLY."""
    mask = conf.ravel() > threshold
    pts = points.reshape(-1, 3)[mask]
    col = colors.reshape(-1, 3)[mask]
    col = (np.clip(col, 0, 1) * 255).astype(np.uint8)

    n = len(pts)
    print(f"  Writing {n:,} vertices to {path}")

    with open(path, "wb") as f:
        header = (
            f"ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"end_header\n"
        )
        f.write(header.encode())
        # Interleave xyz (float32) + rgb (uint8) per vertex
        dt = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"),
                       ("r", "u1"), ("g", "u1"), ("b", "u1")])
        out = np.empty(n, dtype=dt)
        out["x"], out["y"], out["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        out["r"], out["g"], out["b"] = col[:, 0], col[:, 1], col[:, 2]
        f.write(out.tobytes())


def quat_from_rot(R):
    """(N, 3, 3) -> (N, 4) as [qx, qy, qz, qw]."""
    N = R.shape[0]
    q = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        tr = np.trace(R[i])
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1)
            q[i] = [(R[i, 2, 1] - R[i, 1, 2]) * s,
                     (R[i, 0, 2] - R[i, 2, 0]) * s,
                     (R[i, 1, 0] - R[i, 0, 1]) * s,
                     0.25 / s]
        elif R[i, 0, 0] > R[i, 1, 1] and R[i, 0, 0] > R[i, 2, 2]:
            s = 2 * np.sqrt(1 + R[i, 0, 0] - R[i, 1, 1] - R[i, 2, 2])
            q[i] = [0.25 * s, (R[i, 0, 1] + R[i, 1, 0]) / s,
                     (R[i, 0, 2] + R[i, 2, 0]) / s, (R[i, 2, 1] - R[i, 1, 2]) / s]
        elif R[i, 1, 1] > R[i, 2, 2]:
            s = 2 * np.sqrt(1 + R[i, 1, 1] - R[i, 0, 0] - R[i, 2, 2])
            q[i] = [(R[i, 0, 1] + R[i, 1, 0]) / s, 0.25 * s,
                     (R[i, 1, 2] + R[i, 2, 1]) / s, (R[i, 0, 2] - R[i, 2, 0]) / s]
        else:
            s = 2 * np.sqrt(1 + R[i, 2, 2] - R[i, 0, 0] - R[i, 1, 1])
            q[i] = [(R[i, 0, 2] + R[i, 2, 0]) / s, (R[i, 1, 2] + R[i, 2, 1]) / s,
                     0.25 * s, (R[i, 1, 0] - R[i, 0, 1]) / s]
    return q


def write_traj(path, poses):
    """Write TUM-format trajectory."""
    N = poses.shape[0]
    R = poses[:, :3, :3]
    t = poses[:, :3, 3]
    q = quat_from_rot(R)
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for i in range(N):
            f.write(f"{i:.6f} {t[i,0]:.6f} {t[i,1]:.6f} {t[i,2]:.6f} "
                    f"{q[i,0]:.6f} {q[i,1]:.6f} {q[i,2]:.6f} {q[i,3]:.6f}\n")


def export_single_window(windows_dir, idx, output_ply, conf_threshold=0.1):
    """Export a single raw window as a PLY for inspection."""
    w = load_window(windows_dir, idx)
    for k in w:
        w[k] = w[k].astype(np.float32)

    points = w["points"][0]       # (N_w, H, W, 3)
    local_pts = w["local_points"][0]
    conf = w["conf"][0]           # (N_w, H, W, 1)
    poses = w["poses"][0]         # (N_w, 4, 4)

    # Conf postprocess
    from scipy.special import expit
    z = local_pts[..., 2]
    from scipy.ndimage import maximum_filter
    abs_diff = np.abs(maximum_filter(z, size=(1, 3, 3)) - z)
    edges = abs_diff > (0.02 * np.abs(z) + 1e-6)
    conf_sig = expit(conf[..., 0])
    conf_sig[edges] = 0.0

    # Use saved images for color if available, otherwise gray
    if "images" in w:
        colors = w["images"][0]  # (N_w, H, W, 3) float32 [0,1]
    else:
        colors = np.ones_like(points) * 0.7
    write_ply(output_ply, points, colors, conf_sig[..., None], conf_threshold)
    write_traj(output_ply.replace(".ply", "_traj.txt"), poses)


def export_all_windows(windows_dir, output_dir, conf_threshold=0.1):
    """Export every window as a separate PLY + trajectory."""
    meta = load_meta(windows_dir)
    n = meta["n_windows"]
    os.makedirs(output_dir, exist_ok=True)
    for i in range(n):
        ply_path = os.path.join(output_dir, f"window_{i:03d}.ply")
        print(f"\r  Exporting window {i+1}/{n}", end="", flush=True)
        export_single_window(windows_dir, i, ply_path, conf_threshold)
    print(f"\n  Wrote {n} PLY files to {output_dir}/")


def main():
    ap = argparse.ArgumentParser(description="Merge saved window predictions")
    ap.add_argument("--windows_dir", required=True, help="Directory from --save_windows")
    ap.add_argument("--output_ply", default="merged.ply")
    ap.add_argument("--output_traj", default="merged_traj.txt")
    ap.add_argument("--no_se3", action="store_true", help="Disable SE3 alignment")
    ap.add_argument("--conf_threshold", type=float, default=0.1)
    ap.add_argument("--images_npy", default=None,
                    help="Path to images .npy (N,H,W,3 float32 [0,1]) for PLY colors")
    # Single/all window export
    ap.add_argument("--window", type=int, default=None,
                    help="Export a single window by index (0-based)")
    ap.add_argument("--export_all", action="store_true",
                    help="Export every window as a separate PLY")
    ap.add_argument("--export_dir", default="window_plys",
                    help="Output dir for --export_all (default: window_plys/)")
    # Incremental merge
    ap.add_argument("--up_to", type=int, default=None,
                    help="Merge only windows 0..N (e.g. --up_to 3 merges 0,1,2,3)")
    ap.add_argument("--interactive", action="store_true",
                    help="Incremental merge: press Enter to add next window, "
                         "writes PLY after each step")
    args = ap.parse_args()

    meta = load_meta(args.windows_dir)
    print(f"Loaded metadata: {meta['n_windows']} windows, "
          f"overlap={meta['overlap_size']}, total_frames={meta['total_frames']}")

    # Single window mode
    if args.window is not None:
        print(f"Exporting window {args.window}...")
        export_single_window(args.windows_dir, args.window,
                             args.output_ply, args.conf_threshold)
        print(f"Wrote {args.output_ply}")
        return

    # Export all windows individually
    if args.export_all:
        export_all_windows(args.windows_dir, args.export_dir, args.conf_threshold)
        return

    # Interactive mode: add one window at a time, write PLY after each step
    if args.interactive:
        n = meta["n_windows"]
        print(f"Interactive mode: {n} windows available. Press Enter to add next, 'q' to quit.")
        for up_to in range(1, n + 1):
            merged = merge_windows(args.windows_dir, meta["overlap_size"],
                                   use_se3=not args.no_se3, max_windows=up_to,
                                   verbose=False)
            pts = merged["points"][0]
            conf = merged["conf"][0]
            poses = merged["poses"][0]
            N_m, H, W = pts.shape[:3]
            colors = merged.get("images", [None])[0]
            if colors is None:
                colors = np.ones((N_m, H, W, 3), dtype=np.float32) * 0.7
            write_ply(args.output_ply, pts, colors, conf, args.conf_threshold)
            write_traj(args.output_traj, poses)
            print(f"  [{up_to}/{n}] Wrote {args.output_ply} — "
                  f"{N_m} frames merged. Reload in viewer.")
            try:
                resp = input("  Press Enter for next window, 'q' to quit: ")
                if resp.strip().lower() == 'q':
                    break
            except (EOFError, KeyboardInterrupt):
                break
        return

    # Merge up to N windows
    max_win = args.up_to + 1 if args.up_to is not None else None

    # Full merge
    merged = merge_windows(args.windows_dir, meta["overlap_size"],
                           use_se3=not args.no_se3, max_windows=max_win)

    # Squeeze batch dim
    points = merged["points"][0]  # (N, H, W, 3)
    conf = merged["conf"][0]      # (N, H, W, 1)
    poses = merged["poses"][0]    # (N, 4, 4)
    N, H, W = points.shape[:3]

    # Colors: from merged window images, or external file, or gray
    if "images" in merged:
        colors = merged["images"][0]  # (N, H, W, 3) float32 [0,1]
    elif args.images_npy and os.path.exists(args.images_npy):
        print(f"Loading colors from {args.images_npy}")
        colors = np.load(args.images_npy)
        if colors.shape[1] == 3 and colors.shape[2] != 3:
            colors = colors.transpose(0, 2, 3, 1)
    else:
        colors = np.ones((N, H, W, 3), dtype=np.float32) * 0.7

    write_ply(args.output_ply, points, colors, conf, args.conf_threshold)
    write_traj(args.output_traj, poses)
    print(f"Wrote {args.output_ply} and {args.output_traj}")


if __name__ == "__main__":
    main()
