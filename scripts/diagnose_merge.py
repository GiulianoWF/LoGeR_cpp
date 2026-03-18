#!/usr/bin/env python3
"""Diagnose merge quality: show per-window alignment errors."""
import argparse
import os
import numpy as np
from merge_windows import load_meta, load_window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows_dir", default="windows/")
    ap.add_argument("--max_windows", type=int, default=20)
    args = ap.parse_args()

    meta = load_meta(args.windows_dir)
    n = min(meta["n_windows"], args.max_windows)
    ov = meta["overlap_size"]
    print(f"{n} windows, overlap={ov}\n")

    prev_w = load_window(args.windows_dir, 0)
    for k in prev_w:
        prev_w[k] = prev_w[k].astype(np.float32)

    print(f"{'win':>4} | {'R angle (deg)':>13} | {'t diff':>8} | {'scale ratio':>11} | {'depth scale':>11}")
    print("-" * 70)

    for i in range(1, n):
        curr_w = load_window(args.windows_dir, i)
        for k in curr_w:
            curr_w[k] = curr_w[k].astype(np.float32)

        # Overlap poses
        prev_poses = prev_w["poses"][0, -ov:]   # (ov, 4, 4)
        curr_poses = curr_w["poses"][0, :ov]     # (ov, 4, 4)

        # Rotation difference per overlap frame
        angles = []
        t_diffs = []
        for j in range(ov):
            R_rel = prev_poses[j, :3, :3] @ curr_poses[j, :3, :3].T
            # Angle from rotation matrix
            cos_a = (np.trace(R_rel) - 1) / 2
            cos_a = np.clip(cos_a, -1, 1)
            angles.append(np.degrees(np.arccos(cos_a)))

            # Translation difference after applying R_rel
            t_prev = prev_poses[j, :3, 3]
            t_curr = curr_poses[j, :3, 3]
            t_aligned = R_rel @ t_curr + (t_prev - R_rel @ curr_poses[0, :3, 3]) + prev_poses[0, :3, 3] - t_prev
            t_diffs.append(np.linalg.norm(t_prev - (R_rel @ t_curr + t_prev - R_rel @ t_curr)))

        # Scale: depth ratio from local_points
        prev_lp = prev_w["local_points"][0, -ov:]  # (ov, H, W, 3)
        curr_lp = curr_w["local_points"][0, :ov]
        prev_z = prev_lp[..., 2].ravel()
        curr_z = curr_lp[..., 2].ravel()
        valid = (np.abs(curr_z) > 1e-6) & (np.abs(prev_z) > 1e-6) & np.isfinite(prev_z) & np.isfinite(curr_z)
        if valid.sum() > 100:
            ratios = prev_z[valid] / curr_z[valid]
            depth_scale = np.median(ratios)
        else:
            depth_scale = 1.0

        # Translation norm ratio (camera positions)
        prev_tnorm = np.linalg.norm(prev_poses[:, :3, 3], axis=-1).mean()
        curr_tnorm = np.linalg.norm(curr_poses[:, :3, 3], axis=-1).mean()
        tnorm_ratio = prev_tnorm / max(curr_tnorm, 1e-6)

        # Consistency: do all overlap frames agree on the same R?
        r_spread = np.std(angles)

        print(f"{i:4d} | {np.mean(angles):10.3f}±{r_spread:.1f} | {np.mean(t_diffs):8.4f} | "
              f"{tnorm_ratio:11.4f} | {depth_scale:11.4f}")

        prev_w = curr_w


if __name__ == "__main__":
    main()
