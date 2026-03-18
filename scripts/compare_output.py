#!/usr/bin/env python3
"""Compare C++ loger_infer output against a Python reference .pt file.

Usage:
    python scripts/compare_output.py \
        --ref reference/examples_office_0_50_1.pt \
        --traj trajectory.txt \
        --ply output.ply
"""
import argparse
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_ref_pt(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = v.float().numpy()
    return out


def load_traj_txt(path):
    """Load trajectory.txt -> (N, 7) array: tx ty tz qx qy qz qw."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            vals = list(map(float, line.split()))
            rows.append(vals[1:])  # drop timestamp
    return np.array(rows, dtype=np.float32)  # (N, 7)


def load_ply(path):
    """Return (N,3) float32 positions from a binary/ASCII PLY."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("utf-8", errors="replace").strip()
            header.append(line)
            if line == "end_header":
                break
        n_verts = next(int(h.split()[-1]) for h in header if h.startswith("element vertex"))
        props = [h.split() for h in header if h.startswith("property")]
        dtype_map = {"float": "f4", "uchar": "u1", "double": "f8", "int": "i4", "uint": "u4"}
        dt = np.dtype([(p[2], dtype_map.get(p[1], "f4")) for p in props])
        data = np.frombuffer(f.read(dt.itemsize * n_verts), dtype=dt)
    return np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------

def quat_to_mat(q):
    """(N,4) xyzw -> (N,3,3)."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(x*x + y*y + z*z + w*w + 1e-12)
    x, y, z, w = x/n, y/n, z/n, w/n
    R = np.stack([
        1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w),
        2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y),
    ], axis=-1).reshape(-1, 3, 3)
    return R


def poses_to_T(traj_rows):
    """(N,7) tx ty tz qx qy qz qw -> (N,4,4) Twc."""
    N = len(traj_rows)
    T = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)
    T[:, :3, 3] = traj_rows[:, :3]
    T[:, :3, :3] = quat_to_mat(traj_rows[:, 3:])
    return T


def align_umeyama(src, dst):
    """Umeyama alignment (scale=1): returns R,t such that dst ≈ R @ src + t."""
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    H = src_c.T @ dst_c / len(src)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = mu_d - R @ mu_s
    return R, t


def apply_align(T_src, R, t):
    """Apply rigid alignment to (N,4,4) poses."""
    T_align = np.eye(4, dtype=np.float32)
    T_align[:3, :3] = R
    T_align[:3, 3]  = t
    return T_align[None] @ T_src


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ate(T_est, T_ref):
    """Absolute Trajectory Error (translation) after Umeyama alignment."""
    t_est = T_est[:, :3, 3]
    t_ref = T_ref[:, :3, 3]
    R, t = align_umeyama(t_est, t_ref)
    t_est_al = (R @ t_est.T).T + t
    err = np.linalg.norm(t_est_al - t_ref, axis=1)
    return err, t_est_al, t_ref


def rpe_translation(T_est, T_ref):
    """Relative Pose Error (translation part)."""
    errs = []
    for i in range(len(T_est) - 1):
        dE = np.linalg.inv(T_est[i]) @ T_est[i+1]
        dR = np.linalg.inv(T_ref[i]) @ T_ref[i+1]
        diff = np.linalg.inv(dE) @ dR
        errs.append(np.linalg.norm(diff[:3, 3]))
    return np.array(errs)


def rpe_rotation_deg(T_est, T_ref):
    """Relative Pose Error (rotation, degrees)."""
    errs = []
    for i in range(len(T_est) - 1):
        dE = np.linalg.inv(T_est[i]) @ T_est[i+1]
        dR = np.linalg.inv(T_ref[i]) @ T_ref[i+1]
        diff = np.linalg.inv(dE) @ dR
        cos_angle = np.clip((np.trace(diff[:3, :3]) - 1) / 2, -1, 1)
        errs.append(np.degrees(np.arccos(cos_angle)))
    return np.array(errs)


# ---------------------------------------------------------------------------
# Point cloud overlap
# ---------------------------------------------------------------------------

def nearest_neighbor_dist(A, B, max_pts=50_000):
    """Mean nearest-neighbour distance from A to B (subsampled)."""
    from scipy.spatial import cKDTree
    if len(A) > max_pts:
        A = A[np.random.choice(len(A), max_pts, replace=False)]
    if len(B) > max_pts:
        B = B[np.random.choice(len(B), max_pts, replace=False)]
    tree = cKDTree(B)
    dists, _ = tree.query(A, workers=-1)
    return dists


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref",  default="reference/examples_office_0_50_1.pt")
    ap.add_argument("--traj", default="trajectory.txt")
    ap.add_argument("--ply",  default="output.ply")
    ap.add_argument("--no_ply", action="store_true", help="Skip point cloud comparison")
    args = ap.parse_args()

    print(f"Loading reference: {args.ref}")
    ref = load_ref_pt(args.ref)
    ref_poses = ref["camera_poses"]          # (N,4,4) Twc
    N_ref = len(ref_poses)

    print(f"Loading trajectory: {args.traj}")
    traj = load_traj_txt(args.traj)
    N_cpp = len(traj)
    T_cpp = poses_to_T(traj)                 # (N,4,4) Twc

    print(f"\nFrame count — reference: {N_ref}, C++: {N_cpp}")
    N = min(N_ref, N_cpp)
    if N_ref != N_cpp:
        print(f"  WARNING: counts differ, comparing first {N} frames")

    T_ref = ref_poses[:N]
    T_est = T_cpp[:N]

    # --- ATE ---
    ate_errs, t_est_al, t_ref = ate(T_est, T_ref)
    print(f"\n=== Absolute Trajectory Error (ATE) ===")
    print(f"  mean : {ate_errs.mean():.4f} m")
    print(f"  RMSE : {np.sqrt((ate_errs**2).mean()):.4f} m")
    print(f"  max  : {ate_errs.max():.4f} m  (frame {ate_errs.argmax()})")
    print(f"  min  : {ate_errs.min():.4f} m  (frame {ate_errs.argmin()})")

    # --- RPE ---
    rpe_t = rpe_translation(T_est, T_ref)
    rpe_r = rpe_rotation_deg(T_est, T_ref)
    print(f"\n=== Relative Pose Error (RPE, consecutive frames) ===")
    print(f"  translation mean : {rpe_t.mean():.4f} m  |  RMSE: {np.sqrt((rpe_t**2).mean()):.4f} m  |  max: {rpe_t.max():.4f} m")
    print(f"  rotation    mean : {rpe_r.mean():.3f}°  |  RMSE: {np.sqrt((rpe_r**2).mean()):.3f}°  |  max: {rpe_r.max():.3f}°")

    # Per-frame translation magnitude (to sanity-check scale)
    t_ref_norms = np.linalg.norm(T_ref[:, :3, 3], axis=1)
    t_est_norms = np.linalg.norm(T_est[:, :3, 3], axis=1)
    print(f"\n=== Translation magnitude (from origin) ===")
    print(f"  reference C++ median : {np.median(t_est_norms):.4f} m  |  ref median: {np.median(t_ref_norms):.4f} m")

    # --- Point cloud ---
    if not args.no_ply:
        try:
            from scipy.spatial import cKDTree
            print(f"\nLoading PLY: {args.ply}")
            pts_cpp = load_ply(args.ply)
            # Build reference point cloud from the .pt (flatten + conf filter)
            pts_ref_all = ref["points"].reshape(-1, 3)      # (N*H*W, 3)
            conf_ref    = ref["conf"].reshape(-1)            # (N*H*W,)
            mask = conf_ref > 0.2
            pts_ref_all = pts_ref_all[mask]

            print(f"  C++ PLY points    : {len(pts_cpp):,}")
            print(f"  Reference points  : {len(pts_ref_all):,}  (conf > 0.2)")

            dists = nearest_neighbor_dist(pts_cpp, pts_ref_all)
            print(f"\n=== Point cloud nearest-neighbour (C++ -> reference) ===")
            print(f"  mean  : {dists.mean():.4f}")
            print(f"  median: {np.median(dists):.4f}")
            print(f"  90th% : {np.percentile(dists, 90):.4f}")
            print(f"  max   : {dists.max():.4f}")
        except ImportError:
            print("\n[Point cloud comparison skipped — scipy not installed]")
        except Exception as e:
            print(f"\n[Point cloud comparison failed: {e}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
