#!/usr/bin/env python3
"""Visualize loger_cpp output using viser.

Two modes:
  --pt output.pt          Per-frame mode (frame-by-frame point clouds + images).
                          Requires --output_pt from loger_infer.

  --ply output.ply        Flat mode (merged point cloud, camera axes only).
  --traj trajectory.txt
"""
import argparse
import os
import sys
import time
import numpy as np

# Allow importing from the parent LoGeR repo (for viser_wrapper)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOGER_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..'))
sys.path.insert(0, _LOGER_ROOT)


# ---------------------------------------------------------------------------
# Flat-mode helpers (--ply / --traj)
# ---------------------------------------------------------------------------

def load_ply(path, max_points=0):
    """Load a binary PLY, optionally subsampling *during* read to avoid OOM."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("utf-8", errors="replace").strip()
            header.append(line)
            if line == "end_header":
                break
        n_verts = next(int(h.split()[-1]) for h in header if h.startswith("element vertex"))
        has_color = any("red" in h for h in header)
        props = [h.split() for h in header if h.startswith("property")]
        dtype_map = {"float": "f4", "uchar": "u1", "double": "f8", "int": "i4", "uint": "u4"}
        dt = np.dtype([(p[2], dtype_map.get(p[1], "f4")) for p in props])
        data_offset = f.tell()

        if max_points > 0 and n_verts > max_points:
            # Subsample by reading only selected rows via stride, avoiding full read
            print(f"  Subsampling {max_points:,} / {n_verts:,} points ...")
            stride = n_verts / max_points
            indices = np.unique(np.floor(np.arange(max_points) * stride).astype(np.int64))
            # Memory-map the file for efficient random access
            mm = np.memmap(f, dtype=dt, mode='r', offset=data_offset, shape=(n_verts,))
            data = np.array(mm[indices])  # only copies selected rows
            del mm
        else:
            data = np.frombuffer(f.read(dt.itemsize * n_verts), dtype=dt)

    pts = np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float32)
    if has_color:
        rgb = np.stack([data["red"], data["green"], data["blue"]], axis=-1).astype(np.float32) / 255.0
    else:
        rgb = np.ones((len(pts), 3), dtype=np.float32) * 0.7
    return pts, rgb


def load_traj(path):
    poses = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            vals = list(map(float, line.split()))
            poses.append(vals[1:])
    return poses


def _patch_camera_controls(server):
    """Remove polar angle limits so rotation never hits a wall."""
    import viser._messages as _msg

    JS_PATCH = """
    (function patchPolar() {
        const canvas = document.querySelector('canvas');
        if (canvas && canvas.__r3f) {
            const controls = canvas.__r3f.store.getState().controls;
            if (controls) {
                controls.minPolarAngle = -Infinity;
                controls.maxPolarAngle = Infinity;
                return;
            }
        }
        // Controls may not be mounted yet — retry.
        setTimeout(patchPolar, 100);
    })();
    """

    @server.on_client_connect
    def _(client):
        client._websock_connection.queue_message(
            _msg.RunJavascriptMessage(source=JS_PATCH)
        )


def run_flat_mode(args):
    import viser
    print(f"Loading {args.ply} ...")
    pts, rgb = load_ply(args.ply, max_points=args.max_points)
    print(f"  Loaded {len(pts):,} points")

    poses = load_traj(args.traj)
    print(f"  {len(poses)} camera poses")

    server = viser.ViserServer(port=args.port)
    _patch_camera_controls(server)
    print(f"\nOpen http://localhost:{args.port} in your browser\n")

    server.scene.add_point_cloud("/points", points=pts, colors=rgb, point_size=0.005)

    cam_handles = []
    for i, (tx, ty, tz, qx, qy, qz, qw) in enumerate(poses):
        h = server.scene.add_frame(
            f"/cameras/{i:04d}",
            wxyz=np.array([qw, qx, qy, qz]),
            position=np.array([tx, ty, tz]),
            axes_length=0.05,
            axes_radius=0.003,
        )
        h.visible = (i == 0)
        cam_handles.append(h)

    n_frames = len(cam_handles)
    frame_slider = server.gui.add_slider("Frame", min=0, max=max(n_frames - 1, 0), step=1, initial_value=0)
    play_btn  = server.gui.add_button("Play")
    stop_btn  = server.gui.add_button("Stop")
    fps_input = server.gui.add_number("FPS", initial_value=10, min=1, max=60)
    playing = {"value": False}

    @frame_slider.on_update
    def _(_):
        idx = int(frame_slider.value)
        for j, h in enumerate(cam_handles):
            h.visible = (j == idx)

    @play_btn.on_click
    def _(_): playing["value"] = True

    @stop_btn.on_click
    def _(_): playing["value"] = False

    print("Press Ctrl-C to exit.")
    try:
        while True:
            if playing["value"] and n_frames > 0:
                frame_slider.value = (int(frame_slider.value) + 1) % n_frames
                time.sleep(1.0 / fps_input.value)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# Per-frame mode (--pt)
# ---------------------------------------------------------------------------

def load_pt(path):
    """Load per-frame predictions from either:
    - A Python dict .pt  (Python reference format)
    - A set of 4 tensor files written by the C++ binary ({base}_points.pt etc.)
    Returns a dict with numpy arrays: points, conf, images, camera_poses.
    """
    import torch

    base = path[:-3] if path.endswith(".pt") else path
    cpp_points = base + "_points.npy"

    if os.path.exists(cpp_points):
        # C++ output: 4 .npy files written by save_npy()
        pred = {
            "points":       torch.from_numpy(np.load(cpp_points)),
            "conf":         torch.from_numpy(np.load(base + "_conf.npy")),
            "images":       torch.from_numpy(np.load(base + "_images.npy")),
            "camera_poses": torch.from_numpy(np.load(base + "_poses.npy")),
        }
    else:
        # Python reference dict
        raw = torch.load(path, map_location="cpu", weights_only=False)
        pred = {k: v for k, v in raw.items() if torch.is_tensor(v)}

    return {k: v.float().numpy() for k, v in pred.items()}


def run_pt_mode(args):
    from loger.utils.viser_utils import viser_wrapper

    print(f"Loading {args.pt} ...")
    pred = load_pt(args.pt)
    print(f"  {pred['points'].shape[0]} frames, {pred['points'].shape[1]}x{pred['points'].shape[2]} px")

    viser_wrapper(
        pred,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        subsample=args.subsample,
    )


# ---------------------------------------------------------------------------
# Window merge mode (--windows_dir)
# ---------------------------------------------------------------------------

def run_windows_mode(args):
    """Interactive window-by-window merge viewer."""
    import viser
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from merge_windows import (load_meta, load_window, estimate_se3,
                               apply_se3, conf_postprocess)

    meta = load_meta(args.windows_dir)
    n_windows = meta["n_windows"]
    overlap_size = meta["overlap_size"]
    print(f"Loaded {n_windows} windows, overlap={overlap_size}")

    # Pre-load all windows
    print("Pre-loading windows...")
    raw_windows = []
    for i in range(n_windows):
        print(f"\r  Loading window {i+1}/{n_windows}", end="", flush=True)
        w = load_window(args.windows_dir, i)
        for k in w:
            w[k] = w[k].astype(np.float32)
        raw_windows.append(w)
    print(f"\n  Done ({n_windows} windows loaded)")

    # Cache: aligned_windows[i] = SE3-aligned version of window i
    aligned_cache = {}
    # aligned_poses_cache[i] = full aligned poses for window i (for next overlap ref)
    aligned_poses_cache = {}

    def get_aligned(idx):
        """Get SE3-aligned window, using cache."""
        if idx in aligned_cache:
            return aligned_cache[idx]
        w = raw_windows[idx]
        if idx == 0 or not args.se3:
            aligned_cache[idx] = w
            aligned_poses_cache[idx] = w["poses"]
        else:
            prev_poses = aligned_poses_cache[idx - 1]
            R, t = estimate_se3(prev_poses, w["poses"], overlap_size)
            aligned = apply_se3(w, R, t)
            # Carry images through (apply_se3 doesn't return them)
            if "images" in w:
                aligned["images"] = w["images"]
            aligned_cache[idx] = aligned
            aligned_poses_cache[idx] = aligned["poses"]
        return aligned_cache[idx]

    # Per-window cache: processed (pts, rgb, conf) arrays
    window_cache = {}

    def get_window_data(wi):
        """Get processed pts/rgb/conf for a single window (cached)."""
        if wi in window_cache:
            return window_cache[wi]
        from scipy.special import expit
        from scipy.ndimage import maximum_filter

        w = get_aligned(wi)
        skip = 0 if wi == 0 else overlap_size
        pts = w["points"][0, skip:]
        conf_raw = w["conf"][0, skip:]
        lpts = w["local_points"][0, skip:]

        z = lpts[..., 2]
        abs_diff = np.abs(maximum_filter(z, size=(1, 3, 3)) - z)
        edges = abs_diff > (0.02 * np.abs(z) + 1e-6)
        conf_sig = expit(conf_raw[..., 0])
        conf_sig[edges] = 0.0

        if "images" in w:
            rgb = w["images"][0, skip:]
        else:
            rgb = np.ones_like(pts) * 0.7

        result = (pts.reshape(-1, 3), rgb.reshape(-1, 3), conf_sig.ravel())
        window_cache[wi] = result
        return result

    def build_merged(up_to):
        """Merge windows 0..up_to into flat arrays for display."""
        all_pts, all_rgb, all_conf = [], [], []
        for wi in range(up_to + 1):
            pts, rgb, conf = get_window_data(wi)
            all_pts.append(pts)
            all_rgb.append(rgb)
            all_conf.append(conf)

        pts = np.concatenate(all_pts)
        rgb = np.concatenate(all_rgb)
        conf = np.concatenate(all_conf)

        mask = conf > args.conf_threshold
        pts, rgb = pts[mask], rgb[mask]

        if len(pts) > args.max_points:
            idx = np.linspace(0, len(pts) - 1, args.max_points, dtype=int)
            pts, rgb = pts[idx], rgb[idx]

        return pts, np.clip(rgb, 0, 1)

    # --- viser UI ---
    server = viser.ViserServer(port=args.port)
    _patch_camera_controls(server)
    print(f"\nOpen http://localhost:{args.port} in your browser\n")

    state = {"current": 0, "updating": False}

    status_md = server.gui.add_markdown("*Starting...*", order=0)
    window_slider = server.gui.add_slider(
        "Windows", min=1, max=n_windows, step=1, initial_value=1, order=1)
    step_btns = server.gui.add_button_group(
        "Step", options=["<< -10", "< Prev", "Next >", "+10 >>"], order=2)

    def update_scene(up_to):
        if state["updating"]:
            return
        state["updating"] = True
        up_to = max(0, min(up_to, n_windows - 1))
        state["current"] = up_to
        status_md.content = f"**Merging {up_to + 1}/{n_windows}...**"

        pts, rgb = build_merged(up_to)
        server.scene.add_point_cloud(
            "/points", points=pts.astype(np.float32),
            colors=rgb.astype(np.float32), point_size=0.005)

        status_md.content = (f"**{up_to + 1} / {n_windows} windows** — "
                             f"{len(pts):,} points")
        window_slider.value = up_to + 1
        state["updating"] = False

    @window_slider.on_update
    def _(_):
        update_scene(int(window_slider.value) - 1)

    @step_btns.on_click
    def _(event):
        v = event.target.value
        if v == "< Prev":
            update_scene(state["current"] - 1)
        elif v == "Next >":
            update_scene(state["current"] + 1)
        elif v == "<< -10":
            update_scene(state["current"] - 10)
        elif v == "+10 >>":
            update_scene(state["current"] + 10)

    # Initial display: first window
    update_scene(0)

    print("Use the slider or +1/-1 buttons in the browser. Ctrl-C to exit.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    # Per-frame mode
    ap.add_argument("--pt",   default=None, help="Per-frame .pt from loger_infer --output_pt")
    # Flat mode
    ap.add_argument("--ply",  default="output.ply")
    ap.add_argument("--traj", default="trajectory.txt")
    ap.add_argument("--max_points", type=int, default=500_000)
    # Window merge mode
    ap.add_argument("--windows_dir", default=None,
                    help="Directory from --save_windows for interactive merge viewer")
    ap.add_argument("--se3", action="store_true", default=True,
                    help="Enable SE3 alignment (default)")
    ap.add_argument("--no_se3", dest="se3", action="store_false",
                    help="Disable SE3 alignment")
    # Shared
    ap.add_argument("--port",           type=int,   default=8080)
    ap.add_argument("--conf_threshold", type=float, default=0.1)
    ap.add_argument("--subsample",      type=int,   default=2)
    args = ap.parse_args()

    if args.pt:
        run_pt_mode(args)
    elif args.windows_dir:
        run_windows_mode(args)
    else:
        run_flat_mode(args)


if __name__ == "__main__":
    main()
