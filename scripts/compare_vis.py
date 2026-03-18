#!/usr/bin/env python3
"""Side-by-side per-frame visualizer: C++ output vs Python reference .pt.

Spawns two viser servers using viser_wrapper (frame-by-frame, images, frustums):
  - http://localhost:8080  ← C++ output   (--pt / --ply+traj fallback)
  - http://localhost:8081  ← Python reference (--ref)

Usage:
    # With per-frame C++ output (recommended):
    python scripts/compare_vis.py --pt output.pt

    # Flat fallback (no per-frame images):
    python scripts/compare_vis.py --ply output.ply --traj trajectory.txt
"""
import argparse
import os
import sys
import subprocess

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOGER_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..'))
sys.path.insert(0, _LOGER_ROOT)

PYTHON = sys.executable
VIS    = os.path.join(_SCRIPT_DIR, "visualize.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref",            default="reference/examples_office_0_50_1.pt")
    # C++ output — prefer --pt; falls back to --ply/--traj
    ap.add_argument("--pt",             default=None,          help="Per-frame .pt from loger_infer --output_pt")
    ap.add_argument("--ply",            default="output.ply")
    ap.add_argument("--traj",           default="trajectory.txt")
    ap.add_argument("--cpp_port",  type=int, default=8080)
    ap.add_argument("--ref_port",  type=int, default=8081)
    ap.add_argument("--conf",      type=float, default=20.0)
    ap.add_argument("--subsample", type=int,   default=2)
    args = ap.parse_args()

    # --- C++ side ---
    if args.pt:
        cpp_cmd = [PYTHON, VIS, "--pt", args.pt,
                   "--port", str(args.cpp_port),
                   "--conf_threshold", str(args.conf),
                   "--subsample", str(args.subsample)]
        cpp_label = f"C++ per-frame  → http://localhost:{args.cpp_port}"
    else:
        cpp_cmd = [PYTHON, VIS,
                   "--ply", args.ply, "--traj", args.traj,
                   "--port", str(args.cpp_port)]
        cpp_label = f"C++ flat       → http://localhost:{args.cpp_port}"

    # --- Reference side ---
    ref_cmd = [PYTHON, VIS, "--pt", args.ref,
               "--port", str(args.ref_port),
               "--conf_threshold", str(args.conf),
               "--subsample", str(args.subsample)]
    ref_label = f"Python ref     → http://localhost:{args.ref_port}"

    print(f"Starting {cpp_label}")
    proc_cpp = subprocess.Popen(cpp_cmd)

    print(f"Starting {ref_label}")
    proc_ref = subprocess.Popen(ref_cmd)

    print(f"""
Open both in your browser:
  {cpp_label}
  {ref_label}

Press Ctrl-C to stop both servers.
""")

    try:
        proc_cpp.wait()
        proc_ref.wait()
    except KeyboardInterrupt:
        print("\nStopping servers...")
        proc_cpp.terminate()
        proc_ref.terminate()


if __name__ == "__main__":
    main()
