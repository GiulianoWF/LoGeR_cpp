#!/usr/bin/env python3
"""
Convert a LoGeR checkpoint so it is readable by the C++ libtorch torch::pickle_load().

Background
----------
torch::pickle_load() in libtorch 2.7+ uses PyTorchStreamReader (a zip reader) and
looks for the pickle data at the path  "data/data.pkl"  inside the archive.

torch.save(obj, 'latest.pt') creates a zip archive whose internal prefix equals the
filename stem, so 'latest.pt' produces 'latest/data.pkl' — NOT 'data/data.pkl'.
This causes torch::pickle_load to return None and crash with:
  "Expected GenericDict but got None"

The fix: save the checkpoint to a temporary file called 'data.pt' so the zip prefix
becomes 'data', then rename it to the desired output path.

Usage:
    python scripts/convert_checkpoint.py <input.pt> [output.pt]

If output.pt is omitted, the input file is overwritten.
"""
import sys
import os
import shutil
import tempfile
import torch


def convert(src: str, dst: str) -> None:
    print(f"Loading {src} ...")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        converted = {k: v for k, v in ckpt.items()}
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    # Save to a temp directory as 'data.pt' so that the zip archive uses the prefix
    # 'data', producing 'data/data.pkl' — exactly what torch::pickle_load expects.
    tmpdir = tempfile.mkdtemp()
    try:
        tmp_path = os.path.join(tmpdir, "data.pt")
        print(f"Saving {len(converted)} tensors → {dst}  (via tmp {tmp_path})")
        torch.save(converted, tmp_path)   # default zip format, prefix = 'data'
        shutil.move(tmp_path, dst)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    convert(src, dst)
