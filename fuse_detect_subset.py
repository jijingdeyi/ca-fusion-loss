"""Fuse the 80-image MSRS detection subset with one checkpoint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from config import DETECT_PATH
from test_robust import fuse_batch, load_model, save_tensor_as_image

to_tensor = transforms.ToTensor()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--data-root",
        default=DETECT_PATH,
        help="Folder with ir/ and vi/ (MSRS detection subset).",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, backbone="auto")
    det_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in (det_root / "ir").glob("*.png"))
    for i, stem in enumerate(names, 1):
        ir = to_tensor(Image.open(det_root / "ir" / f"{stem}.png").convert("L")).unsqueeze(0)
        vis = to_tensor(Image.open(det_root / "vi" / f"{stem}.png").convert("RGB")).unsqueeze(0)
        fused = fuse_batch(model, ir, vis, device, ufuser_call="named")
        save_tensor_as_image(fused[0], str(outdir / f"{stem}.png"))
        if i % 20 == 0 or i == len(names):
            print(f"[{i}/{len(names)}] {stem}.png")
    print("wrote", outdir, "n=", len(names))


if __name__ == "__main__":
    main()
