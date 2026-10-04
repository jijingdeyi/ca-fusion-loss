"""Fuse a few named IR-VIS pairs with one checkpoint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from config import TEST_PATH
from test_robust import fuse_batch, load_model, save_tensor_as_image

to_tensor = transforms.ToTensor()


def load_pair(root: Path, stem: str):
    ir_path = root / "ir" / f"{stem}.png"
    vi_path = root / "vi" / f"{stem}.png"
    if not ir_path.is_file() or not vi_path.is_file():
        raise FileNotFoundError(f"Missing pair for {stem} under {root}")
    ir = to_tensor(Image.open(ir_path).convert("L")).unsqueeze(0)
    vis = to_tensor(Image.open(vi_path).convert("RGB")).unsqueeze(0)
    return ir, vis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--ufuser-call", dest="ufuser_call", default="named")
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument(
        "--data-path",
        default=TEST_PATH,
        help="Folder with ir/ and vi/. Default: MSRS test set.",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, backbone="auto")
    root = Path(args.data_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for stem in args.names:
        ir, vis = load_pair(root, stem)
        fused = fuse_batch(model, ir, vis, device, ufuser_call=args.ufuser_call)
        save_tensor_as_image(fused[0], str(outdir / f"{stem}.png"))
        print("saved", outdir / f"{stem}.png")


if __name__ == "__main__":
    main()
