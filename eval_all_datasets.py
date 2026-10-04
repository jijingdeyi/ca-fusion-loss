"""Fuse and score the paper transfer datasets with one MSRS-trained checkpoint."""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import EVAL_DATASETS
from dataset import Hinet_Dataset, val_transform
from metric import (
    MI_function,
    Qabf_function,
    SCD_function,
    VIF_function,
    SSIM_function,
)
from rgb2ycbcr import RGB2YCrCb
from test_robust import (
    _align_ir_vis_spatial,
    _ensure_vis_rgb,
    fuse_batch,
    load_model,
    save_tensor_as_image,
)

DATASETS = [(name, str(path)) for name, path in EVAL_DATASETS]


def _to_u8(gray: torch.Tensor) -> np.ndarray:
    return (gray.detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.float32)


def eval_one(model, data_path, outdir, device, ufuser_call, save_images=True):
    os.makedirs(outdir, exist_ok=True)
    loader = DataLoader(
        Hinet_Dataset(transforms_=val_transform, data_path=data_path),
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    ir_files = loader.dataset.files1
    totals = {k: 0.0 for k in ("mi", "qabf", "scd", "vif", "ssim")}
    n = 0
    t0 = time.time()
    for idx, (image_ir, image_vis) in enumerate(loader):
        fused_rgb = fuse_batch(model, image_ir, image_vis, device, ufuser_call=ufuser_call)
        image_vis = image_vis.to(device)
        image_ir = image_ir.to(device)
        image_vis = _ensure_vis_rgb(image_vis)
        image_ir, image_vis = _align_ir_vis_spatial(image_ir, image_vis)
        vis_y = RGB2YCrCb(image_vis)[:, 0:1]
        fused_y = RGB2YCrCb(fused_rgb)[:, 0:1]

        ir_np = _to_u8(image_ir[0, 0])
        vis_np = _to_u8(vis_y[0, 0])
        fused_np = _to_u8(fused_y[0, 0])

        totals["mi"] += MI_function(ir_np, vis_np, fused_np)
        totals["qabf"] += Qabf_function(ir_np, vis_np, fused_np)
        totals["scd"] += SCD_function(ir_np, vis_np, fused_np)
        totals["vif"] += VIF_function(ir_np, vis_np, fused_np)
        totals["ssim"] += SSIM_function(ir_np, vis_np, fused_np)
        n += 1

        if save_images:
            basename = os.path.basename(ir_files[idx]) if idx < len(ir_files) else f"{idx:06d}.png"
            stem, ext = os.path.splitext(basename)
            save_path = os.path.join(outdir, stem + (ext if ext else ".png"))
            save_tensor_as_image(fused_rgb[0], save_path)
            if (idx + 1) % 50 == 0 or (idx + 1) == len(loader):
                print(f"  [{idx+1}/{len(loader)}] {save_path}")
        elif (idx + 1) % 50 == 0 or (idx + 1) == len(loader):
            print(f"  [{idx+1}/{len(loader)}]")
    means = {k: v / max(n, 1) for k, v in totals.items()}
    means["n"] = n
    means["seconds"] = time.time() - t0
    return means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", default="ours")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--ufuser-call", dest="ufuser_call", default="named",
                        choices=["named", "paper"])
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--skip-save", action="store_true",
                        help="Compute metrics only; do not write fused PNGs.")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.checkpoint}")
    model = load_model(args.checkpoint, device, backbone="auto")

    wanted = set(args.datasets) if args.datasets else None
    summary = {"checkpoint": args.checkpoint, "run_name": args.run_name, "datasets": {}}
    for name, path in DATASETS:
        if wanted is not None and name not in wanted:
            continue
        if not os.path.isdir(os.path.join(path, "ir")):
            print(f"[skip] {name}: missing {path}/ir")
            continue
        outdir = os.path.join("results", name, args.run_name)
        print(f"==== {name}  {path} -> {outdir} ====")
        means = eval_one(
            model, path, outdir, device, args.ufuser_call, save_images=not args.skip_save
        )
        summary["datasets"][name] = means
        print(
            f"{name}: n={means['n']}  "
            f"MI={means['mi']:.3f}  Qabf={means['qabf']:.3f}  "
            f"SCD={means['scd']:.3f}  VIF={means['vif']:.3f}  "
            f"SSIM={means['ssim']:.3f}  ({means['seconds']:.1f}s)"
        )

    os.makedirs(os.path.join("results", args.run_name), exist_ok=True)
    out_json = os.path.join("results", args.run_name, "metrics.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote", out_json)
    print("\n{:<12} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "Dataset", "n", "MI", "Qabf", "SCD", "VIF", "SSIM"))
    for name, m in summary["datasets"].items():
        print("{:<12} {:5d} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f}".format(
            name, int(m["n"]), m["mi"], m["qabf"], m["scd"], m["vif"], m["ssim"]))


if __name__ == "__main__":
    main()
