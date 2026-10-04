"""
Lambda grid: train each (lambda_halo, lambda_bloom) cell with Qabf early-stop,
then score MSRS, LLVIP, RoadScene, FMB, and M3FD. Resume-safe.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from config import ROOT

PY = sys.executable
GPU = os.environ.get("IVIF_GPU", "0")

HALO = [0.75, 1.0, 1.25, 1.5]
WASH = [0.5, 0.75, 1.0]
DATASETS = ["MSRS", "LLVIP", "RoadScene", "FMB", "M3FD"]
METRICS = ["mi", "qabf", "scd", "vif", "ssim"]
SUMMARY_CSV = ROOT / "results" / "grid_sweep" / "summary_dense.csv"


def fmt_lam(x: float) -> str:
    s = f"{x:g}"
    return s.replace(".", "p")


def cell_tag(h: float, w: float) -> str:
    return f"grid-h{fmt_lam(h)}-w{fmt_lam(w)}"


def find_best_ckpt(tag: str) -> str | None:
    # Hyphen after the tag so grid-h0p5-w1 does not match grid-h0p5-w10.
    hits = sorted(
        glob.glob(str(ROOT / "model" / f"*-{tag}-*-best.pth")),
        key=os.path.getmtime,
    )
    return hits[-1] if hits else None


def metrics_path(tag: str) -> Path:
    return ROOT / "results" / tag / "metrics.json"


def metrics_complete(tag: str) -> bool:
    p = metrics_path(tag)
    if not p.is_file():
        return False
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return False
    ds = data.get("datasets") or {}
    return all(name in ds for name in DATASETS)


def run(cmd: list[str], env: dict | None = None) -> None:
    print(">>", " ".join(cmd), flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.check_call(cmd, cwd=str(ROOT), env=merged)


def train_cell(h: float, w: float, tag: str, gpu: str) -> str:
    ckpt = find_best_ckpt(tag)
    if ckpt:
        print(f"[skip train] {tag} -> {ckpt}", flush=True)
        return ckpt
    run(
        [
            PY, "-u", "train_robust.py",
            "--gpu", gpu,
            "--lambda_halo", str(h),
            "--lambda_bloom", str(w),
            "--tag", tag,
        ],
        env={"CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"},
    )
    ckpt = find_best_ckpt(tag)
    if not ckpt:
        raise FileNotFoundError(f"No best checkpoint after training {tag}")
    return ckpt


def eval_cell(ckpt: str, tag: str, gpu: str) -> None:
    if metrics_complete(tag):
        print(f"[skip eval] {tag}", flush=True)
        return
    run(
        [
            PY, "-u", "eval_all_datasets.py",
            "--checkpoint", ckpt,
            "--run-name", tag,
            "--gpu", gpu,
            "--skip-save",
            "--datasets", *DATASETS,
        ],
        env={"CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"},
    )


def write_summary(out_csv: Path) -> None:
    rows = []
    for h in HALO:
        for w in WASH:
            tag = cell_tag(h, w)
            p = metrics_path(tag)
            row = {
                "lambda_halo": h,
                "lambda_bloom": w,
                "tag": tag,
                "checkpoint": find_best_ckpt(tag) or "",
            }
            if p.is_file():
                data = json.loads(p.read_text())
                hp_hits = sorted(
                    glob.glob(str(ROOT / "model" / f"*-{tag}-*-best-hparams.json"))
                )
                if hp_hits:
                    hp = json.loads(Path(hp_hits[-1]).read_text())
                    row["val_qabf"] = hp.get("val_qabf", hp.get("val_score"))
                    row["val_epoch"] = hp.get("epoch")
                for name in DATASETS:
                    m = (data.get("datasets") or {}).get(name) or {}
                    for k in METRICS:
                        row[f"{name}_{k}"] = m.get(k)
            rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["lambda_halo", "lambda_bloom", "tag", "checkpoint", "val_qabf", "val_epoch"]
        + [f"{d}_{k}" for d in DATASETS for k in METRICS]
    )
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=GPU)
    args = parser.parse_args()
    gpu = args.gpu
    os.chdir(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    n = len(HALO) * len(WASH)
    print(
        f"Grid {len(HALO)}x{len(WASH)}={n}  halo={HALO}  washout={WASH}  GPU={gpu}",
        flush=True,
    )
    t0 = time.time()
    for i, h in enumerate(HALO):
        for j, w in enumerate(WASH):
            k = i * len(WASH) + j + 1
            tag = cell_tag(h, w)
            print(f"\n======== [{k}/{n}] {tag}  lambda_h={h} lambda_w={w} ========", flush=True)
            ckpt = train_cell(h, w, tag, gpu)
            eval_cell(ckpt, tag, gpu)
            write_summary(SUMMARY_CSV)
    print(f"All {n} cells done in {(time.time() - t0) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    sys.exit(main())
