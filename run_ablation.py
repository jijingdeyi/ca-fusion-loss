"""
Paper term and quantile ablations.

  1) term on/off: (0,0), halo-only (1,0), washout-only (0,0.5)
     Full (1,0.5) is the default model (grid-h1-w0p5 / ours-best.pth).
  2) quantile: q in {0.85, 0.95} at (1.0, 0.5); q=0.90 is the default.

Each cell is trained, then scored on MSRS. Resume-safe: existing best
checkpoints and metrics.json files are skipped.
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
DATASETS = ["MSRS"]
METRICS = ["mi", "qabf", "scd", "vif", "ssim"]
SUMMARY_CSV = ROOT / "results" / "ablation_terms" / "summary.csv"
VIS_NAMES = ["00111D.png", "00931N.png"]
FULL_TAG = "grid-h1-w0p5"

CELLS = [
    # lambda_halo, lambda_bloom, q_bright, tag
    (0.0, 0.0, 0.90, "abl-h0-w0"),
    (1.0, 0.0, 0.90, "abl-h1-w0"),
    (0.0, 0.5, 0.90, "abl-h0-w0p5"),
    (1.0, 0.5, 0.85, "abl-q0p85"),
    (1.0, 0.5, 0.95, "abl-q0p95"),
]


def find_best_ckpt(tag: str) -> str | None:
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


def run(cmd: list[str], gpu: str) -> None:
    print(">>", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)


def train_cell(h: float, w: float, q: float, tag: str, gpu: str) -> str:
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
            "--q-bright", str(q),
            "--tag", tag,
        ],
        gpu,
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
        gpu,
    )


def write_summary() -> None:
    rows = []
    extra = [(1.0, 0.5, 0.90, FULL_TAG)]
    for h, w, q, tag in list(CELLS) + extra:
        p = metrics_path(tag)
        row = {
            "lambda_halo": h,
            "lambda_bloom": w,
            "q_bright": q,
            "tag": tag,
            "checkpoint": find_best_ckpt(tag) or "",
        }
        if p.is_file():
            data = json.loads(p.read_text())
            for name in DATASETS:
                m = (data.get("datasets") or {}).get(name) or {}
                for k in METRICS:
                    row[f"{name}_{k}"] = m.get(k)
        rows.append(row)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["lambda_halo", "lambda_bloom", "q_bright", "tag", "checkpoint"]
        + [f"{d}_{k}" for d in DATASETS for k in METRICS]
    )
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {SUMMARY_CSV}", flush=True)


def fuse_named_pairs(ckpt: str, tag: str, gpu: str) -> Path:
    outdir = ROOT / "results" / "MSRS" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    missing = [n for n in VIS_NAMES if not (outdir / n).is_file()]
    if not missing:
        return outdir
    run(
        [
            PY, "-u", "fuse_named.py",
            "--checkpoint", ckpt,
            "--outdir", str(outdir),
            "--gpu", gpu,
            "--names", *[n.replace(".png", "") for n in VIS_NAMES],
        ],
        gpu,
    )
    return outdir


def make_collages(gpu: str) -> None:
    run([PY, "-u", "make_term_ablation_figs.py"], gpu)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=GPU)
    args = parser.parse_args()
    gpu = args.gpu
    os.chdir(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    n = len(CELLS)
    t0 = time.time()
    for i, (h, w, q, tag) in enumerate(CELLS, 1):
        print(
            f"\n======== [{i}/{n}] {tag}  h={h} w={w} q={q} ========",
            flush=True,
        )
        ckpt = train_cell(h, w, q, tag, gpu)
        eval_cell(ckpt, tag, gpu)
        write_summary()
    for tag in ["abl-h0-w0", "abl-h1-w0", "abl-h0-w0p5", FULL_TAG]:
        ckpt = find_best_ckpt(tag)
        if ckpt:
            fuse_named_pairs(ckpt, tag, gpu)
    try:
        make_collages(gpu)
    except Exception as e:
        print(f"[warn] collage failed: {e}", flush=True)
    print(f"Ablation pipeline done in {(time.time() - t0) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    sys.exit(main())
