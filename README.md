# Halo- and Washout-Aware Unsupervised IR–VIS Fusion

Code for the paper **Rethinking Unsupervised Fusion Objectives: Mitigating Degradation Bias in Infrared-Visible Image Fusion**.

Unsupervised IR–VIS fusion often uses a global max-intensity term. That term treats a locally stronger source response as content that should be copied, including degraded highlights. This repository keeps the classical intensity-and-gradient baseline and adds two regional L1 corrections:

- **Halo:** on pixels where the visible image is both bright (top quantile) and stronger than infrared, pull the fused intensity toward infrared.
- **Washout:** on pixels where infrared is both bright and stronger than visible, pull the fused intensity toward visible.

The default model is U-fuser (the EMMA backbone) trained on MSRS with \(\lambda_h=1\), \(\lambda_w=0.5\), \(q=0.90\). Validation uses Q\(_{abf}\) only.

## Setup

```bash
conda create -n ivif-loss python=3.10
conda activate ivif-loss
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set the dataset root:

```bash
cp .env.example .env
# then edit IVIF_DATA_ROOT
```

`config.py` reads `.env`. Alternatively export `IVIF_DATA_ROOT` in the shell.

## Data

This project does **not** redistribute IR–VIS images. Point `IVIF_DATA_ROOT` at a folder with:

```
$IVIF_DATA_ROOT/
  MSRS/train/{ir,vi}          # 1083 pairs; seed 42 split → 1051 train / 32 val
  MSRS/test/{ir,vi}
  MSRS/detection/{ir,vi}      # 80-image YOLO subset
  llvip_test/{ir,vi}          # 40-pair eval split used in the paper
  RoadScene/test/{ir,vi}
  FMB/test/{ir,vi}
  M3FD/M3FD_Fusion/{ir,vi}
  OpIVF/test/{ir,vi}
```

Each `{ir,vi}` pair folder must use matching filenames. Public sources: [MSRS](https://github.com/Linfeng-Tang/MSRS), [LLVIP](https://github.com/bupt-ai-cz/LLVIP), [RoadScene](https://github.com/hanna-xu/RoadScene), [FMB](https://github.com/JinyuanLiu-CV/SegMiF), [M3FD](https://github.com/JinyuanLiu-CV/TarDAL), [OpIVF](https://github.com/xieqj666/OpIVF).

## Weights

The paper default checkpoint is `model/ours-best.pth` (U-fuser, \(\lambda_h=1\), \(\lambda_w=0.5\), \(q=0.90\)). Other experimental weights are not included; retrain them with the scripts below.

## Quick start

Fuse the MSRS test set:

```bash
python test_robust.py \
  --checkpoint model/ours-best.pth \
  --data-path $IVIF_DATA_ROOT/MSRS/test \
  --outdir results/MSRS/ours \
  --gpu 0
```

Train the paper default (same as `python train_robust.py`):

```bash
python train_robust.py --gpu 0 --lambda_halo 1.0 --lambda_bloom 0.5 --q-bright 0.90
```

`--lambda_bloom` is the washout weight; the flag name is historical.

Score the paper datasets:

```bash
python eval_all_datasets.py \
  --checkpoint model/ours-best.pth \
  --run-name ours \
  --gpu 0
```

## Script index

### Training and inference

| File | What it is | Typical use |
|---|---|---|
| `loss_v3.py` | Paper loss. Global L1 to \(\max(I,V)\) + Sobel max-gradient + mask-mean halo/washout L1. Masks are detached, quantile \(q\) + cross-modal dominance. | Imported by `train_robust.py`. Not a CLI. |
| `train_robust.py` | Train U-fuser or MetaFusion. Adam, 200 epochs, ReduceLROnPlateau, early-stop on **val Q\(_{abf}\)** (patience 20). Saves `model/<stamp>-<tag>-<score>-best.pth`. | `python train_robust.py --gpu 0 --lambda_halo 1 --lambda_bloom 0.5 --q-bright 0.90 --tag ours` |
| `test_robust.py` | Fuse a folder that contains `ir/` and `vi/`. Backbone is inferred from checkpoint keys (`ufuser` / `metafusion`). | `python test_robust.py --checkpoint model/ours-best.pth --data-path /path/to/test --outdir results/demo --gpu 0` |
| `fuse_named.py` | Fuse a short list of stems (qualitative examples). | `python fuse_named.py --checkpoint model/ours-best.pth --outdir results/qual --names 00931N 00111D --gpu 0` |
| `fuse_detect_subset.py` | Fuse the 80-image MSRS detection subset. | `python fuse_detect_subset.py --checkpoint model/ours-best.pth --outdir results/MSRS_detect/ours --gpu 0` |

`--ufuser-call named` (default) is `model(IR, VIS_Y)`, which is the protocol used in the paper tables. `--ufuser-call paper` swaps the arguments.

### Evaluation and paper tables

| File | What it is | Typical use |
|---|---|---|
| `eval_all_datasets.py` | Fuse and score MI / Q\(_{abf}\) / SCD / VIF / SSIM on Y. Writes `results/<run-name>/metrics.json` and optional PNGs under `results/<dataset>/<run-name>/`. | `python eval_all_datasets.py --checkpoint model/ours-best.pth --run-name ours --datasets MSRS LLVIP RoadScene FMB M3FD OpIVF --gpu 0` |
| `metric.py` | MI, SCD, VIF, SSIM wrappers. | Imported only. |
| `Qabf.py` | Q\(_{abf}\) implementation. | Imported only. |
| `run_ablation.py` | Term on/off + \(q\in\{0.85,0.95\}\) pipeline. Resume-safe. Writes `results/ablation_terms/summary.csv`. | `python run_ablation.py --gpu 0` |
| `run_grid_sweep.py` | \(\lambda_h\times\lambda_w\) grid. Resume-safe. Writes `results/grid_sweep/summary_dense.csv`. | `python run_grid_sweep.py --gpu 0` |
| `make_term_ablation_figs.py` | Six-column IR / VIS / Base / Halo-only / Washout-only / Full collage with a zoom row. | `python make_term_ablation_figs.py --outdir results/term_ablation` |

YOLO detection is **not** bundled (it depends on a local YOLOv5 install). After `fuse_detect_subset.py`, run your detector on `results/MSRS_detect/ours` with the same 80 images / 193 labels, `yolov5s`, `conf=0.001`, `iou=0.6`.

### Models and data helpers

| File | What it is |
|---|---|
| `Ufuser.py` | Default backbone (U-fuser / EMMA). |
| `metafusion_net.py` | Optional MetaFusion backbone (`--backbone metafusion`). |
| `dataset.py` | IR/VIS loaders, MSRS train/val split (`VAL_RATIO=0.03`, `RANDOM_SEED=42`), 480 crop + flip. |
| `config.py` | Resolves `IVIF_DATA_ROOT`, train/test/detection paths, and the eval-set table. |
| `rgb2ycbcr.py` | RGB \(\leftrightarrow\) YCrCb. Fusion and metrics use Y; CrCb come from the visible image. |
| `logger.py` | File + stdout logging for training. |

## Training notes

- Crop size 480, batch size 4, seed `2026` in `train_robust.py`.
- Intensity/gradient weights are \(w_{l1}=w_{grad}=20\).
- Halo/washout terms are **fixed scalars**, not learned.
- Best checkpoint is selected by validation Q\(_{abf}\) only.

## What is not in this repo

Large fusion dumps (`results/`), TensorBoard logs, unpublished manuscript files, and extra experimental checkpoints are gitignored. Do not commit raw datasets.

## License

MIT. U-fuser / EMMA and MetaFusion backbones follow their original papers; this repository releases the loss, training/eval scripts, and the default weight trained with them.
