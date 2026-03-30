import glob
import os
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from loss_mef import fusion_loss_mef


TRAIN_PATH = "/data/ykx/MSRS/train"
PREVIEW_IMAGE_IDS = ["00326D", "00328D", "00917N", "01154N", "01185N"]
OUT_DIR = "/home/ykx/ca-fusion-loss/outputs/saliency_preview"


def _first_match(patterns: Iterable[str]) -> str | None:
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None


def _load_ir(path: str) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]


def _to_gray_3ch(arr_2d_01: np.ndarray) -> np.ndarray:
    return np.repeat(arr_2d_01[..., None], 3, axis=2)


def _save_compare(ir_1chw: torch.Tensor, saliency_1chw: torch.Tensor, out_path: str) -> None:
    ir = ir_1chw.detach().cpu().squeeze(0).squeeze(0).clamp(0, 1).numpy()
    s = saliency_1chw.detach().cpu().squeeze(0).squeeze(0).numpy()

    # Display-only normalization for saliency map
    s_min, s_max = float(s.min()), float(s.max())
    s_vis = (s - s_min) / (s_max - s_min + 1e-6)
    s_vis = np.clip(s_vis, 0, 1)

    left = (_to_gray_3ch(ir) * 255.0).astype(np.uint8)
    right = (_to_gray_3ch(s_vis) * 255.0).astype(np.uint8)
    canvas = np.concatenate([left, right], axis=1)  # [H, 2W, 3]
    Image.fromarray(canvas, mode="RGB").save(out_path)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    loss_obj = fusion_loss_mef()
    loss_obj.eval()

    for sample_id in PREVIEW_IMAGE_IDS:
        ir_path = _first_match([os.path.join(TRAIN_PATH, "ir", f"{sample_id}.*")])
        if ir_path is None:
            print(f"[skip] {sample_id}: IR file not found")
            continue

        ir = _load_ir(ir_path)
        with torch.no_grad():
            saliency, _ = loss_obj._compute_saliency(ir)

        out_path = os.path.join(OUT_DIR, f"{sample_id}_ir_vs_saliency.png")
        _save_compare(ir, saliency, out_path)
        print(f"[ok] {sample_id}: {out_path}")


if __name__ == "__main__":
    main()
