import glob
import os
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from loss_v3 import fusion_loss_mef
from rgb2ycbcr import RGB2YCrCb


TRAIN_PATH = "/data/ykx/MSRS/train"
PREVIEW_IMAGE_IDS = ["00326D", "00328D", "00917N", "01154N", "01185N"]
OUT_DIR = "/home/ykx/ca-fusion-loss/outputs/mlight_preview"


def _first_match(patterns: Iterable[str]) -> str | None:
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None


def _load_ir(path: str) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]


def _load_vis(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return arr  # [H,W,3] in [0,1]


def _vis_np_to_vis_y_tensor(vis_rgb: np.ndarray) -> torch.Tensor:
    # vis_rgb: [H, W, 3] -> tensor [1, 3, H, W]
    vis_tensor = torch.from_numpy(vis_rgb).permute(2, 0, 1).contiguous().unsqueeze(0)
    vis_ycrcb = RGB2YCrCb(vis_tensor)
    return vis_ycrcb[:, 0:1, :, :]  # [1,1,H,W]


def _to_rgb_mask(mask_1chw: torch.Tensor) -> np.ndarray:
    m = mask_1chw.detach().cpu().squeeze(0).squeeze(0).clamp(0, 1).numpy()
    return np.repeat(m[..., None], 3, axis=2)


def _save_compare(
    vis_rgb: np.ndarray,
    ir_1chw: torch.Tensor,
    mlight_1chw: torch.Tensor,
    out_path: str,
) -> None:
    left = (vis_rgb * 255.0).astype(np.uint8)
    ir_rgb = (_to_rgb_mask(ir_1chw) * 255.0).astype(np.uint8)
    mlight_rgb = (_to_rgb_mask(mlight_1chw) * 255.0).astype(np.uint8)
    canvas = np.concatenate([left, ir_rgb, mlight_rgb], axis=1)  # [H, 3W, 3]

    Image.fromarray(canvas, mode="RGB").save(out_path)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    loss_obj = fusion_loss_mef()
    loss_obj.eval()

    for sample_id in PREVIEW_IMAGE_IDS:
        ir_path = _first_match([os.path.join(TRAIN_PATH, "ir", f"{sample_id}.*")])
        vi_path = _first_match([os.path.join(TRAIN_PATH, "vi", f"{sample_id}.*")])
        if ir_path is None or vi_path is None:
            print(f"[skip] {sample_id}: file not found")
            continue

        ir = _load_ir(ir_path)
        vis = _load_vis(vi_path)
        vis_y = _vis_np_to_vis_y_tensor(vis)
        with torch.no_grad():
            m_light = loss_obj.get_M_light_mask(ir, vis_y)

        out_path = os.path.join(OUT_DIR, f"{sample_id}_vis_vs_mlight.png")
        _save_compare(vis, ir, m_light, out_path)
        print(f"[ok] {sample_id}: {out_path}")


if __name__ == "__main__":
    main()
