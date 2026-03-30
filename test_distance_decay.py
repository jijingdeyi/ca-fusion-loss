import os
import numpy as np
import torch
from PIL import Image

from loss_mef import fusion_loss_mef


def make_fake_mlight(h=256, w=320):
    """
    Build a fake M_light map with three circular bright spots.
    Returns tensor shape: [1, 1, H, W], values in [0, 1].
    """
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )

    circles = [
        (80.0, 90.0, 5),
        (140.0, 210.0, 6),
        (190.0, 130.0, 8),
    ]

    m = torch.zeros((h, w), dtype=torch.float32)
    for cy, cx, r in circles:
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)
        m = torch.maximum(m, mask.float())

    return m.unsqueeze(0).unsqueeze(0)


def save_gray_image(tensor_1chw, path):
    """
    Save [1, 1, H, W] tensor to grayscale PNG.
    """
    x = tensor_1chw.detach().cpu().clamp(0, 1).squeeze(0).squeeze(0).numpy()
    x = (x * 255.0).astype(np.uint8)
    Image.fromarray(x, mode="L").save(path)


def save_side_by_side_tensors(left_tensor_1chw, right_tensor_1chw, out_path):
    left_np = left_tensor_1chw.detach().cpu().clamp(0, 1).squeeze(0).squeeze(0).numpy()
    right_np = right_tensor_1chw.detach().cpu().clamp(0, 1).squeeze(0).squeeze(0).numpy()

    left = Image.fromarray((left_np * 255.0).astype(np.uint8), mode="L")
    right = Image.fromarray((right_np * 255.0).astype(np.uint8), mode="L")

    if left.height != right.height:
        right = right.resize((right.width, left.height), resample=Image.Resampling.BILINEAR)

    canvas = Image.new("L", (left.width + right.width, left.height))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    canvas.save(out_path)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    m_light = make_fake_mlight()

    loss_obj = fusion_loss_mef()
    with torch.no_grad():
        decay = loss_obj.build_distance_decay(m_light)

    compare_path = os.path.join(out_dir, "mlight_vs_decay.png")

    save_side_by_side_tensors(m_light, decay, compare_path)

    print(f"Saved compare image: {compare_path}")


if __name__ == "__main__":
    main()
