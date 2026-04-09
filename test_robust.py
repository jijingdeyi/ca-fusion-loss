from PIL import Image
import os
import argparse
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from Ufuser import Ufuser
from dataset import testloader
from rgb2ycbcr import RGB2YCrCb, YCrCb2RGB


warnings.filterwarnings('ignore')


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    加载与 train_robust.py 中一致的 Ufuser 模型。
    """
    model = Ufuser().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _ensure_vis_rgb(image_vis: torch.Tensor) -> torch.Tensor:
    """可见光为单通道（灰度）时复制为 3 通道，便于 RGB2YCrCb。"""
    if image_vis.dim() != 4:
        raise ValueError(f"image_vis 期望 [B,C,H,W]，当前 shape={tuple(image_vis.shape)}")
    c = image_vis.size(1)
    if c == 1:
        return image_vis.expand(-1, 3, -1, -1).contiguous()
    if c == 3:
        return image_vis
    raise ValueError(f"可见光通道数应为 1 或 3，当前 C={c}")


def _align_ir_vis_spatial(image_ir: torch.Tensor, image_vis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """IR / VIS 高宽不一致时，将可见光双线性缩放到与红外一致（与按对融合一致）。"""
    _, _, hi, wi = image_ir.shape
    _, _, hv, wv = image_vis.shape
    if (hi, wi) == (hv, wv):
        return image_ir, image_vis
    image_vis = F.interpolate(
        image_vis, size=(hi, wi), mode="bilinear", align_corners=False
    )
    return image_ir, image_vis


def _pad_hw_to_multiple(x: torch.Tensor, mult: int = 8) -> tuple[torch.Tensor, int, int]:
    """
    Ufuser 三次 stride-2 + 反卷积上采样时，若 H/W 非 mult 的倍数，解码端与 skip 会差 1 像素。
    返回 (pad 后的张量, 原始 H, 原始 W)，用于输出裁回。
    """
    _, _, h, w = x.shape
    pad_h = (-h) % mult
    pad_w = (-w) % mult
    if pad_h == 0 and pad_w == 0:
        return x, h, w
    # F.pad: (左, 右, 上, 下) 对应 W 与 H
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, h, w


def fuse_batch(model, image_ir, image_vis, device: torch.device):
    """
    参考 train_robust.py：使用 VIS 的 Y 通道 + IR 作为输入，输出融合后的 RGB 图像。

    兼容：可见光为灰度（1 通道）；IR/VI 分辨率略有差异；任意 H/W 通过 pad 到 8 的倍数避免解码与 skip 尺寸不一致。
    """
    image_vis = image_vis.to(device)
    image_ir = image_ir.to(device)

    image_vis = _ensure_vis_rgb(image_vis)
    image_ir, image_vis = _align_ir_vis_spatial(image_ir, image_vis)

    # RGB -> YCrCb
    image_vis_ycrcb = RGB2YCrCb(image_vis)
    image_vis_y = image_vis_ycrcb[:, 0:1, :, :]

    image_ir, h0, w0 = _pad_hw_to_multiple(image_ir, 8)
    image_vis_y, _, _ = _pad_hw_to_multiple(image_vis_y, 8)
    image_vis_ycrcb, _, _ = _pad_hw_to_multiple(image_vis_ycrcb, 8)

    with torch.no_grad():
        fused_y = model(image_vis_y, image_ir)  # [B,1,H',W']
        fused_y = fused_y[:, :, :h0, :w0]
        fused_y_clamped = fused_y.clamp(0, 1)

        cr = image_vis_ycrcb[:, 1:2, :h0, :w0]
        cb = image_vis_ycrcb[:, 2:3, :h0, :w0]
        fused_ycrcb = torch.cat([fused_y_clamped, cr, cb], dim=1)  # [B,3,h0,w0]
        fused_rgb = YCrCb2RGB(fused_ycrcb).clamp(0, 1)

    return fused_rgb


def save_tensor_as_image(tensor: torch.Tensor, save_path: str):
    """
    将 [3,H,W] 的张量保存为 PNG 图像，范围 [0,1] -> [0,255]。
    """
    tensor = tensor.detach().cpu().clamp(0, 1)
    np_img = tensor.numpy()
    np_img = (np_img * 255.0).round().astype(np.uint8)  # [3,H,W]
    np_img = np.transpose(np_img, (1, 2, 0))            # [H,W,3]
    img = Image.fromarray(np_img)
    img.save(save_path)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = args.checkpoint

    print(f'Loading model from: {checkpoint_path}')
    model = load_model(checkpoint_path, device)

    os.makedirs(args.outdir, mode=0o777, exist_ok=True)

    print('Start fusion on test set...')
    start_time = time.time()

    # 这里复用 dataset.testloader（MSRS 测试集），与 train_robust.py 保持一致
    total = len(testloader)
    ir_files = getattr(testloader.dataset, "files1", None)

    for idx, (image_ir, image_vis) in enumerate(testloader):
        fused_rgb = fuse_batch(model, image_ir, image_vis, device)  # [1,3,H,W]
        fused_img = fused_rgb[0]  # [3,H,W]

        # 使用 IR 路径的文件名保存（若可用），否则用索引命名
        if ir_files is not None and idx < len(ir_files):
            ir_path = ir_files[idx]
            basename = os.path.basename(ir_path)
        else:
            basename = f"{idx:06d}.png"
        save_path = os.path.join(args.outdir, basename)

        save_tensor_as_image(fused_img, save_path)

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f'[{idx+1}/{total}] saved: {save_path}')

    elapsed = time.time() - start_time
    print(f'Fusion finished! Total time: {elapsed:.2f}s')


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        default="./results/LLVIP/ours",
        type=str,
        nargs="?",
        help="dir to write fused results",
    )
    parser.add_argument(
        "--checkpoint",
        default="model/20260403-151334-0.670514-best.pth",
        type=str,
        nargs="?",
        help="checkpoint path",
    )

    args = parser.parse_args()
    main(args)


