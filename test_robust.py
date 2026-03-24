from PIL import Image
import os
import argparse
import time
import warnings

import numpy as np
import torch

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


def fuse_batch(model, image_ir, image_vis, device: torch.device):
    """
    参考 train_robust.py：使用 VIS 的 Y 通道 + IR 作为输入，输出融合后的 RGB 图像。
    """
    image_vis = image_vis.to(device)
    image_ir = image_ir.to(device)

    # RGB -> YCrCb
    image_vis_ycrcb = RGB2YCrCb(image_vis)
    image_vis_y = image_vis_ycrcb[:, 0:1, :, :]

    with torch.no_grad():
        fused_y = model(image_vis_y, image_ir)          # [B,1,H,W]
        fused_y_clamped = fused_y.clamp(0, 1)

        # 用原图的 Cr/Cb 通道恢复彩色
        fused_ycrcb = torch.cat(
            [fused_y_clamped, image_vis_ycrcb[:, 1:2, :, :], image_vis_ycrcb[:, 2:3, :, :]],
            dim=1
        )  # [B,3,H,W]
        fused_rgb = YCrCb2RGB(fused_ycrcb).clamp(0, 1)  # [B,3,H,W]

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
        default="./results/ori+halo",
        type=str,
        nargs="?",
        help="dir to write fused results",
    )
    parser.add_argument(
        "--checkpoint",
        default="model/A2RNet-20260323-145246-best.pth",
        type=str,
        nargs="?",
        help="checkpoint path",
    )

    args = parser.parse_args()
    main(args)


