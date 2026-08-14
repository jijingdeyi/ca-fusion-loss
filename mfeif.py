import argparse
from pathlib import Path

import cv2
import kornia
import torch
import torch.nn as nn
from kornia.filters import SpatialGradient
from torch import Tensor
from tqdm import tqdm


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 3, s: int = 1, p: int = 0, d: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (k, k), (s, s), (p, p), (d, d))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = ConvBlock(1, 16, p=1)
        self.conv_a1 = ConvBlock(16, 32, p=1)
        self.conv_a2 = ConvBlock(32, 48, p=1)
        self.conv_a3 = ConvBlock(48, 64, p=1)
        self.conv_b1 = ConvBlock(16, 32, p=2, d=2)
        self.conv_b2 = ConvBlock(32, 48, p=1)
        self.conv_b3 = ConvBlock(48, 64, p=1)
        self.conv_c1 = ConvBlock(16, 32, p=3, d=3)
        self.conv_c2 = ConvBlock(32, 48, p=1)
        self.conv_c3 = ConvBlock(48, 64, p=1)

    def forward(self, x: Tensor):
        x = self.conv_1(x)
        a1 = self.conv_a1(x)
        a2 = self.conv_a2(a1)
        a3 = self.conv_a3(a2)
        b1 = self.conv_b1(x)
        b2 = self.conv_b2(b1)
        b3 = self.conv_b3(b2)
        c1 = self.conv_c1(x)
        c2 = self.conv_c2(c1)
        c3 = self.conv_c3(c2)
        f = 0.1 * a3 + 0.1 * b3 + 1.0 * c3
        b_2 = a1 + b1 + c1
        b_1 = a2 + b2 + c2
        return f, b_1, b_2


class EdgeDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = SpatialGradient("diff")
        self.max_pool = nn.MaxPool2d(3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        s = self.spatial(x)
        dx, dy = s[:, :, 0, :, :], s[:, :, 1, :, :]
        u = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
        y = self.max_pool(u)
        return y


class ConvConv(nn.Module):
    def __init__(self, a_channels: int, b_channels: int, c_channels: int):
        super().__init__()
        self.conv_1 = nn.Conv2d(a_channels, b_channels, (3, 3), padding=(1, 1))
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(b_channels, c_channels, (3, 3), padding=(2, 2), dilation=(2, 2))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        return x


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = ConvConv(1, 32, 32)
        self.conv_2 = ConvConv(32, 64, 128)
        self.conv_3 = ConvConv(128, 64, 32)
        self.conv_4 = nn.Conv2d(32, 1, (1, 1))
        self.ed = EdgeDetect()

    def forward(self, x: Tensor) -> Tensor:
        e = self.ed(x)
        x = x + e
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        return x


class FeatherFuse(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ir_b: tuple[Tensor, Tensor], vi_b: tuple[Tensor, Tensor], mode: str = "min-mean") -> tuple[Tensor, Tensor]:
        b_1 = torch.min(ir_b[0], vi_b[0])
        b_2 = torch.min(ir_b[1], vi_b[1])
        b_3 = (ir_b[0] + vi_b[0] + b_1) / 3
        b_4 = (ir_b[1] + vi_b[1] + b_2) / 3
        return (b_1, b_2) if mode == "min" else (b_3, b_4)


class Constructor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = ConvBlock(64, 16, p=1)
        self.conv_2 = ConvBlock(64, 32, p=1)
        self.conv_3 = ConvBlock(64, 16, p=1)
        self.conv_4 = ConvBlock(16, 1, p=1)

    def forward(self, x: Tensor, b_1: Tensor, b_2: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = torch.cat([x, b_1], dim=1)
        x = self.conv_2(x)
        x = torch.cat([x, b_2], dim=1)
        x = self.conv_3(x)
        x = self.conv_4(x)
        return x


class MFEIFFuser:
    """All-in-one MFEIF runner for folder-based fusion."""

    def __init__(self, model_path: str, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        params = torch.load(model_path, map_location="cpu")
        self.net_ext = Extractor().to(self.device).eval()
        self.net_con = Constructor().to(self.device).eval()
        self.net_att = Attention().to(self.device).eval()
        self.net_ext.load_state_dict(params["ext"])
        self.net_con.load_state_dict(params["con"])
        self.net_att.load_state_dict(params["att"])

        self.softmax = nn.Softmax(dim=1)
        self.feather_fuse = FeatherFuse()

    @torch.no_grad()
    def forward(self, ir: Tensor, vi: Tensor) -> Tensor:
        ir_1, ir_b_1, ir_b_2 = self.net_ext(ir)
        vi_1, vi_b_1, vi_b_2 = self.net_ext(vi)

        ir_att = self.net_att(ir)
        vi_att = self.net_att(vi)

        fus_1 = ir_1 * ir_att + vi_1 * vi_att
        fus_1 = self.softmax(fus_1)
        fus_b_1, fus_b_2 = self.feather_fuse((ir_b_1, ir_b_2), (vi_b_1, vi_b_2))
        fus_2 = self.net_con(fus_1, fus_b_1, fus_b_2)
        return fus_2

    @staticmethod
    def _imread_gray(path: str) -> Tensor:
        im_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im_cv is None:
            raise ValueError(f"Failed to read image: {path}")
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.0).float()
        return im_ts

    @staticmethod
    def _imsave(path: Path, image: Tensor, vi_path: str, color_output: bool):
        path.parent.mkdir(parents=True, exist_ok=True)
        im_ts = image.squeeze().cpu()
        fu_gray = kornia.utils.tensor_to_image(im_ts)
        fu_min = fu_gray.min()
        fu_max = fu_gray.max()
        if fu_max > fu_min:
            fu_gray = (fu_gray - fu_min) / (fu_max - fu_min)
        else:
            fu_gray = fu_gray * 0
        fu_gray = (fu_gray * 255.0).clip(0, 255).astype("uint8")

        if not color_output:
            cv2.imwrite(str(path), fu_gray)
            return

        vi_color = cv2.imread(vi_path, cv2.IMREAD_COLOR)
        if vi_color is None:
            raise ValueError(f"Failed to read visible image: {vi_path}")
        if fu_gray.shape != vi_color.shape[:2]:
            fu_gray = cv2.resize(fu_gray, (vi_color.shape[1], vi_color.shape[0]), interpolation=cv2.INTER_LINEAR)
        vi_yuv = cv2.cvtColor(vi_color, cv2.COLOR_BGR2YUV)
        vi_yuv[:, :, 0] = fu_gray
        fu_color = cv2.cvtColor(vi_yuv, cv2.COLOR_YUV2BGR)
        cv2.imwrite(str(path), fu_color.clip(0, 255).astype("uint8"))

    def fuse_folders(self, ir_dir: str, vi_dir: str, out_dir: str, color_output: bool = True):
        ir_paths = sorted([p for p in Path(ir_dir).glob("*") if p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"}])
        vi_paths = sorted([p for p in Path(vi_dir).glob("*") if p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"}])

        if len(ir_paths) != len(vi_paths):
            raise ValueError(f"IR/VI file count mismatch: {len(ir_paths)} vs {len(vi_paths)}")

        for ir_path, vi_path in tqdm(list(zip(ir_paths, vi_paths)), desc="Fusing"):
            if ir_path.stem != vi_path.stem:
                raise ValueError(f"Pair name mismatch: {ir_path.name} vs {vi_path.name}")
            ir = self._imread_gray(str(ir_path)).unsqueeze(0).to(self.device)
            vi = self._imread_gray(str(vi_path)).unsqueeze(0).to(self.device)
            fu = self.forward(ir, vi)
            self._imsave(Path(out_dir) / ir_path.name, fu, str(vi_path), color_output=color_output)


def main():
    parser = argparse.ArgumentParser(description="Single-file MFEIF fusion runner")
    parser.add_argument("--model_path", required=True, help="Path to MFEIF checkpoint (.pth)")
    parser.add_argument("--ir_dir", required=True, help="Infrared image folder")
    parser.add_argument("--vi_dir", required=True, help="Visible image folder")
    parser.add_argument("--out_dir", required=True, help="Output folder")
    parser.add_argument("--device", default=None, help='Device like "cuda" or "cpu"')
    parser.add_argument("--gray_output", action="store_true", help="Save grayscale fusion only")
    args = parser.parse_args()

    runner = MFEIFFuser(args.model_path, args.device)
    runner.fuse_folders(args.ir_dir, args.vi_dir, args.out_dir, color_output=not args.gray_output)


if __name__ == "__main__":
    main()
