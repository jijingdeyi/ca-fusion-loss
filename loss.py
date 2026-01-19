import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
import warnings


def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    
    if len(input.shape) == 4:
        # 2D input: (B, C, H, W)
        C = input.shape[1]
        out = input
        win_size = win.shape[-1]
        pad_size = win_size // 2
        
        # Apply 1D filter along H dimension
        if input.shape[2] >= win_size:
            win_h = win.view(1, 1, 1, win_size)  # (1, 1, 1, win_size)
            out = F.conv2d(out, weight=win_h, stride=1, padding=(pad_size, 0), groups=C)
        
        # Apply 1D filter along W dimension
        if input.shape[3] >= win_size:
            win_w = win.view(1, 1, win_size, 1)  # (1, 1, win_size, 1)
            out = F.conv2d(out, weight=win_w, stride=1, padding=(0, pad_size), groups=C)
        
        return out
    elif len(input.shape) == 5:
        # 3D input: (B, C, D, H, W)
        raise NotImplementedError("3D gaussian filter not implemented")
    else:
        raise NotImplementedError(f"Unsupported input shape: {input.shape}")


def ssim_map_fn(
    X: torch.Tensor,
    Y: torch.Tensor,
    data_range: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
    K=(0.01, 0.03),
):
    """
    Return per-pixel SSIM map: (B, C, H, W)
    """
    K1, K2 = K
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # window
    win = _fspecial_gauss_1d(win_size, win_sigma)
    win = win.repeat([X.shape[1]] + [1] * (X.dim() - 2))
    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(X * X, win) - mu1_sq
    sigma2_sq = gaussian_filter(Y * Y, win) - mu2_sq
    sigma12   = gaussian_filter(X * Y, win) - mu1_mu2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    return ssim_map.clamp(0, 1)



class Sobel2D(nn.Module):
    """Return Gx, Gy; using normal zero padding (padding=1)"""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32)[None, None]  # [1,1,3,3]
        ky = torch.tensor([[-1, -2, -1],
                           [0,  0,  0],
                           [1,  2,  1]], dtype=torch.float32)[None, None]  # [1,1,3,3]
        self.kx = nn.Parameter(kx, requires_grad=False)
        self.ky = nn.Parameter(ky, requires_grad=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        C = x.shape[1]
        kx_expanded = self.kx.repeat(C, 1, 1, 1)  # [C,1,3,3]
        ky_expanded = self.ky.repeat(C, 1, 1, 1)  # [C,1,3,3]
        gx = F.conv2d(x, kx_expanded, padding=1, groups=C)
        gy = F.conv2d(x, ky_expanded, padding=1, groups=C)
        return gx, gy


class SoftGradlossAligned(nn.Module):
    """
    Multi-scale soft direction-aligned gradient loss.
    Softly selects VIS / IR gradient by magnitude difference.
    """

    def __init__(self, scales=(1.0, 0.5, 0.25), slope=15.0):
        super().__init__()
        self.scales = scales
        self.slope = slope
        self.sobel = Sobel2D()

    def _resize(self, x, ratio):
        if ratio == 1.0:
            return x
        H, W = x.shape[-2:]
        return F.interpolate(
            x,
            size=(int(H * ratio), int(W * ratio)),
            mode='bilinear',
            align_corners=False
        )

    def forward(self, vis, ir, fused):
        """
        vis, ir, fused: (B,1,H,W)
        fused: RAW output (no clamp)
        """
        loss = 0.0
        w_scale = 1.0 / len(self.scales)

        for r in self.scales:
            vis_s   = self._resize(vis, r)
            ir_s    = self._resize(ir, r)
            fused_s = self._resize(fused, r)

            gvx, gvy = self.sobel(vis_s)
            gix, giy = self.sobel(ir_s)
            gfx, gfy = self.sobel(fused_s)

            # Compute gradient magnitude (L1 norm: |gx| + |gy|)
            gv = torch.abs(gvx) + torch.abs(gvy)
            gi = torch.abs(gix) + torch.abs(giy)
            gf = torch.abs(gfx) + torch.abs(gfy)

            # soft selection weight
            w = torch.sigmoid(self.slope * (gi - gv))

            # softly aligned target gradient
            target = w * gi + (1.0 - w) * gv

            loss = loss + w_scale * F.l1_loss(gf, target)

        return loss


class fusion_loss(nn.Module):
    def __init__(self,
                 blur_ks=9,
                 tau=0.20,
                 mask_slope=20.0,
                 thr_sigma=0.5):
        super().__init__()

        self.blur_ks = blur_ks
        self.tau = tau
        self.mask_slope = mask_slope
        self.thr_sigma = thr_sigma

        self.grad_loss = SoftGradlossAligned(
            scales=(1.0, 0.5, 0.25),
            slope=15.0
        )

    def _blur(self, x):
        k = self.blur_ks
        return F.avg_pool2d(x, k, stride=1, padding=k // 2)

    def forward(self, ir, vis, fused):
        """
        ir, vis: (B,1,H,W), in [0,1]
        fused: RAW output (not clamped)
        """
        fused_raw = fused
        fused_clamp = fused_raw.clamp(0, 1)

        # -------------------------
        # 1) Base / Detail decomposition
        # -------------------------
        ir_b   = self._blur(ir)
        vis_b  = self._blur(vis)
        fused_b = self._blur(fused_clamp)

        ir_d   = ir - ir_b
        fused_d = fused_clamp - fused_b

        # -------------------------
        # 2) Base loss: soft-max fusion
        # -------------------------
        w = torch.softmax(
            torch.cat([ir_b / self.tau, vis_b / self.tau], dim=1),
            dim=1
        )
        w_ir, w_vis = w[:, 0:1], w[:, 1:2]
        target_b = w_ir * ir_b + w_vis * vis_b
        loss_base = F.l1_loss(fused_b, target_b)

        # -------------------------
        # 3) IR bright-region mask
        # -------------------------
        ir_mean = ir.mean(dim=[2, 3], keepdim=True)
        ir_std  = ir.std(dim=[2, 3], keepdim=True) + 1e-6
        thr = ir_mean + self.thr_sigma * ir_std

        m = torch.sigmoid(self.mask_slope * (ir - thr)).detach()

        # -------------------------
        # 4) Detail loss (IR shape prior)
        # -------------------------
        loss_detail = (m * (fused_d - ir_d).abs()).mean()

        # -------------------------
        # 5) Soft gradient aligned loss
        # -------------------------
        loss_grad = self.grad_loss(vis, ir, fused_raw)

        # -------------------------
        # 6) Bloom suppression (conditional)
        # -------------------------
        loss_bloom = (m * F.relu(fused_b - vis_b)).mean()

        # ---------- Region-aware SSIM ----------
        ssim_map = ssim_map_fn(fused_clamp, vis, data_range=1.0)
        w_bg = (1.0 - m)
        loss_ssim = (w_bg * (1.0 - ssim_map)).sum() / (w_bg.sum() + 1e-6)

        # -------------------------
        # 8) Weights
        # -------------------------
        alpha_base   = 1.0
        alpha_detail = 3.0
        beta_grad    = 2.0
        gamma_ssim   = 0.2
        lambda_bloom = 0.2

        loss = (
            alpha_base * loss_base +
            alpha_detail * loss_detail +
            beta_grad * loss_grad +
            gamma_ssim * loss_ssim +
            lambda_bloom * loss_bloom
        )

        return loss, {
            "loss_base": loss_base.item(),
            "loss_detail": loss_detail.item(),
            "loss_grad": loss_grad.item(),
            "loss_ssim": loss_ssim.item(),
            "loss_bloom": loss_bloom.item()
        }





