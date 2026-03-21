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
        loss = 0.0
        w_scale = 1.0 / len(self.scales)

        for r in self.scales:
            vis_s   = self._resize(vis, r)
            ir_s    = self._resize(ir, r)
            fused_s = self._resize(fused, r)

            gvx, gvy = self.sobel(vis_s)
            gix, giy = self.sobel(ir_s)
            gfx, gfy = self.sobel(fused_s)

            # magnitude for gating only (no direction lost in supervision)
            gv = torch.abs(gvx) + torch.abs(gvy)
            gi = torch.abs(gix) + torch.abs(giy)

            w = torch.sigmoid(self.slope * (gi - gv))

            # direction-aligned soft target (component-wise, sign-preserving)
            tx = w * gix + (1.0 - w) * gvx
            ty = w * giy + (1.0 - w) * gvy

            loss_s = F.l1_loss(gfx, tx) + F.l1_loss(gfy, ty)
            loss = loss + w_scale * loss_s

        return loss


class fusion_loss(nn.Module):
    def __init__(self,
                 blur_ks=9,
                 tau=0.30,
                 # mask for salient IR region M
                 mask_slope=20.0,
                 thr_sigma=0.5,
                 # detail soft select
                 kappa_detail=15.0,
                 # halo ring（ring_ks 越小环越窄；lambda_halo/eps_halo 过大会在目标周围压出黑边）
                 ring_ks=3,          # dilation kernel size (odd)，5→约2像素环宽，9→约4像素
                 eta_halo=15.0,      # slope for halo candidate h
                 delta_halo=0.5,    # tolerance threshold δ in (BV - BI - δ)
                 ir_brightness_thr=0.4,  # IR 亮度下限，仅用于 M_hard：只有 ir > 此值才视为高置信目标参与膨胀
                 eps_bloom=0.02,     # ε_bloom
                 eps_halo=0.04,      # ε_halo，略放宽可减轻黑边
                 # weights
                 alpha_base=1.0,
                 alpha_detail=4.0,
                 beta_grad=3.0,
                 gamma_ssim=0.2,
                 lambda_bloom=0.1,
                 lambda_halo=0.04,   # 过大会在目标周围压出明显黑边，可适当减小
                 ):
        super().__init__()

        self.blur_ks = blur_ks
        self.tau = tau
        self.mask_slope = mask_slope
        self.thr_sigma = thr_sigma

        self.kappa_detail = kappa_detail

        self.ring_ks = ring_ks
        self.eta_halo = eta_halo
        self.delta_halo = delta_halo
        self.ir_brightness_thr = ir_brightness_thr
        self.eps_bloom = eps_bloom
        self.eps_halo = eps_halo

        self.alpha_base = alpha_base
        self.alpha_detail = alpha_detail
        self.beta_grad = beta_grad
        self.gamma_ssim = gamma_ssim
        self.lambda_bloom = lambda_bloom
        self.lambda_halo = lambda_halo

        self.grad_loss = SoftGradlossAligned(
            scales=(1.0, 0.5, 0.25),
            slope=15.0
        )

    def _blur(self, x):
        k = self.blur_ks
        return F.avg_pool2d(x, k, stride=1, padding=k // 2)

    def _dilate_soft(self, m):
        """Differentiable-ish dilation via maxpool (but we will detach masks anyway)."""
        k = self.ring_ks
        return F.max_pool2d(m, kernel_size=k, stride=1, padding=k // 2)

    @staticmethod
    def _weighted_l1(pred, target, weight, eps=1e-6, min_den=1.0):
        # pred/target/weight: (B,1,H,W)；min_den 防止 Mbase 极稀疏时分母过小导致 loss 爆炸
        num = (weight * (pred - target).abs()).sum()
        den = (weight.sum() + eps).clamp(min=min_den)
        return num / den

    def forward(self, ir, vis, fused):
        """
        ir, vis: (B,1,H,W), in [0,1]
        fused: RAW output (not clamped)
        """
        # 1) 使用 raw 输出参与主体 loss，避免全局 clamp 截断梯度
        fused_raw = fused
        # 可视化 / 日志用的安全版本（不参与梯度）
        fused_clamped_for_log = fused_raw.clamp(0, 1).detach()
        # -------------------------
        # 1) Base / Detail decomposition（用 raw 输出）
        # -------------------------
        ir_b    = self._blur(ir)
        vis_b   = self._blur(vis)
        fused_b = self._blur(fused_raw)
        ir_d    = ir - ir_b
        vis_d   = vis - vis_b
        fused_d = fused_raw - fused_b
        # -------------------------
        # 2) Multi-scale local-contrast salient masks (soft + hard)
        #    这一段保持你现在的多尺度 S 版本，不改
        # -------------------------
        window_sizes = [15, 31]
        S_multi = []
        for k in window_sizes:
            ir_bg_k = F.avg_pool2d(ir, kernel_size=k, stride=1, padding=k // 2)
            S_k = ir - ir_bg_k
            S_multi.append(S_k)
        S = torch.stack(S_multi, dim=0).max(dim=0)[0]
        s_mean = S.mean(dim=[2, 3], keepdim=True)
        s_std  = S.std(dim=[2, 3], keepdim=True) + 1e-6
        thr_soft = s_mean + self.thr_sigma * s_std
        M_soft = torch.sigmoid(self.mask_slope * (S - thr_soft)).detach()  # (B,1,H,W)
        thr_hard = s_mean + 1.5 * s_std
        M_hard = ((S > thr_hard) & (ir > self.ir_brightness_thr)).float().detach()
        # -------------------------
        # 3) Halo-ring mask Mhalo（保持不变）
        # -------------------------
        Mdil = self._dilate_soft(M_hard)
        Mring = (Mdil - M_hard).clamp(0.0, 1.0)
        h = torch.sigmoid(self.eta_halo * (vis_b - ir_b - self.delta_halo))
        Mhalo = (Mring * h).detach()
        Mbase = ((1.0 - M_soft) * (1.0 - Mhalo)).detach()
        # -------------------------
        # 4) Base loss（用 raw fused_b）
        # -------------------------
        w = torch.softmax(
            torch.cat([ir_b / self.tau, vis_b / self.tau], dim=1),
            dim=1
        )
        w_ir, w_vis = w[:, 0:1], w[:, 1:2]
        target_b = w_ir * ir_b + w_vis * vis_b
        loss_base = self._weighted_l1(fused_b, target_b, Mbase, min_den=1.0)
        # -------------------------
        # 5) Detail loss（用 raw fused_d）
        # -------------------------
        wD = torch.sigmoid(self.kappa_detail * (ir_d.abs() - vis_d.abs()))
        DT = wD * ir_d + (1.0 - wD) * vis_d
        sum_m_soft = (M_soft.sum() + 1e-6).clamp(min=1.0)
        loss_detail = (M_soft * (fused_d - DT).abs()).sum() / sum_m_soft
        # -------------------------
        # 6) Soft gradient aligned loss（用 raw fused_raw）
        # -------------------------
        loss_grad = self.grad_loss(vis, ir, fused_raw)
        # -------------------------
        # 7) Bloom / Halo：在亮度空间里解释，需要 [0,1]，这里局部 clamp
        # -------------------------
        BF_for_bloom = self._blur(fused_raw).clamp(0, 1)  # 只用于 bloom/halo 中的阈值逻辑
        loss_bloom = (M_soft * F.relu(BF_for_bloom - (ir_b + self.eps_bloom))).sum() / sum_m_soft
        sum_mhalo = (Mhalo.sum() + 1e-6).clamp(min=1.0)
        loss_halo = (Mhalo * F.relu(BF_for_bloom - (ir_b + self.eps_halo))).sum() / sum_mhalo
        # -------------------------
        # 8) Region-aware SSIM：SSIM 定义在 [0,1]，这里只对这一项做 clamp
        # -------------------------
        ssim_map = ssim_map_fn(fused_raw.clamp(0, 1), vis, data_range=1.0)
        sum_mbase = (Mbase.sum() + 1e-6).clamp(min=1.0)
        loss_ssim = (Mbase * (1.0 - ssim_map)).sum() / sum_mbase
        # -------------------------
        # 9) Total（后面保持不变）
        # -------------------------
        loss = (
            self.alpha_base   * loss_base +
            self.alpha_detail * loss_detail +
            self.beta_grad    * loss_grad +
            self.gamma_ssim   * loss_ssim +
            self.lambda_bloom * loss_bloom +
            self.lambda_halo  * loss_halo
        )

        # 标量日志信息
        loss_stats = {
            "loss_base":  float(loss_base.detach().cpu()),
            "loss_detail": float(loss_detail.detach().cpu()),
            "loss_grad":  float(loss_grad.detach().cpu()),
            "loss_ssim":  float(loss_ssim.detach().cpu()),
            "loss_bloom": float(loss_bloom.detach().cpu()),
            "loss_halo":  float(loss_halo.detach().cpu()),
            "m_mean":     float(M_soft.mean().detach().cpu()),
            "mhalo_mean": float(Mhalo.mean().detach().cpu()),
            "mbase_mean": float(Mbase.mean().detach().cpu()),
        }

        # 掩膜张量（用于可视化）
        mask_dict = {
            "M": M_soft,
            "Mhard": M_hard,
            "Mbase": Mbase,
            "Mhalo": Mhalo,
        }

        # 返回：总 loss、标量日志、掩膜
        return loss, loss_stats, mask_dict




