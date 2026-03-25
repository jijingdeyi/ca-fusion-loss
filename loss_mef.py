import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import ssim


class Sobelxy(nn.Module):
    # Sobelxy gradient loss with just norm
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        image_A_Y = image_A[:, :1, :, :]
        image_B_Y = image_B[:, :1, :, :]
        image_fused_Y = image_fused[:, :1, :, :]
        gradient_A = self.sobelconv(image_A_Y)
        gradient_B = self.sobelconv(image_B_Y)
        gradient_fused = self.sobelconv(image_fused_Y)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        weight_A = 0.5
        weight_B = 0.5
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        return Loss_SSIM


class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_fused, intensity_target):
        Loss_intensity = F.l1_loss(image_fused, intensity_target)
        return Loss_intensity


class fusion_loss_mef(nn.Module):
    def __init__(self,
                 blur_ks=9,                   # 均值模糊核大小：用于 halo 分支中的 IR/VIS/fused 局部亮度比较
                 mask_slope=20.0,             # M_soft 的 sigmoid 斜率：越大越“二值”，越小越平滑
                 ir_brightness_thr=0.6,       # IR 亮度阈值：与 VIS 极亮条件联合，触发“强制取 max”
                 vis_brightness_thr=0.6,      # VIS 亮度阈值：显著区内超过该阈值时倾向取 min(IR, VIS)
                 vis_super_brightness_thr=0.9,  # VIS 极亮阈值：灯泡等高亮区域回退到 max(IR, VIS)
                 vis_gate_quantile=0.8,       # VIS 分位数门控：用于过滤 M_hard，仅保留较亮 VIS 区域
                 ring_ks=3,                   # ring 膨胀核大小：用于从 M_hard 生成环带 Mring
                 eta_halo=15.0,               # halo sigmoid 斜率：控制 (vis_b-ir_b-delta) 到权重 h 的过渡陡峭程度
                 delta_halo=0.5,              # halo 偏移量：提高该值会减少被判定为 halo 的区域
                 eps_halo=0.04,               # halo 容忍边际：允许 fused_b 略高于 ir_b，超出部分才惩罚
                 lambda_halo=0.1):            # halo 损失权重
        super(fusion_loss_mef, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()
        self.blur_ks = blur_ks
        self.mask_slope = mask_slope
        self.ir_brightness_thr = ir_brightness_thr
        self.vis_brightness_thr = vis_brightness_thr
        self.vis_super_brightness_thr = vis_super_brightness_thr
        self.vis_gate_quantile = vis_gate_quantile
        self.ring_ks = ring_ks
        self.eta_halo = eta_halo
        self.delta_halo = delta_halo
        self.eps_halo = eps_halo
        self.lambda_halo = lambda_halo

    def _blur(self, x):
        return F.avg_pool2d(x, self.blur_ks, stride=1, padding=self.blur_ks // 2)

    def _dilate(self, x):
        return F.max_pool2d(x, kernel_size=self.ring_ks, stride=1, padding=self.ring_ks // 2)

    @staticmethod
    def _erode(x, k):
        # min-pooling via negative max-pooling
        return -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k // 2)

    def _opening(self, x, k):
        # grayscale opening = dilate(erode(x))
        eroded = self._erode(x, k)
        return F.max_pool2d(eroded, kernel_size=k, stride=1, padding=k // 2)

    @staticmethod
    def _compute_quantile(x, q):
        # Per-image quantile (no cross-image mixing)
        B = x.shape[0]
        x_flat = x.view(B, -1)
        q_val = torch.quantile(x_flat, q, dim=1, keepdim=True)
        return q_val.view(B, 1, 1, 1)

    def get_saliency_masks(self, image_A):
        """Return M_soft and M_hard using multi-scale Top-hat + quantile thresholds."""
        ir1 = image_A[:, :1, :, :]
        kernel_sizes = [9, 15, 31]

        S_multi = []
        for k in kernel_sizes:
            if k % 2 == 0:
                raise ValueError(f"Kernel size must be odd, got {k}")
            opened = self._opening(ir1, k)
            S_k = ir1 - opened  # Top-hat
            S_multi.append(S_k)

        S = torch.stack(S_multi, dim=0).max(dim=0)[0]
        S = S / (S.mean(dim=[2, 3], keepdim=True) + 1e-6)

        p_soft = 0.03
        thr_soft = self._compute_quantile(S, 1.0 - p_soft)
        M_soft = torch.sigmoid(self.mask_slope * (S - thr_soft))

        p_hard = 0.03
        thr_hard = self._compute_quantile(S, 1.0 - p_hard)

        q_bright = 0.8
        thr_bright = self._compute_quantile(ir1, q_bright)
        M_hard = ((S > thr_hard) & (ir1 > thr_bright)).float()
        return M_soft, M_hard

    def get_halo_masks(self, image_A, image_B):
        """Return M_hard and Mhalo for visualization.
        image_A: IR, image_B: VIS-Y
        """
        ir1 = image_A[:, :1, :, :]
        vis1 = image_B[:, :1, :, :]
        ir_b = self._blur(ir1)
        vis_b = self._blur(vis1)

        _, M_hard = self.get_saliency_masks(image_A)
        thr_vis = self._compute_quantile(vis1, self.vis_gate_quantile)
        M_hard = (M_hard * (vis1 > thr_vis).float())

        Mdil = self._dilate(M_hard)
        Mring = (Mdil - M_hard).clamp(0.0, 1.0)
        h = torch.sigmoid(self.eta_halo * (vis_b - ir_b - self.delta_halo))
        Mhalo = Mring * h
        return M_hard, Mhalo

    def forward(self, image_A, image_B, image_fused):
        # image_A: IR, image_B: VIS-Y, image_fused: fused-Y
        M_soft, _ = self.get_saliency_masks(image_A)
        M_soft = M_soft.detach()

        intensity_max = torch.max(image_A, image_B)
        intensity_min = torch.min(image_A, image_B)
        # In salient region, keep very bright VIS regions on max (e.g., lamps),
        # otherwise use min when VIS is bright.
        ir_bright = image_A > self.ir_brightness_thr
        vis_bright = image_B > self.vis_brightness_thr
        vis_super_bright = image_B > self.vis_super_brightness_thr
        salient_target = torch.where(vis_bright, intensity_min, intensity_max)
        salient_target = torch.where(vis_super_bright & ir_bright, intensity_max, salient_target)
        intensity_target = M_soft * salient_target + (1.0 - M_soft) * intensity_max

        loss_l1 = 20 * self.L_Inten(image_fused, intensity_target)
        loss_gradient = 20 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused))

        # Halo loss branch (adapted from loss.py)
        ir_b = self._blur(image_A[:, :1, :, :])
        vis_b = self._blur(image_B[:, :1, :, :])
        fused_b = self._blur(image_fused[:, :1, :, :]).clamp(0, 1)

        M_hard, Mhalo = self.get_halo_masks(image_A, image_B)
        M_hard = M_hard.detach()
        Mhalo = Mhalo.detach()

        sum_mhalo = (Mhalo.sum() + 1e-6).clamp(min=1.0)
        loss_halo = (Mhalo * F.relu(fused_b - (ir_b + self.eps_halo))).sum() / sum_mhalo

        fusion_loss = loss_l1 + loss_gradient + loss_SSIM + self.lambda_halo * loss_halo
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM, self.lambda_halo * loss_halo
