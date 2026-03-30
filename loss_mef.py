import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import ssim
from loss_halo import PhysicsConsistentHaloMask


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
                 lambda_halo=0.3,             # halo 损失权重（由 0.1 上调）
                 lambda_bloom=0.6,           # bloom 损失权重（由 0.2 上调）
                 w_l1=15.0,                   # L1 主损失权重（由 20 下调，降低主导性）
                 w_grad=14.0,                 # 梯度损失权重（由 20 下调，避免压制其他分支）
                 w_ssim=20.0):                # SSIM 损失权重（由 10 上调，提升结构约束）
        super(fusion_loss_mef, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()
        self.blur_ks = blur_ks
        self.halo_mask_builder = PhysicsConsistentHaloMask()
        self.eta_halo = 15.0  # halo sigmoid 斜率：控制 (vis_b-ir_b-delta) 到权重 h 的过渡陡峭程度
        self.delta_halo = 0.5  # halo 偏移量：提高该值会减少被判定为 halo 的区域
        self.lambda_halo = lambda_halo
        self.lambda_bloom = lambda_bloom
        self.w_l1 = w_l1
        self.w_grad = w_grad
        self.w_ssim = w_ssim

    def _blur(self, x):
        return F.avg_pool2d(x, self.blur_ks, stride=1, padding=self.blur_ks // 2)

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

    def build_distance_decay(self, M_light, near_radius=50):
        """
        Build a simple binary template:
        near M_light -> 1, far from M_light -> 0.
        """
        near_radius = int(max(1, near_radius))
        ks = 2 * near_radius + 1

        coords = torch.arange(-near_radius, near_radius + 1, device=M_light.device)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        disk = ((xx * xx + yy * yy) <= (near_radius * near_radius)).to(M_light.dtype)
        disk_kernel = disk.view(1, 1, ks, ks)

        near_score = F.conv2d((M_light > 0).to(M_light.dtype), disk_kernel, padding=near_radius)
        return (near_score > 0).to(M_light.dtype)

    def _compute_saliency(self, image_A):
        """Compute shared saliency map from IR image."""
        ir1 = image_A[:, :1, :, :]
        kernel_sizes = [9, 15, 31]

        saliency_scales = []
        for k in kernel_sizes:
            opened = self._opening(ir1, k)
            saliency_k = ir1 - opened  # Top-hat
            saliency_scales.append(saliency_k)

        saliency = torch.stack(saliency_scales, dim=0).max(dim=0)[0]
        saliency = saliency / (saliency.mean(dim=[2, 3], keepdim=True) + 1e-6)
        return saliency, ir1

    def get_M_mask(self, image_A):
        saliency, _ = self._compute_saliency(image_A)
        mask_slope = 20.0

        top_ratio_m = 0.1
        thr_m = self._compute_quantile(saliency, 1.0 - top_ratio_m)
        M = torch.sigmoid(mask_slope * (saliency - thr_m))
        return M

    def get_M_light_base_mask(self, image_A):
        saliency, ir1 = self._compute_saliency(image_A)
        ir_gate_quantile = 0.97

        top_ratio_light = 0.03
        thr_m_light = self._compute_quantile(saliency, 1.0 - top_ratio_light)

        thr_ir_bright = self._compute_quantile(ir1, ir_gate_quantile)
        M_light = ((saliency > thr_m_light) & (ir1 > thr_ir_bright)).float()  # 显著性大并且红外亮度大
        return M_light

    def get_M_light_mask(self, image_A, image_B):
        vis1 = image_B[:, :1, :, :]
        M_light = self.get_M_light_base_mask(image_A)
        light_gate_quantile = 0.97

        thr_vis_super = self._compute_quantile(vis1, light_gate_quantile)
        M_light = (M_light * (vis1 > thr_vis_super).float())  # 显著性大并且红外亮度大且可见光亮度大，潜在灯泡区域
        return M_light

    def get_M_halo_mask(self, image_A, image_B):
        ir1 = image_A[:, :1, :, :]
        M_light = self.get_M_light_mask(image_A, image_B)

        # Use halo mask from loss_halo.py (some versions return tensor, others return tuple)
        halo_out = self.halo_mask_builder(ir1, image_B[:, :1, :, :])
        if isinstance(halo_out, tuple):
            halo_out = halo_out[0]

        # Remove the light core itself from halo candidate.
        Mhalo = torch.clamp(halo_out - M_light, 0, 1)

        # Step3: 空间衰减（平滑）
        decay = self.build_distance_decay(M_light)

        # Final
        Mhalo = Mhalo * decay
        return Mhalo

    def forward(self, image_A, image_B, image_fused):
        # image_A: IR, image_B: VIS-Y, image_fused: fused-Y
        M = self.get_M_mask(image_A)
        M_light = self.get_M_light_mask(image_A, image_B)
        M = M.detach()
        M_light = M_light.detach()

        intensity_max = torch.max(image_A, image_B)
        intensity_min = torch.min(image_A, image_B)
        # 如果红外亮，可见光也亮，说明目标照明良好，压制亮度
        # 如果红外亮，可见光极亮，说明很可能是灯泡，最大亮度
        ir_gate_quantile = 0.7
        vis_gate_quantile = 0.4
        light_gate_quantile = 0.85
        thr_ir = self._compute_quantile(image_A, ir_gate_quantile).clamp(0.45, 0.85)
        thr_vis = self._compute_quantile(image_B, vis_gate_quantile).clamp(0.30, 0.70)
        thr_vis_super = self._compute_quantile(image_B, light_gate_quantile).clamp(0.75, 0.98)

        ir_bright = image_A > thr_ir
        vis_bright = image_B > thr_vis
        vis_super_bright = image_B > thr_vis_super
        salient_target = torch.where(vis_bright, intensity_min, intensity_max)
        salient_target = torch.where(vis_super_bright & ir_bright, intensity_max, salient_target)
        intensity_target = M * salient_target + (1.0 - M) * intensity_max

        loss_l1 = self.w_l1 * self.L_Inten(image_fused, intensity_target)
        loss_gradient = self.w_grad * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = self.w_ssim * (1 - self.L_SSIM(image_A, image_B, image_fused))

        # Halo loss branch (adapted from loss.py)
        ir_b = self._blur(image_A[:, :1, :, :])
        vis_b = self._blur(image_B[:, :1, :, :])
        fused_b = self._blur(image_fused[:, :1, :, :]).clamp(0, 1)

        Mhalo = self.get_M_halo_mask(image_A, image_B)
        Mhalo = Mhalo.detach()

        # Halo loss: penalize fused brightness exceeding IR brightness in halo regions
        sum_mhalo = (Mhalo.sum() + 1e-6).clamp(min=1.0)
        loss_halo = (Mhalo * F.relu(fused_b - ir_b)).sum() / sum_mhalo

        # Bloom loss: constrain fused brightness to match visible brightness in M regions (salient targets)
        sum_m = (M.sum() + 1e-6).clamp(min=1.0)
        loss_bloom = (M * F.relu(fused_b - vis_b)).sum() / sum_m

        weighted_halo = self.lambda_halo * loss_halo
        weighted_bloom = self.lambda_bloom * loss_bloom
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM + weighted_halo + weighted_bloom
        # Keep training script interface unchanged; the last term now reflects total anti-halo regularization.
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM, (weighted_halo + weighted_bloom)