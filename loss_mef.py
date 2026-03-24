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

    def forward(self, image_A, image_B, image_fused):
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity


class fusion_loss_mef(nn.Module):
    def __init__(self,
                 blur_ks=9,
                 ring_ks=3,
                 eta_halo=15.0,
                 delta_halo=0.5,
                 ir_brightness_thr=0.4,
                 eps_halo=0.04,
                 lambda_halo=0.1):
        super(fusion_loss_mef, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()
        self.blur_ks = blur_ks
        self.ring_ks = ring_ks
        self.eta_halo = eta_halo
        self.delta_halo = delta_halo
        self.ir_brightness_thr = ir_brightness_thr
        self.eps_halo = eps_halo
        self.lambda_halo = lambda_halo

    def _blur(self, x):
        return F.avg_pool2d(x, self.blur_ks, stride=1, padding=self.blur_ks // 2)

    def _dilate(self, x):
        return F.max_pool2d(x, kernel_size=self.ring_ks, stride=1, padding=self.ring_ks // 2)

    def get_halo_masks(self, image_A, image_B):
        """Return M_hard and Mhalo for visualization.
        image_A: IR, image_B: VIS-Y
        """
        ir1 = image_A[:, :1, :, :]
        ir_b = self._blur(ir1)
        vis_b = self._blur(image_B[:, :1, :, :])

        window_sizes = [15, 31]
        S_multi = []
        for k in window_sizes:
            ir_bg_k = F.avg_pool2d(ir1, kernel_size=k, stride=1, padding=k // 2)
            S_multi.append(ir1 - ir_bg_k)
        S = torch.stack(S_multi, dim=0).max(dim=0)[0]

        s_mean = S.mean(dim=[2, 3], keepdim=True)
        s_std = S.std(dim=[2, 3], keepdim=True) + 1e-6
        thr_hard = s_mean + 1.5 * s_std
        M_hard = ((S > thr_hard) & (ir1 > self.ir_brightness_thr)).float()

        Mdil = self._dilate(M_hard)
        Mring = (Mdil - M_hard).clamp(0.0, 1.0)
        h = torch.sigmoid(self.eta_halo * (vis_b - ir_b - self.delta_halo))
        Mhalo = Mring * h
        return M_hard, Mhalo

    def forward(self, image_A, image_B, image_fused):
        # image_A: IR, image_B: VIS-Y, image_fused: fused-Y
        loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
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
