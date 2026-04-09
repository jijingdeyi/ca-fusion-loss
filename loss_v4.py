import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import ssim


class Sobelxy(nn.Module):
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
        return F.l1_loss(gradient_fused, gradient_joint)


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        weight_A = 0.5
        weight_B = 0.5
        return weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)


class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_fused, intensity_target):
        return F.l1_loss(image_fused, intensity_target)


class fusion_loss_mef(nn.Module):
    """
    Baseline only: max-intensity L1 + max-gradient L1 + SSIM term.
    image_A: IR, image_B: VIS-Y, image_fused: fused-Y
    """

    def __init__(self, w_l1=20, w_grad=20, w_ssim=10):
        super(fusion_loss_mef, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()
        self.w_l1 = w_l1
        self.w_grad = w_grad
        self.w_ssim = w_ssim

    def forward(self, image_A, image_B, image_fused):
        intensity_target = torch.max(image_A, image_B)
        loss_l1 = self.w_l1 * self.L_Inten(image_fused, intensity_target)
        loss_gradient = self.w_grad * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = self.w_ssim * (1 - self.L_SSIM(image_A, image_B, image_fused))
        # fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        fusion_loss = loss_l1 + loss_gradient
        reg_aux = torch.zeros((), device=image_fused.device, dtype=image_fused.dtype)
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM, reg_aux
