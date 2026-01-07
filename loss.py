import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)
    def forward(self,x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


class fusion_loss(nn.Module):
    def __init__(self):
        super(fusion_loss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, ir, vis, fused):

        y_grad = self.sobelconv(vis)
        fused_grad = self.sobelconv(fused)
        loss_grad = F.l1_loss(y_grad, fused_grad)

        loss_ssim = 1 - ssim(fused, vis, data_range=1.0, size_average=True)

        k = 2 #temperature parameter
        ir_mean = ir.mean(dim=[2,3], keepdim=True)
        omega_ir = torch.sigmoid(k * (ir - ir_mean)).detach()
        loss_sal = (omega_ir * (fused - ir).abs()).mean()

        vis_mean  = vis.mean(dim=[2,3], keepdim=True)
        fused_mean = fused.mean(dim=[2,3], keepdim=True)
        loss_mean = (fused_mean - vis_mean).abs().mean()

        tv_h = torch.abs(fused[:, :, 1:, :] - fused[:, :, :-1, :]).mean()
        tv_w = torch.abs(fused[:, :, :, 1:] - fused[:, :, :, :-1]).mean()
        loss_tv = tv_h + tv_w

        alpha = 1
        beta = 5
        gamma = 1
        rho = 0.2
        delta = 0.001
        loss_fusion = alpha * loss_sal + beta * loss_grad + gamma * loss_ssim + rho * loss_mean + delta * loss_tv
        
        return loss_fusion


