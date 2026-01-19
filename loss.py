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
    def __init__(self,
                 blur_ks=9,          # base/detail分解的模糊核大小
                 tau=0.20,           # base融合温度，越小越接近hard max
                 mask_slope=20.0,    # 亮区mask的sigmoid斜率
                 thr_mu=0.0,         # 阈值 = mean + thr_mu*std
                 thr_sigma=0.5):
        super().__init__()
        self.sobelconv = Sobelxy()
        self.blur_ks = blur_ks
        self.tau = tau
        self.mask_slope = mask_slope
        self.thr_mu = thr_mu
        self.thr_sigma = thr_sigma

    def _blur(self, x):
        # 简单可导的低频提取：均值滤波
        k = self.blur_ks
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

    def forward(self, ir, vis, fused):
        """
        ir, vis, fused: (B,1,H,W) 且在 [0,1]（至少输入应如此）
        """
        fused_raw = fused
        fused_clamp = fused_raw.clamp(0, 1)

        # -------------------------
        # 1) Grad loss (结构对齐)
        # -------------------------
        vis_grad = self.sobelconv(vis)
        ir_grad  = self.sobelconv(ir)
        target_grad = torch.max(vis_grad, ir_grad)
        fused_grad  = self.sobelconv(fused_raw)   # 用 raw，避免 clamp 截断梯度
        loss_grad = F.l1_loss(fused_grad, target_grad)

        # -------------------------
        # 2) SSIM loss (观感对齐VIS)
        # -------------------------
        loss_ssim = 1 - ssim(fused_clamp, vis, data_range=1.0, size_average=True)

        # -------------------------
        # 3) Base/Detail 分解
        # -------------------------
        ir_b   = self._blur(ir)
        vis_b  = self._blur(vis)
        fused_b = self._blur(fused_clamp)   # base用clamp更稳

        ir_d   = ir - ir_b
        vis_d  = vis - vis_b
        fused_d = fused_clamp - fused_b

        # -------------------------
        # 4) Base（低频亮度）: 温和max融合，避免VIS光晕硬搬
        #    target_b = softmax([ir_b, vis_b]) 加权和
        # -------------------------
        tau = self.tau
        w = torch.softmax(torch.cat([ir_b / tau, vis_b / tau], dim=1), dim=1)  # (B,2,H,W)
        w_ir = w[:, 0:1, :, :]
        w_vis = w[:, 1:2, :, :]
        target_b = w_ir * ir_b + w_vis * vis_b
        loss_base = F.l1_loss(fused_b, target_b)

        # -------------------------
        # 5) Detail（高频形状）: 在IR亮区强制细节来自IR
        #    关键：亮区用 IR 的 mean+std 做阈值，而不是 (ir - mean) 这种相对量
        # -------------------------
        ir_mean = ir.mean(dim=[2, 3], keepdim=True)
        ir_std  = ir.std(dim=[2, 3], keepdim=True) + 1e-6
        thr = self.thr_mu * ir_mean + (1 - self.thr_mu) * ir_mean  # 兼容参数写法（可忽略）
        thr = ir_mean + self.thr_sigma * ir_std                    # 默认 mean + 0.5*std

        m = torch.sigmoid(self.mask_slope * (ir - thr)).detach()   # IR亮区mask

        # 在亮区：fused_d ≈ ir_d（保灯泡/目标边界）
        loss_detail = (m * (fused_d - ir_d).abs()).mean()

        # -------------------------
        # 6) 可选：抑制“VIS低频光晕过大”进一步保险（建议先开小权重）
        #    在IR亮区，抑制 fused_base 比 vis_base 更亮（防止复制VIS大光晕）
        # -------------------------
        loss_bloom = (m * F.relu(fused_b - vis_b)).mean()

        # -------------------------
        # 7) 总损失权重（建议从这组起步）
        # -------------------------
        alpha_base   = 1.0
        alpha_detail = 3.0    # 让高频形状更强势，直接解决你提的“灯泡形状”
        beta_grad    = 2.0
        gamma_ssim   = 0.5
        lambda_bloom = 0.2    # 先小一点，不然可能压暗整体

        loss = (alpha_base * loss_base
                + alpha_detail * loss_detail
                + beta_grad * loss_grad
                + gamma_ssim * loss_ssim
                + lambda_bloom * loss_bloom)

        return loss, {
            "loss_base": loss_base.item(),
            "loss_detail": loss_detail.item(),
            "loss_grad": loss_grad.item(),
            "loss_ssim": loss_ssim.item(),
            "loss_bloom": loss_bloom.item(),
        }





