import torch
import torch.nn as nn
import torch.nn.functional as F


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
        gradient_joint = torch.max(self.sobelconv(image_A_Y), self.sobelconv(image_B_Y))
        return F.l1_loss(self.sobelconv(image_fused_Y), gradient_joint)


class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_fused, intensity_target):
        return F.l1_loss(image_fused, intensity_target)


class fusion_loss_mef(nn.Module):
    """
    Paper-style additive loss, no bins / no learnable weights / no SSIM.

      L = w_l1 ||F - max(I,V)||_1
        + w_grad ||∇F - max(∇I,∇V)||_1
        + λ_h ||M_h ⊙ (F - I)||_1
        + λ_w ||M_w ⊙ (F - V)||_1

    Masks (training-only, detached):
      halo:     V is in the top intensity quantile AND V > I + delta
      washout:  I is in the top intensity quantile AND I > V + delta
    """

    def __init__(self,
                 w_l1=20,
                 w_grad=20,
                 lambda_halo=0.5,
                 lambda_bloom=0.2,
                 q_bright=0.90,
                 delta=0.0):
        super(fusion_loss_mef, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.w_l1 = w_l1
        self.w_grad = w_grad
        self.lambda_halo = float(lambda_halo)
        self.lambda_bloom = float(lambda_bloom)
        self.q_bright = q_bright
        self.delta = delta

    @staticmethod
    def _compute_quantile(x, q):
        B = x.shape[0]
        q_val = torch.quantile(x.view(B, -1), q, dim=1, keepdim=True)
        return q_val.view(B, 1, 1, 1)

    def _bright_and_dominant(self, bright_src, other, q, delta):
        thr = self._compute_quantile(bright_src, q)
        return ((bright_src >= thr) & (bright_src > other + delta)).float()

    def get_M_halo_mask_union(self, image_A, image_B=None):
        if image_B is None:
            raise TypeError("Halo mask needs IR and VIS: get_M_halo_mask_union(ir, vis_y)")
        ir = image_A[:, :1, :, :]
        vis = image_B[:, :1, :, :]
        return self._bright_and_dominant(vis, ir, self.q_bright, self.delta).detach()

    def get_M_bloom_mask_union(self, image_A, image_B=None):
        if image_B is None:
            raise TypeError("Washout mask needs IR and VIS: get_M_bloom_mask_union(ir, vis_y)")
        ir = image_A[:, :1, :, :]
        vis = image_B[:, :1, :, :]
        return self._bright_and_dominant(ir, vis, self.q_bright, self.delta).detach()

    @staticmethod
    def _masked_l1(pred, target, mask):
        denom = (mask.sum() + 1e-6).clamp(min=1.0)
        return (mask * (pred - target).abs()).sum() / denom

    def forward(self, image_A, image_B, image_fused):
        ir = image_A[:, :1, :, :]
        vis = image_B[:, :1, :, :]
        fused = image_fused[:, :1, :, :].clamp(0, 1)

        intensity_target = torch.max(image_A, image_B)
        loss_l1 = self.w_l1 * self.L_Inten(image_fused, intensity_target)
        loss_gradient = self.w_grad * self.L_Grad(image_A, image_B, image_fused)

        m_halo = self.get_M_halo_mask_union(ir, vis)
        m_wash = self.get_M_bloom_mask_union(ir, vis)
        loss_halo = self._masked_l1(fused, ir, m_halo)
        loss_wash = self._masked_l1(fused, vis, m_wash)
        loss_reg = self.lambda_halo * loss_halo + self.lambda_bloom * loss_wash

        fusion_loss = loss_l1 + loss_gradient + loss_reg
        mask_cov = m_halo.mean() + m_wash.mean()
        return fusion_loss, loss_gradient, loss_l1, loss_reg, mask_cov
