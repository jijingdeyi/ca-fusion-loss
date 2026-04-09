import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from pytorch_msssim import ssim


class Sobelxy(nn.Module):
    # Sobel gradient magnitude
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
        loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return loss_gradient


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        weight_A = 0.5
        weight_B = 0.5
        loss_ssim = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        return loss_ssim


class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_fused, intensity_target):
        loss_intensity = F.l1_loss(image_fused, intensity_target)
        return loss_intensity


class fusion_loss_mef(nn.Module):
    """
    Minimal MEF loss:
    1) Baseline: max-intensity + max-gradient + SSIM
    2) Halo mask: VIS high-brightness quantile
    3) Bloom mask: IR high-brightness quantile
    """

    def __init__(self,
                 w_l1=20,
                 w_grad=20,
                 w_ssim=10,
                 halo_bins=(0.90, 0.95, 0.98, 1.00),
                 bloom_bins=(0.90, 0.95, 0.98, 1.00),
                 lambda_halo_init=(0.46, 0.66, 0.92),
                 lambda_bloom_init=(0.46, 0.46, 0.46),
                 lambda_min=0.20,
                 eps_halo=0.00,
                 eps_bloom=0.00):
        super(fusion_loss_mef, self).__init__()

        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()

        self.w_l1 = w_l1
        self.w_grad = w_grad
        self.w_ssim = w_ssim
        self.halo_bins = halo_bins
        self.bloom_bins = bloom_bins
        self.lambda_min = lambda_min
        self.lambda_halo_raw = nn.Parameter(self._init_raw_lambdas(lambda_halo_init))
        self.lambda_bloom_raw = nn.Parameter(self._init_raw_lambdas(lambda_bloom_init))

        self.eps_halo = eps_halo
        self.eps_bloom = eps_bloom

    @staticmethod
    def _compute_quantile(x, q):
        B = x.shape[0]
        x_flat = x.view(B, -1)
        q_val = torch.quantile(x_flat, q, dim=1, keepdim=True)
        return q_val.view(B, 1, 1, 1)

    @staticmethod
    def _inv_softplus(y):
        # stable inverse for softplus(y) where y > 0
        return math.log(math.exp(float(y)) - 1.0)

    def _init_raw_lambdas(self, init_vals):
        vals = torch.tensor(init_vals, dtype=torch.float32)
        vals = vals.clamp(min=self.lambda_min + 1e-4)
        shifted = vals - self.lambda_min
        raw = torch.tensor([self._inv_softplus(v.item()) for v in shifted], dtype=torch.float32)
        return raw

    def get_lambda_halo(self):
        return self.lambda_min + F.softplus(self.lambda_halo_raw)

    def get_lambda_bloom(self):
        return self.lambda_min + F.softplus(self.lambda_bloom_raw)

    def _multi_bin_masks(self, x, bins):
        """
        Split the original large soft mask into interval sub-masks.
        The sum of all sub-masks equals the original soft mask:
            M_big = clamp((x - thr(q0)) / (1 - thr(q0)), 0, 1)
            M_big = M1 + M2 + ... + Mk
        """
        assert len(bins) >= 2, "bins length must be >= 2"
        thrs = [self._compute_quantile(x, q) for q in bins]
        base = ((x - thrs[0]) / (1.0 - thrs[0] + 1e-6)).clamp(0.0, 1.0)
        masks = []
        for i in range(len(thrs) - 1):
            lo = thrs[i]
            hi = thrs[i + 1]
            in_bin = (x >= lo).float() * (x < hi).float()
            if i == len(thrs) - 2:
                in_bin = (x >= lo).float() * (x <= hi).float()
            masks.append(base * in_bin)
        return masks

    def _union_mask_from_bins(self, x, bins):
        # Return the original large soft mask used before splitting.
        thr0 = self._compute_quantile(x, bins[0])
        return ((x - thr0) / (1.0 - thr0 + 1e-6)).clamp(0.0, 1.0)

    def get_M_halo_mask_simple(self, image_B):
        vis1 = image_B[:, :1, :, :]
        masks = self._multi_bin_masks(vis1, self.halo_bins)
        return torch.stack(masks, dim=0).sum(dim=0).clamp(0.0, 1.0)

    def get_M_bloom_mask_simple(self, image_A):
        ir1 = image_A[:, :1, :, :]
        masks = self._multi_bin_masks(ir1, self.bloom_bins)
        return torch.stack(masks, dim=0).sum(dim=0).clamp(0.0, 1.0)

    def get_M_halo_mask_union(self, image_B):
        vis1 = image_B[:, :1, :, :]
        return self._union_mask_from_bins(vis1, self.halo_bins)

    def get_M_bloom_mask_union(self, image_A):
        ir1 = image_A[:, :1, :, :]
        return self._union_mask_from_bins(ir1, self.bloom_bins)

    def forward(self, image_A, image_B, image_fused):
        """
        image_A: IR
        image_B: VIS-Y
        image_fused: fused-Y
        """
        intensity_target = torch.max(image_A, image_B)

        loss_l1 = self.w_l1 * self.L_Inten(image_fused, intensity_target)
        loss_gradient = self.w_grad * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = self.w_ssim * (1 - self.L_SSIM(image_A, image_B, image_fused))

        ir_b = image_A[:, :1, :, :]
        vis_b = image_B[:, :1, :, :]
        fused_b = image_fused[:, :1, :, :].clamp(0, 1)

        halo_masks = [m.detach() for m in self._multi_bin_masks(image_B[:, :1, :, :], self.halo_bins)]
        bloom_masks = [m.detach() for m in self._multi_bin_masks(image_A[:, :1, :, :], self.bloom_bins)]

        loss_halo_bins = []
        for m in halo_masks:
            denom = (m.sum() + 1e-6).clamp(min=1.0)
            l = (m * torch.abs(fused_b - (ir_b + self.eps_halo))).sum() / denom
            loss_halo_bins.append(l)

        loss_bloom_bins = []
        for m in bloom_masks:
            denom = (m.sum() + 1e-6).clamp(min=1.0)
            l = (m * torch.abs(fused_b - (vis_b + self.eps_bloom))).sum() / denom
            loss_bloom_bins.append(l)

        lambda_halo = self.get_lambda_halo()
        lambda_bloom = self.get_lambda_bloom()
        weighted_halo = sum(w * l for w, l in zip(lambda_halo, loss_halo_bins))
        weighted_bloom = sum(w * l for w, l in zip(lambda_bloom, loss_bloom_bins))

        fusion_loss = loss_l1 + loss_gradient + loss_SSIM + weighted_halo + weighted_bloom
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM, (weighted_halo + weighted_bloom)