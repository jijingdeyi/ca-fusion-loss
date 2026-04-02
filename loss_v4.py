import torch
import torch.nn as nn
import torch.nn.functional as F

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
    Baseline + two lightweight degradation suppression losses

    Main idea:
    1) Keep the strong baseline: max-intensity + max-gradient + SSIM
    2) Use IR-saliency mask M for target over-exposure suppression (bloom loss)
    3) Use VIS-bright-core ring mask Mhalo for halo suppression (halo loss)

    Notes:
    - No PhysicsConsistentHaloMask
    - M_light is defined from VIS super-bright compact cores
    - Mhalo is simply a ring around M_light
    """

    def __init__(self,
                 blur_ks=9,
                 lambda_halo=0.3,
                 lambda_bloom=0.6,
                 w_l1=15.0,
                 w_grad=14.0,
                 w_ssim=20.0,
                 # ----- target mask (IR saliency) -----
                 top_ratio_m=0.10,
                 mask_slope=20.0,
                 # ----- light core mask (VIS bright core) -----
                 light_core_quantile=0.98,   # top 0.5% as light core candidate
                 light_open_ks=5,             # opening kernel to suppress tiny noise
                 light_max_area_ratio=0.1,   # suppress huge bright area (sky/wall), per-image
                 # ----- halo ring -----
                 ring_inner_ks=9,             # inner dilation kernel
                 ring_outer_ks=31,            # outer dilation kernel
                 # ----- loss margin -----
                 eps_halo=0.00,
                 eps_bloom=0.00):
        super(fusion_loss_mef, self).__init__()

        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()

        self.blur_ks = blur_ks
        self.lambda_halo = lambda_halo
        self.lambda_bloom = lambda_bloom
        self.w_l1 = w_l1
        self.w_grad = w_grad
        self.w_ssim = w_ssim

        self.top_ratio_m = top_ratio_m
        self.mask_slope = mask_slope

        self.light_core_quantile = light_core_quantile
        self.light_open_ks = light_open_ks
        self.light_max_area_ratio = light_max_area_ratio

        self.ring_inner_ks = ring_inner_ks
        self.ring_outer_ks = ring_outer_ks

        self.eps_halo = eps_halo
        self.eps_bloom = eps_bloom

    # ------------------------------------------------------------------
    # basic ops
    # ------------------------------------------------------------------
    def _blur(self, x):
        return F.avg_pool2d(x, self.blur_ks, stride=1, padding=self.blur_ks // 2)

    @staticmethod
    def _erode(x, k):
        # min-pooling via negative max-pooling
        return -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k // 2)

    def _opening(self, x, k):
        eroded = self._erode(x, k)
        return F.max_pool2d(eroded, kernel_size=k, stride=1, padding=k // 2)

    @staticmethod
    def _dilate(x, k):
        return F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

    @staticmethod
    def _compute_quantile(x, q):
        # Per-image quantile
        B = x.shape[0]
        x_flat = x.view(B, -1)
        q_val = torch.quantile(x_flat, q, dim=1, keepdim=True)
        return q_val.view(B, 1, 1, 1)

    @staticmethod
    def _remove_large_bright_regions(mask, max_area_ratio=0.01):
        """
        A lightweight area suppression without connected-component analysis:
        if a mask is too large globally, down-weight / reject it.
        This is crude but stable and GPU-friendly.
        """
        B, _, H, W = mask.shape
        area = mask.view(B, -1).mean(dim=1, keepdim=True).view(B, 1, 1, 1)  # area ratio
        keep = (area <= max_area_ratio).float()
        return mask * keep

    # ------------------------------------------------------------------
    # target mask M: from IR saliency
    # ------------------------------------------------------------------
    def _compute_saliency(self, image_A):
        """
        Compute IR saliency using multi-scale top-hat.
        image_A: IR
        """
        ir1 = image_A[:, :1, :, :]
        kernel_sizes = [9, 15, 31]

        saliency_scales = []
        for k in kernel_sizes:
            opened = self._opening(ir1, k)
            saliency_k = ir1 - opened
            saliency_scales.append(saliency_k)

        saliency = torch.stack(saliency_scales, dim=0).max(dim=0)[0]
        saliency = saliency / (saliency.mean(dim=[2, 3], keepdim=True) + 1e-6)
        return saliency, ir1

    def get_M_mask(self, image_A):
        """
        Target mask M for salient IR targets.
        """
        saliency, _ = self._compute_saliency(image_A)
        thr_m = self._compute_quantile(saliency, 1.0 - self.top_ratio_m)
        M = torch.sigmoid(self.mask_slope * (saliency - thr_m))
        return M

    # ------------------------------------------------------------------
    # light core mask: from VIS super-bright compact core
    # ------------------------------------------------------------------
    def get_M_light_mask(self, image_A, image_B):
        """
        Build light-core mask from VIS only, with optional target exclusion.

        image_A: IR
        image_B: VIS-Y
        """
        vis1 = image_B[:, :1, :, :]
        ir1 = image_A[:, :1, :, :]

        # 1) super-bright core from VIS
        thr_vis_core = self._compute_quantile(vis1, self.light_core_quantile)
        M_vis_core = (vis1 > thr_vis_core).float()

        # 2) suppress isolated tiny noise by opening
        if self.light_open_ks > 1:
            M_vis_core = self._opening(M_vis_core, self.light_open_ks)
            M_vis_core = (M_vis_core > 0).float()

        # 3) suppress globally huge bright region (e.g. bright sky / wall)
        M_vis_core = self._remove_large_bright_regions(
            M_vis_core,
            max_area_ratio=self.light_max_area_ratio
        )

        # 4) optional target exclusion:
        # remove strong IR salient targets, because halo source should be VIS light source,
        # not pedestrian / hot object core.
        M_target_hard = (self.get_M_mask(image_A) > 0.5).float()
        M_light = M_vis_core * (1.0 - M_target_hard)

        # 5) optional IR exclusion for extremely hot large targets
        # keep it weak: only exclude pixels that are both very bright in IR and in target region
        thr_ir_hot = self._compute_quantile(ir1, 0.99)
        M_ir_hot = (ir1 > thr_ir_hot).float()
        M_light = M_light * (1.0 - M_ir_hot * M_target_hard)

        return M_light

    # ------------------------------------------------------------------
    # halo mask: ring around light core
    # ------------------------------------------------------------------
    def get_M_halo_mask(self, image_A, image_B):
        """
        Halo region = ring around VIS light core.
        No explicit halo-shape estimation.
        """
        M_light = self.get_M_light_mask(image_A, image_B)

        inner = self._dilate(M_light, self.ring_inner_ks)
        outer = self._dilate(M_light, self.ring_outer_ks)

        Mhalo = (outer - inner).clamp(0.0, 1.0)

        # exclude salient target region to avoid mixing with bloom branch
        M_target_hard = (self.get_M_mask(image_A) > 0.5).float()
        Mhalo = Mhalo * (1.0 - M_target_hard)

        return Mhalo

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, image_A, image_B, image_fused):
        """
        image_A: IR
        image_B: VIS-Y
        image_fused: fused-Y
        """
        # --------------------------------------------------------------
        # 1) target mask for bloom suppression
        # --------------------------------------------------------------
        M = self.get_M_mask(image_A).detach()

        # --------------------------------------------------------------
        # 2) baseline intensity target
        # --------------------------------------------------------------
        intensity_max = torch.max(image_A, image_B)
        intensity_min = torch.min(image_A, image_B)

        # If IR bright and VIS also bright -> likely well-illuminated target,
        # suppress excessive IR-dominated brightness.
        # If VIS is super bright and IR is also bright -> likely light-source-like region,
        # keep max intensity.
        ir_gate_quantile = 0.70
        vis_gate_quantile = 0.40
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

        # --------------------------------------------------------------
        # 3) baseline losses
        # --------------------------------------------------------------
        loss_l1 = self.w_l1 * self.L_Inten(image_fused, intensity_target)
        loss_gradient = self.w_grad * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = self.w_ssim * (1 - self.L_SSIM(image_A, image_B, image_fused))

        # --------------------------------------------------------------
        # 4) blur space for halo / bloom regulation
        # --------------------------------------------------------------
        ir_b = self._blur(image_A[:, :1, :, :])
        vis_b = self._blur(image_B[:, :1, :, :])
        fused_b = self._blur(image_fused[:, :1, :, :]).clamp(0, 1)

        # --------------------------------------------------------------
        # 5) halo loss: ring around VIS light core
        # --------------------------------------------------------------
        Mhalo = self.get_M_halo_mask(image_A, image_B).detach()
        sum_mhalo = (Mhalo.sum() + 1e-6).clamp(min=1.0)

        loss_halo = (
            Mhalo * F.relu(fused_b - (ir_b + self.eps_halo))
        ).sum() / sum_mhalo

        # --------------------------------------------------------------
        # 6) bloom loss: salient IR target region
        # --------------------------------------------------------------
        sum_m = (M.sum() + 1e-6).clamp(min=1.0)
        loss_bloom = (
            M * F.relu(fused_b - (vis_b + self.eps_bloom))
        ).sum() / sum_m

        # --------------------------------------------------------------
        # 7) total
        # --------------------------------------------------------------
        weighted_halo = self.lambda_halo * loss_halo
        weighted_bloom = self.lambda_bloom * loss_bloom

        fusion_loss = loss_l1 + loss_gradient + loss_SSIM + weighted_halo + weighted_bloom

        # keep your original training interface unchanged
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM, (weighted_halo + weighted_bloom)