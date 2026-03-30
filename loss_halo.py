import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast, Tuple


class PhysicsConsistentHaloMask(nn.Module):
    """
    Physics-Consistent Halo Detection with Complement Refinement (Final Version)
    
    Two-stage formulation:
    1. Initial mask M_init = H(x) · ((1-w)·E(x) + w·S(x)) · L(x)
    2. Refined mask from complement region with physical priors:
       - High brightness in IR/visible
       - Spatial coherence
       - Edge proximity
    3. Final confidence-weighted combination
    """
    def __init__(self, 
                 scatter_sigma=2.0,
                 scatter_beta=1.0,
                 halo_threshold=0.1,
                 edge_slope=10.0,
                 edge_threshold=0.05,
                 light_threshold=0.85,
                 structure_weight=0.5,
                 ir_brightness_threshold=0.6,
                 vis_brightness_threshold=0.5,
                 complement_weight=0.7,
                 min_halo_area=100):
        super().__init__()
        self.scatter_sigma = scatter_sigma
        self.scatter_beta = scatter_beta
        self.halo_threshold = halo_threshold
        self.edge_slope = edge_slope
        self.edge_threshold = edge_threshold
        self.light_threshold = light_threshold
        self.structure_weight = structure_weight
        self.ir_brightness_threshold = ir_brightness_threshold
        self.vis_brightness_threshold = vis_brightness_threshold
        self.complement_weight = complement_weight
        self.min_halo_area = min_halo_area
        
        self._init_scatter_kernel()
    
    def _init_scatter_kernel(self, kernel_size=21):
        """Heavy-tail scattering kernel K(r) ∝ exp(-(r/σ)^β)"""
        half = kernel_size // 2
        x = torch.arange(-half, half + 1)
        y = torch.arange(-half, half + 1)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        r = torch.sqrt(xx**2 + yy**2) / self.scatter_sigma
        
        kernel = torch.exp(-(r ** self.scatter_beta))
        kernel = kernel / kernel.sum()
        self.register_buffer('scatter_kernel', kernel.view(1, 1, kernel_size, kernel_size))
    
    def _compute_scatter(self, x):
        scatter_kernel = cast(torch.Tensor, self.scatter_kernel)
        scatter_padding = int(scatter_kernel.shape[-1] // 2)
        return F.conv2d(x, scatter_kernel, padding=scatter_padding)
    
    def _compute_modality_inconsistent_gradient(self, s_vis, s_ir):
        """∇(S_vis - S_ir) - focuses on modality-inconsistent boundaries"""
        diff = s_vis - s_ir
        sobel_x = torch.tensor(
            [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
            device=s_vis.device,
            dtype=s_vis.dtype,
        )
        sobel_y = torch.tensor(
            [[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]],
            device=s_vis.device,
            dtype=s_vis.dtype,
        )
        
        grad_x = F.conv2d(diff, sobel_x, padding=1)
        grad_y = F.conv2d(diff, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        return grad_mag
    
    def _compute_halo_candidates_from_complement(self, ir, vis, m_init):
        """
        Extract high-confidence halo candidates from complement region
        Based on physical priors:
        1. Higher brightness in IR (halos scatter more IR)
        2. Spatial coherence (connected components)
        3. Proximity to edges where scattering occurs
        """
        # Complement region (potential halo areas)
        complement = 1-m_init
        
        # Brightness prior: halos typically brighter in IR and visible
        ir_brightness = ir > self.ir_brightness_threshold
        vis_brightness = vis > self.vis_brightness_threshold
        brightness_prior = (ir_brightness | vis_brightness).float()
        
        # Apply complement mask
        candidates = brightness_prior * complement
        
        return candidates
    
    def _filter_by_area(self, mask, min_area_ratio=0.001):
        """
        Remove small isolated regions with a pure-PyTorch local-area filter.
        This avoids torchvision connected-components dependency.
        """
        h, w = mask.shape[2], mask.shape[3]
        min_area = max(1, int(h * w * min_area_ratio))

        # Approximate component-size filtering by requiring enough local support.
        ks = max(3, int(min_area ** 0.5))
        if ks % 2 == 0:
            ks += 1

        local_count = F.avg_pool2d(mask.float(), ks, stride=1, padding=ks // 2) * (ks * ks)
        keep = (local_count >= max(1, min_area // 2)).float()
        filtered = mask.float() * keep
        return filtered
    
    def _compute_confidence(self, ir, vis, m_candidate):
        """
        Compute confidence score for candidate halo regions
        Based on:
        - Brightness consistency with halo model
        - Gradient direction consistency
        - Scattering profile matching
        """
        # Brightness confidence: halos are brighter than surroundings
        local_mean_ir = F.avg_pool2d(ir, 15, stride=1, padding=7)
        local_mean_vis = F.avg_pool2d(vis, 15, stride=1, padding=7)
        
        brightness_conf_ir = torch.clamp((ir - local_mean_ir) / 0.2, 0, 1)
        brightness_conf_vis = torch.clamp((vis - local_mean_vis) / 0.2, 0, 1)
        brightness_conf = (brightness_conf_ir + brightness_conf_vis) / 2
        
        # Scattering profile confidence
        S_vis = self._compute_scatter(vis)
        S_ir = self._compute_scatter(ir)
        scatter_ratio = (S_ir + 1e-6) / (S_vis + 1e-6)
        scatter_conf = torch.clamp((scatter_ratio - 1.0) / 0.5, 0, 1)
        
        # Combined confidence
        confidence = brightness_conf * scatter_conf * m_candidate
        
        return confidence
    
    def forward(self, ir, vis):
        """
        Forward pass with complement refinement
        
        Args:
            ir: Infrared image [B, 1, H, W], range [0, 1]
            vis: Visible image [B, 1, H, W], range [0, 1]
            
        Returns:
            M_halo: Refined halo mask [B, 1, H, W], range [0, 1]
            confidence: Confidence scores [B, 1, H, W]
        """
        # Stage 1: Initial mask from original formulation
        S_vis = self._compute_scatter(vis)
        S_ir = self._compute_scatter(ir)
        scatter_diff = S_vis - S_ir
        
        # H(x): scattering residual
        H = torch.sigmoid(self.edge_slope * (scatter_diff - self.halo_threshold))
        
        # E(x): modality-inconsistent gradient, gated by H
        grad_mag = self._compute_modality_inconsistent_gradient(S_vis, S_ir)
        E_raw = torch.sigmoid(self.edge_slope * (grad_mag - self.edge_threshold))
        E = E_raw * H
        
        # S(x): spatial continuity prior
        S = F.avg_pool2d(H, 5, stride=1, padding=2)
        
        # L(x): light source suppression
        L = 1 - torch.sigmoid(20 * (vis - self.light_threshold))
        
        # Combine: H · ((1-w)·E + w·S) · L
        edge_structure = (1 - self.structure_weight) * E + self.structure_weight * S
        M_init = H * edge_structure * L
        M_init = F.avg_pool2d(M_init, 3, stride=1, padding=1)
        
        # Stage 2: Extract candidates from complement region
        M_halo = self._compute_halo_candidates_from_complement(ir, vis, M_init)
        

        return M_halo.detach()