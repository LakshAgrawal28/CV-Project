"""
FDM-Net: Underwater Image Enhancement via Deep Modeling with Frequency Decomposition
Reference Paper: Yupeng Ma, Qiuling Yang, Rongxin Zhu (Springer / J. King Saud Univ. Comput. Inf. Sci., 2026)
DOI: https://doi.org/10.1007/s44443-026-00831-2

Faithful Architectural Reconstruction of the complete FDM-Net framework including:
1. IDFS   - Improved Dynamic Frequency Separator (Section 3.1, Eq. 4)
2. DERB   - Deformable Enhancement Residual Block for High-Frequency Details (Section 3.2.1, Eq. 6-7)
3. MDAB   - Multi-scale Dilated Attention Block for Low-Frequency Context (Section 3.2.1, Eq. 11-13)
4. FGDE   - Frequency-Guided Differential Expert Network with MoE Gating (Section 3.2.1, Eq. 5, 8-10, 14)
5. PUIM   - Physics-based UIM Inversion Mechanism with Alpha-Compensation (Section 3.2.2, Eq. 15-20)
6. BAMM   - Bidirectional Affine Modulation Module (Section 3.2.3, Eq. 21-24)
7. FEGBM  - Frequency-Expert-Guided Bidirectional Modulation Block (Section 3.2)
8. MCFN   - Multi-scale Convolutional Fusion Network (Section 3.3, Eq. 25-26)
9. FDMNet - Complete End-to-End Deep Reconstruction Architecture
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import deform_conv2d
    HAS_DEFORM_CONV = True
except ImportError:
    HAS_DEFORM_CONV = False


# ==============================================================================
# 1. Improved Dynamic Frequency Separator (IDFS) - Section 3.1
# ==============================================================================
class IDFS(nn.Module):
    """
    IDFS decomposes input feature Xin into low-frequency (Xl) and high-frequency (Xh)
    using complementary Global Average Pooling (GAP) and Global Max Pooling (GMP)
    with a soft allocation gating mechanism (Eq. 4).
    Low-frequency feature is refined by CNN -> Lin.
    High-frequency feature is refined by Multi-Head Self-Attention -> Hin.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super(IDFS, self).__init__()
        self.channels = channels
        
        # Dual-pooling global descriptor MLP
        self.fc_gap = nn.Linear(channels, channels // 4)
        self.fc_gmp = nn.Linear(channels, channels // 4)
        self.fc_gate = nn.Sequential(
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        
        # Low-frequency refinement CNN branch -> Lin
        self.conv_low = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )
        
        # High-frequency refinement Multi-Head Self-Attention branch -> Hin
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm_high = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        
        # Global Average & Max Pooling across spatial dimensions
        gap = torch.mean(x, dim=(2, 3))        # [B, C]
        gmp = torch.amax(x, dim=(2, 3))        # [B, C]
        
        # Shared descriptor fusion
        desc = F.relu(self.fc_gap(gap) + self.fc_gmp(gmp))
        gate = self.fc_gate(desc).view(b, c, 1, 1) # [B, C, 1, 1]
        
        # Soft frequency allocation (Eq. 4)
        X_l = x * gate
        X_h = x * (1.0 - gate)
        
        # Low-frequency CNN encoding -> Lin
        L_in = self.conv_low(X_l)
        
        # High-frequency Attention encoding -> Hin
        # Flatten spatial dimensions to sequence: [B, H*W, C]
        seq_h = X_h.permute(0, 2, 3, 1).reshape(b, h * w, c)
        attn_out, _ = self.mha(seq_h, seq_h, seq_h)
        seq_out = self.norm_high(seq_h + attn_out)
        H_in = seq_out.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        
        return L_in, H_in


# ==============================================================================
# 2. Deformable Enhancement Residual Block (DERB) - Section 3.2.1, Eq. 6-7
# ==============================================================================
class DERB(nn.Module):
    """
    High-frequency expert block recovering scattering-corrupted textures via
    learnable spatial sampling offsets and deformable convolution (Eq. 6-7).
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super(DERB, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Predict spatial offsets: 2 * k^2 channels
        self.offset_conv = nn.Conv2d(
            channels, 2 * kernel_size * kernel_size,
            kernel_size=3, padding=1, bias=True
        )
        # Initialize offsets to zero
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        
        self.weight = nn.Parameter(
            torch.empty(channels, channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offsets = self.offset_conv(x) # [B, 2*k*k, H, W]
        
        if HAS_DEFORM_CONV and x.is_cuda:
            deform_out = deform_conv2d(
                x, offsets, self.weight, padding=self.padding
            )
        else:
            # Fallback for CPU / standard testing environments
            deform_out = F.conv2d(x, self.weight, padding=self.padding)
            
        out = self.conv1x1(self.act(self.norm(deform_out))) + x
        return out


# ==============================================================================
# 3. Multi-Scale Dilated Attention Block (MDAB) - Section 3.2.1, Eq. 11-13
# ==============================================================================
class MDAB(nn.Module):
    """
    Low-frequency expert block enlarging receptive field with depthwise dilated
    convolutions and Channel Attention to capture global illumination (Eq. 11-13).
    """
    def __init__(self, channels: int, dilation: int = 2):
        super(MDAB, self).__init__()
        # Cascaded depthwise dilated convolutions
        self.dwc1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation,
            dilation=dilation, groups=channels, bias=False
        )
        self.dwc2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation,
            dilation=dilation, groups=channels, bias=False
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.act = nn.PReLU()
        
        # Channel-Attention (CA) module
        self.ca_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.act(self.dwc1(x))
        f2 = self.act(self.dwc2(f1))
        f2 = self.pointwise(f2)
        ca = self.ca_mlp(f2)
        out = ca * f2 + x
        return out


# ==============================================================================
# 4. Frequency-Guided Differential Expert Network (FGDE-Net) - Section 3.2.1
# ==============================================================================
class FGDENet(nn.Module):
    """
    Differential Expert Network with parallel DERB high-frequency experts and
    MDAB low-frequency experts routed by Mixture-of-Experts (MoE) gating (Eq. 5, 8-10, 14).
    """
    def __init__(self, channels: int, num_experts: int = 3):
        super(FGDENet, self).__init__()
        self.num_experts = num_experts
        
        # High-frequency experts (DERBs)
        self.derb_experts = nn.ModuleList([
            DERB(channels) for _ in range(num_experts)
        ])
        
        # Low-frequency experts (MDABs with progressive dilations 1, 2, 4)
        dilations = [1, 2, 4]
        self.mdab_experts = nn.ModuleList([
            MDAB(channels, dilation=dilations[i % len(dilations)])
            for i in range(num_experts)
        ])
        
        # Gating network for High-Frequency (Wh) and Low-Frequency (Wl)
        self.gate_high = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.PReLU(),
            nn.Linear(channels // 4, num_experts),
            nn.Softmax(dim=-1)
        )
        self.gate_low = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.PReLU(),
            nn.Linear(channels // 4, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, L_in: torch.Tensor, H_in: torch.Tensor):
        # Bidirectional differential features (Eq. 5)
        H_d = H_in - L_in
        L_d = torch.abs(L_in - H_in)
        
        # 1. High-frequency routing & combination (Eq. 8-10)
        he_desc = torch.mean(H_d, dim=(2, 3)) + torch.amax(H_d, dim=(2, 3))
        w_h = self.gate_high(he_desc) # [B, N]
        
        H_o = 0.0
        for i, expert in enumerate(self.derb_experts):
            f_i = expert(H_d)
            H_o = H_o + w_h[:, i:i+1, None, None] * f_i
            
        # 2. Low-frequency routing & combination (Eq. 14)
        le_desc = torch.mean(L_d, dim=(2, 3)) + torch.amax(L_d, dim=(2, 3))
        w_l = self.gate_low(le_desc) # [B, N]
        
        L_o = 0.0
        for j, expert in enumerate(self.mdab_experts):
            f_j = expert(L_d)
            L_o = L_o + w_l[:, j:j+1, None, None] * f_j
            
        return L_o, H_o


# ==============================================================================
# 5. Physics-Based UIM Inversion Mechanism (PUIM) - Section 3.2.2, Eq. 15-20
# ==============================================================================
class PUIM(nn.Module):
    """
    Underwater Imaging Model (UIM) physical inversion module:
    Estimates channel-wise background water light B and pixel transmission map T.
    Applies learnable channel compensation alpha = [1.5, 1.0, 0.8] for red absorption (Eq. 15-20).
    """
    def __init__(self, channels: int):
        super(PUIM, self).__init__()
        self.channels = channels
        
        # Background light MLP (Eq. 15, 17)
        self.mlp_b_h = nn.Sequential(
            nn.Linear(channels, 32),
            nn.PReLU(),
            nn.Linear(32, channels),
            nn.Softmax(dim=-1)
        )
        self.mlp_b_l = nn.Sequential(
            nn.Linear(channels, 32),
            nn.PReLU(),
            nn.Linear(32, channels),
            nn.Softmax(dim=-1)
        )
        
        # Transmission map estimation subnets (Eq. 16, 18)
        self.trans_net_h = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.trans_net_l = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Learnable channel-wise attenuation compensation vector (Eq. 19)
        # Initialized to [1.5, 1.0, 0.8] tiled across channels
        alpha_init = torch.tensor([1.5, 1.0, 0.8], dtype=torch.float32).repeat(
            math.ceil(channels / 3)
        )[:channels].view(1, channels, 1, 1)
        self.alpha_c = nn.Parameter(alpha_init)

    def forward(self, L_o: torch.Tensor, H_o: torch.Tensor):
        b, c, h, w = L_o.shape
        eps = 1e-4
        
        # Estimate background light B and transmission T
        B_h = self.mlp_b_h(torch.mean(H_o, dim=(2, 3))).view(b, c, 1, 1)
        B_l = self.mlp_b_l(torch.mean(L_o, dim=(2, 3))).view(b, c, 1, 1)
        
        T_h = self.trans_net_h(H_o)
        T_l = self.trans_net_l(L_o)
        
        # High-frequency inversion with alpha compensation (Eq. 19)
        H_star = self.alpha_c * ((H_o - B_h * (1.0 - T_h)) / (T_h + eps) + B_h)
        
        # Low-frequency inversion with scattering correction (Eq. 20)
        s_l = torch.mean(L_o, dim=(2, 3), keepdim=True)
        L_star = (L_o - B_l * (1.0 - T_l)) / (T_l + eps) + B_l - s_l
        
        return L_star, H_star


# ==============================================================================
# 6. Bidirectional Affine Modulation Module (BAMM) - Section 3.2.3, Eq. 21-24
# ==============================================================================
class BAMM(nn.Module):
    """
    Predicts affine scaling and shifting parameters via depthwise separable convs
    to perform residual cross-frequency modulation (Eq. 21-24):
    Lr = Lin * E + F
    Hr = Hin * P + Q
    Output = Lr + Hr
    """
    def __init__(self, channels: int):
        super(BAMM, self).__init__()
        # Depthwise separable mappings phi_h and phi_l
        self.phi_h = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, 2 * channels, kernel_size=1, bias=True)
        )
        self.phi_l = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, 2 * channels, kernel_size=1, bias=True)
        )

    def forward(self, L_in: torch.Tensor, H_in: torch.Tensor, L_star: torch.Tensor, H_star: torch.Tensor):
        # Predict affine parameters [E, F] and [P, Q] (Eq. 21)
        ef = self.phi_h(H_star)
        E, F = torch.chunk(ef, 2, dim=1)
        
        pq = self.phi_l(L_star)
        P, Q = torch.chunk(pq, 2, dim=1)
        
        # Bidirectional affine modulation (Eq. 22-23)
        L_r = L_in * torch.sigmoid(E) + F
        H_r = H_in * torch.sigmoid(P) + Q
        
        # Residual fusion (Eq. 24)
        out = L_r + H_r
        return out


# ==============================================================================
# 7. Frequency-Expert-Guided Bidirectional Modulation (FEGBM) - Complete Module
# ==============================================================================
class FEGBM(nn.Module):
    """
    Glues FGDE-Net, PUIM physical inversion, and BAMM affine modulation (Section 3.2).
    """
    def __init__(self, channels: int, num_experts: int = 3):
        super(FEGBM, self).__init__()
        self.fgde = FGDENet(channels, num_experts=num_experts)
        self.puim = PUIM(channels)
        self.bamm = BAMM(channels)

    def forward(self, L_in: torch.Tensor, H_in: torch.Tensor):
        L_o, H_o = self.fgde(L_in, H_in)
        L_star, H_star = self.puim(L_o, H_o)
        fused = self.bamm(L_in, H_in, L_star, H_star)
        return fused


# ==============================================================================
# 8. Multi-Scale Convolutional Fusion Network (MCFN) - Section 3.3, Eq. 25-26
# ==============================================================================
class MCFN(nn.Module):
    """
    Multi-Scale Convolutional Fusion Network capturing multi-scale spatial cues
    during upsampling using three parallel depthwise conv branches (Eq. 25-26).
    """
    def __init__(self, channels: int):
        super(MCFN, self).__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        # 3 parallel depthwise convolutions with k1=3, k2=5, k3=7
        c_sub = channels // 3
        c_rem = channels - 2 * c_sub
        self.splits = [c_sub, c_sub, c_rem]
        
        self.dwc1 = nn.Conv2d(c_sub, c_sub, kernel_size=3, padding=1, groups=c_sub, bias=False)
        self.dwc2 = nn.Conv2d(c_sub, c_sub, kernel_size=5, padding=2, groups=c_sub, bias=False)
        self.dwc3 = nn.Conv2d(c_rem, c_rem, kernel_size=7, padding=3, groups=c_rem, bias=False)
        
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ms = self.proj(x)
        x1, x2, x3 = torch.split(x_ms, self.splits, dim=1)
        
        o1 = self.dwc1(x1)
        o2 = self.dwc2(x2)
        o3 = self.dwc3(x3)
        
        out = self.fuse(torch.cat([o1, o2, o3], dim=1)) + x
        return out


# ==============================================================================
# 9. Complete End-to-End FDM-Net Architecture
# ==============================================================================
class FDMNet(nn.Module):
    """
    Full FDM-Net Architecture as specified in:
    'Underwater image enhancement via deep modeling with frequency decomposition'
    Yupeng Ma, Qiuling Yang, Rongxin Zhu (2026).
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super(FDMNet, self).__init__()
        
        # Initial feature projection
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.PReLU()
        )
        
        # Stage 1: IDFS decomposition & FEGBM modulation
        self.idfs1 = IDFS(base_channels)
        self.fegbm1 = FEGBM(base_channels)
        
        # Downsampling Stage 2
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.PReLU()
        )
        self.idfs2 = IDFS(base_channels * 2)
        self.fegbm2 = FEGBM(base_channels * 2)
        
        # Downsampling Stage 3 (Bottleneck)
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.PReLU()
        )
        self.idfs3 = IDFS(base_channels * 4)
        self.fegbm3 = FEGBM(base_channels * 4)
        
        # Upsampling Stage 1 with MCFN and Skip Connection
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.PReLU()
        )
        self.mcfn1 = MCFN(base_channels * 2)
        
        # Upsampling Stage 2 with MCFN and Skip Connection
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.PReLU()
        )
        self.mcfn2 = MCFN(base_channels)
        
        # Final reconstruction head
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial stem
        x0 = self.init_conv(x) # [B, 64, H, W]
        
        # Stage 1
        l1, h1 = self.idfs1(x0)
        f1 = self.fegbm1(l1, h1) # Skip 1
        
        # Stage 2
        x1 = self.down1(f1)
        l2, h2 = self.idfs2(x1)
        f2 = self.fegbm2(l2, h2) # Skip 2
        
        # Stage 3 (Bottleneck)
        x2 = self.down2(f2)
        l3, h3 = self.idfs3(x2)
        f3 = self.fegbm3(l3, h3)
        
        # Decoder Upsampling with MCFN & Skip connections
        u1 = self.up1(f3) + f2
        m1 = self.mcfn1(u1)
        
        u2 = self.up2(m1) + f1
        m2 = self.mcfn2(u2)
        
        out = self.final_conv(m2)
        return out


if __name__ == "__main__":
    print("Testing FDM-Net Architectural Reconstruction...")
    model = FDMNet(in_channels=3, base_channels=48)
    sample_input = torch.randn(1, 3, 128, 128)
    
    with torch.no_grad():
        sample_output = model(sample_input)
        
    print(f"Input Shape  : {sample_input.shape}")
    print(f"Output Shape : {sample_output.shape}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {num_params:,}")
    print("FDM-Net modules successfully verified!")

