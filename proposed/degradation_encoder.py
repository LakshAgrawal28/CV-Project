import torch
import torch.nn as nn
import torch.nn.functional as F

class DegradationFeatureExtractor(nn.Module):
    """
    Extracts 10 compact physics & degradation indicators directly from input tensor [B, 3, H, W] in [0, 1].
    Features:
    1-3. mean_R, mean_G, mean_B
    4-6. R-G diff, R-B diff, G-B diff
    7. Global brightness (luminance)
    8. Global contrast (luminance std)
    9. Color cast ratio (R+B)/(2G)
    10. Laplacian variance (sharpness/blur indicator)
    """
    def __init__(self):
        super(DegradationFeatureExtractor, self).__init__()
        # 3x3 Laplacian kernel for edge/sharpness variance estimation
        lap_kernel = torch.tensor([[0., 1., 0.],
                                   [1., -4., 1.],
                                   [0., 1., 0.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("lap_kernel", lap_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        # 1-3. Mean RGB channels
        mean_r = torch.mean(x[:, 0:1, :, :], dim=(2, 3))
        mean_g = torch.mean(x[:, 1:2, :, :], dim=(2, 3))
        mean_b = torch.mean(x[:, 2:3, :, :], dim=(2, 3))
        
        # 4-6. Channel differences
        rg_diff = mean_r - mean_g
        rb_diff = mean_r - mean_b
        gb_diff = mean_g - mean_b
        
        # 7-8. Luminance brightness & contrast
        lum = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        brightness = torch.mean(lum, dim=(2, 3))
        contrast = torch.std(lum, dim=(2, 3))
        
        # 9. Color cast indicator
        color_cast = (mean_r + mean_b) / (2.0 * mean_g + 1e-6)
        
        # 10. Laplacian variance (blur/attenuation)
        lum_gray = lum
        lap_map = F.conv2d(lum_gray, self.lap_kernel, padding=1)
        lap_var = torch.var(lap_map, dim=(2, 3))
        
        features = torch.cat([
            mean_r, mean_g, mean_b,
            rg_diff, rb_diff, gb_diff,
            brightness, contrast, color_cast, lap_var
        ], dim=1) # [B, 10]
        
        return features

class DegradationEncoder(nn.Module):
    """
    Degradation Encoder network mapping compact degradation features f to learned embedding z = D(x).
    """
    def __init__(self, in_features: int = 10, embed_dim: int = 64):
        super(DegradationEncoder, self).__init__()
        self.extractor = DegradationFeatureExtractor()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_feats = self.extractor(x)
        z = self.mlp(raw_feats)
        return z

if __name__ == "__main__":
    encoder = DegradationEncoder()
    x = torch.rand(2, 3, 256, 256)
    z = encoder(x)
    print("Degradation embedding z shape:", z.shape)
