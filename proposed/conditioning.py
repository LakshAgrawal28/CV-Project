import torch
import torch.nn as nn

class FiLMModulation(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) block.
    Given degradation embedding z [B, D] and feature map x [B, C, H, W],
    predicts scale gamma [B, C, 1, 1] and shift beta [B, C, 1, 1]:
    y = (1 + gamma) * x + beta
    """
    def __init__(self, embed_dim: int = 64, num_features: int = 64):
        super(FiLMModulation, self).__init__()
        self.fc_gamma = nn.Linear(embed_dim, num_features)
        self.fc_beta = nn.Linear(embed_dim, num_features)
        
        # Initialize gamma to zero so identity (unmodulated) feature pass holds at start
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        gamma = self.fc_gamma(z).view(b, c, 1, 1)
        beta = self.fc_beta(z).view(b, c, 1, 1)
        
        out = (1.0 + gamma) * x + beta
        return out
