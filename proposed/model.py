import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT_DIR / "baseline" / "UIE_Benchmark"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from model.UIEC2Net import UIEC2Net, piece_function_org
from proposed.degradation_encoder import DegradationEncoder
from proposed.conditioning import FiLMModulation

class DegradationAwareUIEC2Net(nn.Module):
    """
    Proposed Method: Degradation-Aware Underwater Image Enhancement.
    Wraps the UIEC2Net baseline architecture with a lightweight Degradation Encoder (D)
    and FiLM feature conditioning blocks.
    
    Baseline:  Y = F(X)
    Proposed:  z = D(X)
               Y = F(X, z)
    """
    def __init__(self, embed_dim: int = 64):
        super(DegradationAwareUIEC2Net, self).__init__()
        self.baseline = UIEC2Net()
        self.degradation_encoder = DegradationEncoder(in_features=10, embed_dim=embed_dim)
        
        # FiLM feature conditioning blocks inserted into confidence estimation branch
        self.film_rgb = FiLMModulation(embed_dim=embed_dim, num_features=64)
        self.film_conf = FiLMModulation(embed_dim=embed_dim, num_features=64)

    def load_baseline_weights(self, checkpoint_path: str, device: torch.device):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
            
        new_ckpt = {}
        for key, value in ckpt.items():
            if key.startswith("model."):
                new_ckpt[key[6:]] = value
            else:
                new_ckpt[key] = value
                
        self.baseline.load_state_dict(new_ckpt, strict=False)

    def freeze_baseline(self, freeze: bool = True):
        for param in self.baseline.parameters():
            param.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Estimate degradation vector z = D(x)
        z = self.degradation_encoder(x)
        
        # 2. Baseline RGB branch with degradation conditioning
        h_rgb = self.baseline.rgb_fuction_down(self.baseline.rgb_norm_batch1(self.baseline.rgb_con1(x)))
        h_rgb = self.baseline.rgb_fuction_down(self.baseline.rgb_norm_batch2(self.baseline.rgb_con2(h_rgb)))
        h_rgb = self.baseline.rgb_fuction_down(self.baseline.rgb_norm_batch3(self.baseline.rgb_con3(h_rgb)))
        
        # Apply FiLM conditioning to RGB features
        h_rgb = self.film_rgb(h_rgb, z)
        
        h_rgb = self.baseline.rgb_fuction_down(self.baseline.rgb_norm_batch4(self.baseline.rgb_con4(h_rgb)))
        h_rgb = self.baseline.rgb_fuction_up(self.baseline.rgb_norm_batch5(self.baseline.rgb_con5(h_rgb)))
        h_rgb = self.baseline.rgb_fuction_up(self.baseline.rgb_norm_batch6(self.baseline.rgb_con6(h_rgb)))
        rgb_out = torch.sigmoid(self.baseline.rgb_con7(h_rgb))[:, 0:3, :, :]
        
        # 3. HSV branch
        hsv_fromrgbout = self.baseline.rgb2hsv(rgb_out)
        hsv_input = torch.cat([hsv_fromrgbout, hsv_fromrgbout], dim=1)
        batch_size = hsv_input.size()[0]
        
        x1 = self.baseline.relu(self.baseline.e_conv1(hsv_input))
        x1 = self.baseline.maxpool(x1)
        x2 = self.baseline.relu(self.baseline.e_conv2(x1))
        x2 = self.baseline.maxpool(x2)
        x3 = self.baseline.relu(self.baseline.e_conv3(x2))
        x3 = self.baseline.maxpool(x3)
        x4 = self.baseline.relu(self.baseline.e_conv4(x3))
        x_r = self.baseline.relu(self.baseline.e_conv7(x4))
        x_r = self.baseline.avagepool(x_r).view(batch_size, -1)
        x_r = self.baseline.e_convfc(x_r)
        
        H, S, V, H2S = torch.split(x_r, self.baseline.M, dim=1)
        H_in, S_in, V_in = hsv_input[:, 0:1, :, :], hsv_input[:, 1:2, :, :], hsv_input[:, 2:3, :, :]
        H_out = piece_function_org(H_in, H, self.baseline.M)
        S_out1 = piece_function_org(S_in, S, self.baseline.M)
        V_out = piece_function_org(V_in, V, self.baseline.M)
        S_out2 = piece_function_org(H_in, H2S, self.baseline.M)
        S_out = (S_out1 + S_out2) / 2
        
        zero_lab = torch.zeros(S_out.shape, device=S_out.device)
        one_lab = torch.ones(S_out.shape, device=S_out.device)
        S_out = torch.clamp(S_out, 0, 1)
        V_out = torch.clamp(V_out, 0, 1)
        
        hsv_out = torch.cat([H_out, S_out, V_out], dim=1)
        hsv_out_rgb = self.baseline.hsv2rgb(hsv_out)
        
        # 4. Confidence fusion branch with degradation conditioning
        conf_input = torch.cat([x, rgb_out, hsv_out_rgb], dim=1)
        h_conf = self.baseline.fuction_down(self.baseline.norm_batch1(self.baseline.con1(conf_input)))
        h_conf = self.baseline.fuction_down(self.baseline.norm_batch2(self.baseline.con2(h_conf)))
        h_conf = self.baseline.fuction_down(self.baseline.norm_batch3(self.baseline.con3(h_conf)))
        
        # Apply FiLM conditioning to fusion confidence features
        h_conf = self.film_conf(h_conf, z)
        
        h_conf = self.baseline.fuction_down(self.baseline.norm_batch4(self.baseline.con4(h_conf)))
        h_conf = self.baseline.fuction_down(self.baseline.norm_batch5(self.baseline.con5(h_conf)))
        h_conf = self.baseline.fuction_down(self.baseline.norm_batch6(self.baseline.con6(h_conf)))
        conf_out = torch.sigmoid(self.baseline.con7(h_conf))
        
        conf_rgb = conf_out[:, 0:3, :, :]
        conf_hsv = conf_out[:, 3:6, :, :]
        
        output = 0.5 * conf_rgb * rgb_out + 0.5 * conf_hsv * hsv_out_rgb
        return output

if __name__ == "__main__":
    model = DegradationAwareUIEC2Net()
    x = torch.rand(2, 3, 256, 256)
    out = model(x)
    print("Output shape:", out.shape)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters:", trainable_params)
