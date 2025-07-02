import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Diffusion parameters
        self.num_timesteps = 1000
        self.beta = torch.linspace(1e-4, 0.02, self.num_timesteps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # U-Net architecture for 128x128
        self.down1 = self._block(1, 64)
        self.down2 = self._block(64, 128)
        self.down3 = self._block(128, 256)
        
        self.mid = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU()
        )
        
        self.up1 = self._block(512 + 256, 256)
        self.up2 = self._block(256 + 128, 128)
        self.up3 = self._block(128 + 64, 64)
        
        self.final = nn.Conv2d(64, 1, 1)
        
    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x, t=5):
        device = x.device
        """x shape: (batch, 1, 16, 16) → output shape: (batch, 1, 128, 128)"""
        # Step 1: Upscale input from 16x16 to 128x128
        x = F.interpolate(x, size=(128, 128), mode='bilinear')
        # Add noise based on timestep
        noise = torch.randn_like(x)
        alpha_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        alpha_t = alpha_t.to(device)
        noisy_x = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        
        # U-Net forward pass
        # Downsample
        d1 = self.down1(noisy_x)
        d2 = self.down2(F.max_pool2d(d1, 2))
        d3 = self.down3(F.max_pool2d(d2, 2))
        
        # Middle
        m = self.mid(F.max_pool2d(d3, 2))
        
        # Upsample with skip connections
        u1 = self.up1(torch.cat([F.interpolate(m, scale_factor=2), d3], 1))
        u2 = self.up2(torch.cat([F.interpolate(u1, scale_factor=2), d2], 1))
        u3 = self.up3(torch.cat([F.interpolate(u2, scale_factor=2), d1], 1))
        
        return self.final(u3)
    
    def generate(self, batch_size=1, steps=None):
        if steps is None:
            steps = self.num_timesteps
            
        with torch.no_grad():
            # Start with pure noise
            z = torch.randn(batch_size, 1, 128, 128)
            
            for t in reversed(range(steps)):
                # Predict and remove noise
                pred_noise = self(z, t)
                
                alpha_t = self.alpha[t]
                alpha_bar_t = self.alpha_bar[t]
                beta_t = self.beta[t]
                
                if t > 0:
                    noise = torch.randn_like(z)
                else:
                    noise = 0
                    
                z = (1 / torch.sqrt(alpha_t)) * (
                    z - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise
                ) + torch.sqrt(beta_t) * noise
                
            return z.clamp(0, 1)

# Test the model
if __name__ == "__main__":
    # Assuming you have already defined the Diffusion class somewhere
    model = Diffusion()

    # Move the model to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Test forward pass
    test_input = torch.rand(1, 1, 16, 16).to(device)  # Move input to same device as model
    timestep = 500
    output = model(test_input, timestep)
    print("Forward pass output shape:", output.shape)  # Should be [1, 1, 128, 128]

