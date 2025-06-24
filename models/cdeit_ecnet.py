import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 特征融合模块 ---
class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.conv1x1 = nn.Conv2d(2, 16, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(2, 48, kernel_size=5, padding=2)

    def forward(self, x):
        f1 = self.conv1x1(x)
        f2 = self.conv3x3(x)
        f3 = self.conv5x5(x)
        return torch.cat([f1, f2, f3], dim=1)  # 输出 (B, 96, H, W)

# --- 单个膨胀卷积Block ---
class DilatedConvBlock(nn.Module):
    def __init__(self):
        super(DilatedConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class MLPUpSample(nn.Module):
    def __init__(self, input_channels, output_channels, input_size, output_size):
        super(MLPUpSample, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # 输入展平后的维度
        self.input_dim = input_channels * input_size * input_size
        self.output_dim = output_channels * output_size * output_size
        
        # 定义 MLP 模块
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim)  # 输出层
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # 将图像展平为一维向量
        x = x.view(B, -1)  # 展平为 (B, C * H * W)
        
        # 通过 MLP 进行特征转换
        x = self.mlp(x)  # 输出 (B, output_channels * output_size * output_size)
        
        # 将输出重塑为目标大小
        x = x.view(B, -1, self.output_size, self.output_size)  # 还原为 (B, output_channels, output_size, output_size)
        
        return x

class EcNet(nn.Module):
    def __init__(self):
        super(EcNet, self).__init__()
        # 使用 MLP 上采样
        self.mlp_upsample = MLPUpSample(input_channels=1, output_channels=1, input_size=16, output_size=128)
        self.feature_fusion = FeatureFusion()
        self.dilated_blocks = nn.Sequential(
            DilatedConvBlock(),
            DilatedConvBlock(),
            DilatedConvBlock(),
            DilatedConvBlock()
        )
        self.residual_reconstruct = nn.Conv2d(96, 1, kernel_size=3, padding=1)

    def forward(self, x, cond):
        # Resize condition to match input size if needed (condition is 16x16, resize to 128x128)
        cond = self.mlp_upsample(cond)
        x = torch.cat((x, cond), dim=1)
        fused = self.feature_fusion(x)  # (B, 96, 128, 128)
        features = self.dilated_blocks(fused)  # (B, 96, 128, 128)

        # 🔁 加入 long skip connection（特征融合模块输出 + 膨胀卷积模块输出）
        combined = features + fused

        residual = self.residual_reconstruct(combined)  # (B, 1, 128, 128)
        return residual  # 输出残差图像，可加回初始图得到最终图像
    

# Conditional Diffusion Model (CDEIT)
class CDEIT(nn.Module):
    def __init__(self, unet_model=EcNet(), num_timesteps=1000):
        super(CDEIT, self).__init__()
        self.unet = unet_model
        self.num_timesteps = num_timesteps
        self.beta_schedule = torch.linspace(1e-5, 0.02, num_timesteps)  # Linear beta schedule for simplicity

    def forward(self, x, cond):
        t = torch.randint(0, self.num_timesteps, (x.size(0),))  # Ensure `t` is on the same device as `x`
        
        noise = torch.randn_like(x, device=x.device)  # Shape: [batch_size, 1, 128, 128] (random noise)
        # beta_t = self.beta_schedule[t].to(x.device)  # Move beta_t to the same device as `x`

        # beta_t_expanded = beta_t.view(-1, 1, 1, 1)  # Shape: [batch_size, 1, 1, 1] (broadcastable to noise)
        # noise = noise * beta_t_expanded  # Broadcasting beta_t_expanded over the noise tensor
        x_t = x + noise  # Shape: [batch_size, 1, 128, 128] (noisy image)

        denoised_image = self.unet(x_t, cond)  # Shape: [batch_size, 1, 128, 128] (denoised image)
        return denoised_image, noise

    def get_loss(self, x, cond):
        recon, noise = self.forward(x, cond)
        loss = F.mse_loss(recon, x)  # Loss between the denoised image and the original image
        return loss
    
# Example usage to test CDEIT (Conditional Diffusion Model)
if __name__ == "__main__":
    batch_size = 16
    x = torch.randn(batch_size, 1, 128, 128)  # Example input image: [batch_size, 1, 128, 128]
    cond = torch.randn(batch_size, 1, 16, 16)  # Example condition: [batch_size, 1, 16, 16]
    
    # Initialize the UNet model and the CDEIT model
    unet_model = EcNet()  # In_channels=2 (image + condition)
    cdeit_model = CDEIT(unet_model=unet_model)
    
    # Run a forward pass to get the denoised image
    denoised_image, noise = cdeit_model(x, cond)

    # Print the shapes of the noisy image and denoised image
    print("Noisy Image Shape:", x.shape)  # Expected: [batch_size, 1, 128, 128]
    print("Denoised Image Shape:", denoised_image.shape)  # Expected: [batch_size, 1, 128, 128]