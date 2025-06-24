import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Updated UNet architecture for 128x128 input images
class Unet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(Unet, self).__init__()
        self.mlp_upsample = MLPUpSample(input_channels=1, output_channels=1, input_size=16, output_size=128)
        self.down1 = self._block(in_channels, 64)
        self.down2 = self._block(64, 128)
        self.down3 = self._block(128, 256)
        self.down4 = self._block(256, 512)  # Additional downsampling block
        self.up1 = self._block(512, 256)
        self.up2 = self._block(256, 128)
        self.up3 = self._block(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
    
    def forward(self, x, cond):
        # Resize condition to match input size if needed (condition is 16x16, resize to 128x128)
        cond = self.mlp_upsample(cond)
        # Concatenate the condition (128x128) with the input image (128x128)
        x = torch.cat((x, cond), dim=1)  # Shape: [batch_size, 2, 128, 128] (input + condition)

        x = self.down1(x)  # Shape: [batch_size, 64, 128, 128]
        x = self.down2(x)  # Shape: [batch_size, 128, 64, 64]
        x = self.down3(x)  # Shape: [batch_size, 256, 32, 32]
        x = self.down4(x)  # Shape: [batch_size, 512, 16, 16]
        x = self.up1(x)    # Shape: [batch_size, 256, 32, 32]
        x = self.up2(x)    # Shape: [batch_size, 128, 64, 64]
        x = self.up3(x)    # Shape: [batch_size, 64, 128, 128]
        return self.final_conv(x)  # Final output: [batch_size, 1, 128, 128] (output image)

# Conditional Diffusion Model (CDEIT)
class CDEIT(nn.Module):
    def __init__(self, unet_model=Unet(in_channels=2, out_channels=1), num_timesteps=1000):
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
    unet_model = Unet(in_channels=2, out_channels=1)  # In_channels=2 (image + condition)
    cdeit_model = CDEIT(unet_model=unet_model)
    
    # Run a forward pass to get the denoised image
    denoised_image, noise = cdeit_model(x, cond)

    # Print the shapes of the noisy image and denoised image
    print("Noisy Image Shape:", x.shape)  # Expected: [batch_size, 1, 128, 128]
    print("Denoised Image Shape:", denoised_image.shape)  # Expected: [batch_size, 1, 128, 128]