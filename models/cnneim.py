import torch
import torch.nn as nn

class CNN_EIM(nn.Module):
    def __init__(self):
        super(CNN_EIM, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # (B, 1, 16, 16) → (B, 64, 16, 16)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu1 = nn.PReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (B, 64, 16, 16) → (B, 64, 8, 8)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # (B, 64, 8, 8) → (B, 128, 8, 8)
        self.bn2 = nn.BatchNorm2d(128)
        self.prelu2 = nn.PReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (B, 128, 8, 8) → (B, 128, 4, 4)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # (B, 128, 4, 4) → (B, 256, 4, 4)
        self.bn3 = nn.BatchNorm2d(256)
        self.prelu3 = nn.PReLU()
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # (B, 256, 4, 4) → (B, 256, 4, 4)
        self.bn4 = nn.BatchNorm2d(256)
        self.prelu4 = nn.PReLU()
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # (B, 256, 4, 4) → (B, 256, 4, 4)
        self.bn5 = nn.BatchNorm2d(256)
        self.prelu5 = nn.PReLU()
        
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)  # (B, 256, 4, 4) → (B, 128, 8, 8)
        self.bn_deconv1 = nn.BatchNorm2d(128)
        self.leakyrelu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)  # (B, 128, 8, 8) → (B, 64, 16, 16)
        self.bn_deconv2 = nn.BatchNorm2d(64)
        self.leakyrelu2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)  # (B, 64, 16, 16) → (B, 32, 32, 32)
        self.bn_deconv3 = nn.BatchNorm2d(32)
        self.leakyrelu3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, padding=0)  # (B, 32, 32, 32) → (B, 1, 64, 64)
        self.bn_deconv4 = nn.BatchNorm2d(1)
        self.leakyrelu4 = nn.LeakyReLU(0.2, inplace=True)
        
        # Add one more deconvolution layer to expand to (B, 1, 128, 128)
        self.deconv5 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, padding=0)  # (B, 1, 64, 64) → (B, 1, 128, 128)
        self.bn_deconv5 = nn.BatchNorm2d(1)
        self.tanh = nn.Tanh()  # Output layer

    def forward(self, x):
        # Convolutional layers with shape changes
        x = self.conv1(x)  # (B, 1, 16, 16) → (B, 64, 16, 16)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.maxpool1(x)  # (B, 64, 16, 16) → (B, 64, 8, 8)
        
        x = self.conv2(x)  # (B, 64, 8, 8) → (B, 128, 8, 8)
        x = self.bn2(x)
        x = self.prelu2(x)
        x = self.maxpool2(x)  # (B, 128, 8, 8) → (B, 128, 4, 4)
        
        x = self.conv3(x)  # (B, 128, 4, 4) → (B, 256, 4, 4)
        x = self.bn3(x)
        x = self.prelu3(x)
        
        x = self.conv4(x)  # (B, 256, 4, 4) → (B, 256, 4, 4)
        x = self.bn4(x)
        x = self.prelu4(x)
        
        x = self.conv5(x)  # (B, 256, 4, 4) → (B, 256, 4, 4)
        x = self.bn5(x)
        x = self.prelu5(x)
        
        # Deconvolutional layers with shape changes
        x = self.deconv1(x)  # (B, 256, 4, 4) → (B, 128, 8, 8)
        x = self.bn_deconv1(x)
        x = self.leakyrelu1(x)
        
        x = self.deconv2(x)  # (B, 128, 8, 8) → (B, 64, 16, 16)
        x = self.bn_deconv2(x)
        x = self.leakyrelu2(x)
        
        x = self.deconv3(x)  # (B, 64, 16, 16) → (B, 32, 32, 32)
        x = self.bn_deconv3(x)
        x = self.leakyrelu3(x)
        
        x = self.deconv4(x)  # (B, 32, 32, 32) → (B, 1, 64, 64)
        x = self.bn_deconv4(x)
        x = self.leakyrelu4(x)
        
        # Final layer to upscale to (B, 1, 128, 128)
        x = self.deconv5(x)  # (B, 1, 64, 64) → (B, 1, 128, 128)
        x = self.bn_deconv5(x)
        x = self.tanh(x)  # Output layer

        return x

# 测试脚本
if __name__ == '__main__':
    b = 1
    input_size = 16
    x = torch.rand(b, 1, input_size, input_size)  # 输入: (B, 1, 16, 16)
    model = CNN_EIM()

    # --- Shape 测试 ---
    try:
        out = model(x)
        print(f"✅ Forward Pass Success: {x.shape} → {out.shape}")
    except Exception as e:
        print(f"❌ Forward Failed: {e}")

    # --- FLOPs 和 参数统计 ---
    try:
        from nni.compression.utils.counter import count_flops_params
        flops, params, _ = count_flops_params(model, x=(x,))
        print(f"📊 FLOPs: {flops / 1e6:.2f} MFLOPs | Params: {params / 1e6:.2f} M")
    except ImportError:
        print("⚠️ NNI not installed. Run: pip install nni")
