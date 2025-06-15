import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 特征融合模块 ---
class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.conv1x1 = nn.Conv2d(1, 16, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(1, 48, kernel_size=5, padding=2)

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

# --- Ec-Net 主体 ---
class EcNet(nn.Module):
    def __init__(self):
        super(EcNet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)  # 从16x16到128x128
        self.feature_fusion = FeatureFusion()
        self.dilated_blocks = nn.Sequential(
            DilatedConvBlock(),
            DilatedConvBlock(),
            DilatedConvBlock(),
            DilatedConvBlock()
        )
        self.residual_reconstruct = nn.Conv2d(96, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample(x)  # (B, 1, 128, 128)
        fused = self.feature_fusion(x)  # (B, 96, 128, 128)
        features = self.dilated_blocks(fused)  # (B, 96, 128, 128)

        # 🔁 加入 long skip connection（特征融合模块输出 + 膨胀卷积模块输出）
        combined = features + fused

        residual = self.residual_reconstruct(combined)  # (B, 1, 128, 128)
        return residual  # 输出残差图像，可加回初始图得到最终图像


# 测试脚本
if __name__ == '__main__':
    b = 1
    input_size = 16
    x = torch.rand(b, 1, input_size, input_size)  # 输入: (B, 1, 16, 16)
    model = EcNet()

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
