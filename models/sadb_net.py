import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractorS(nn.Module):
    def __init__(self):
        super(FeatureExtractorS, self).__init__()
        
        # 卷积层 + 批归一化 + ReLU 激活
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 输入: (B, 1, 128, 128) → 输出: (B, 16, 128, 128)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # (B, 16, 128, 128) → (B, 32, 128, 128)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        
        self.pool1 = nn.MaxPool2d(2, 2)  # (B, 32, 128, 128) → (B, 32, 64, 64)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (B, 32, 64, 64) → (B, 64, 64, 64)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        
        self.pool2 = nn.MaxPool2d(2, 2)  # (B, 64, 64, 64) → (B, 64, 32, 32)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # (B, 64, 32, 32) → (B, 128, 32, 32)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        
        self.pool3 = nn.MaxPool2d(2, 2)  # (B, 128, 32, 32) → (B, 128, 16, 16)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))  # (B, 1, 128, 128) → (B, 16, 128, 128)
        x = self.relu2(self.bn2(self.conv2(x)))  # (B, 16, 128, 128) → (B, 32, 128, 128)
        x = self.pool1(x)  # (B, 32, 128, 128) → (B, 32, 64, 64)
        
        x = self.relu3(self.bn3(self.conv3(x)))  # (B, 32, 64, 64) → (B, 64, 64, 64)
        x = self.pool2(x)  # (B, 64, 64, 64) → (B, 64, 32, 32)
        
        x = self.relu4(self.bn4(self.conv4(x)))  # (B, 64, 32, 32) → (B, 128, 32, 32)
        x = self.pool3(x)  # (B, 128, 32, 32) → (B, 128, 16, 16)

        return x
    

class FeatureExtractorC(nn.Module):
    def __init__(self):
        super(FeatureExtractorC, self).__init__()

        # 输入: (B, 16, 16)
        self.conv1 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()

        self.pool1 = nn.MaxPool1d(2)  # (B, 32, 16) → (B, 32, 8)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()

        self.pool2 = nn.MaxPool1d(2)  # (B, 64, 8) → (B, 64, 4)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()

        self.pool3 = nn.MaxPool1d(2)  # (B, 128, 4) → (B, 128, 2)

        self.conv5 = nn.Conv1d(128, 16, kernel_size=3, padding=1)  # → (B, 16, 2)

    def forward(self, x):
        # 输入 x: (B, 1, 16, 16)
        # 转为 (B, 16, 16)，也就是每个样本是 16 个序列，每个序列长度为 16
        x = x.squeeze(1)  # → (B, 16, 16)

        x = self.relu1(self.bn1(self.conv1(x)))   # (B, 16, 16)
        x = self.relu2(self.bn2(self.conv2(x)))   # (B, 32, 16)
        x = self.pool1(x)                         # (B, 32, 8)

        x = self.relu3(self.bn3(self.conv3(x)))   # (B, 64, 8)
        x = self.pool2(x)                         # (B, 64, 4)

        x = self.relu4(self.bn4(self.conv4(x)))   # (B, 128, 4)
        x = self.pool3(x)                         # (B, 128, 2)

        x = self.conv5(x)                         # (B, 16, 2)

        return x

class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels - out_channels, in_channels - out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)  # in_channels because of cat

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FCUNet(nn.Module):
    def __init__(self):
        super(FCUNet, self).__init__()

        # Upsample from 16x16 → 128x128
        self.upsample_to_128 = nn.Sequential(
            nn.ConvTranspose2d(1, 8, kernel_size=4, stride=2, padding=1),  # 16x16 → 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),  # 32x32 → 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),  # 64x64 → 128x128
        )

        # UNet encoder
        self.inc = DoubleConv(1, 64)          # 128x128
        self.down1 = Down(64, 128)            # 64x64
        self.down2 = Down(128, 256)           # 32x32
        self.down3 = Down(256, 512)           # 16x16

        # UNet decoder
        self.up1 = Up(512 + 256, 256)         # 32x32
        self.up2 = Up(256 + 128, 128)         # 64x64
        self.up3 = Up(128 + 64, 64)           # 128x128

        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.upsample_to_128(x)           # (B, 1, 16, 16) → (B, 1, 128, 128)

        x1 = self.inc(x)                      # (B, 64, 128, 128)
        x2 = self.down1(x1)                   # (B, 128, 64, 64)
        x3 = self.down2(x2)                   # (B, 256, 32, 32)
        x4 = self.down3(x3)                   # (B, 512, 16, 16)

        x = self.up1(x4, x3)                  # (B, 256, 32, 32)
        x = self.up2(x, x2)                   # (B, 128, 64, 64)
        x = self.up3(x, x1)                   # (B, 64, 128, 128)

        x = self.outc(x)                      # (B, 1, 128, 128)
        return x

class FeatureConditionProjector(nn.Module):
    def __init__(self):
        super().__init__()

        input_dim = 128 * 16 * 16 + 16 * 2  # 
        output_dim = 128 * 128              # 16384

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 10000),
            nn.ReLU(inplace=True),
            nn.Linear(10000, 8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, output_dim),
        )

    def forward(self, feat, cond):
        """
        feat: B x 128 x 16 x 16
        cond: B x 16 x 2
        """
        B = feat.size(0)

        # Flatten both inputs
        feat_flat = feat.view(B, -1)    # B x (128 x 16 x 16)
        cond_flat = cond.view(B, -1)    # B x 32

        x = torch.cat([feat_flat, cond_flat], dim=1)  # B x (128 x 16 x 16 + 32)

        x = self.mlp(x)                 # B x 16384

        x = x.view(B, 1, 128, 128)      # B x 1 x 128 x 128
        return x
    
class SADB_Net(nn.Module):
    def __init__(self):
        super(SADB_Net, self).__init__()
        self.feature_c = FeatureExtractorC()
        self.unet = FCUNet()
        self.feature_s = FeatureExtractorS()
        self.projector = FeatureConditionProjector()

    def forward(self, x):
        """
        输入:
            x: (B, 1, 16, 16)

        输出:
            out: (B, 1, 128, 128)
        """
        # 条件特征提取
        cond_feat = self.feature_c(x)  # (B, 16, 2)

        # 上采样 + 图像特征提取
        upsampled = self.unet(x)                      # (B, 1, 128, 128)
        semantic_feat = self.feature_s(upsampled)     # (B, 128, 16, 16)

        # MLP 融合
        out = self.projector(semantic_feat, cond_feat)  # (B, 1, 128, 128)

        # Apply Sigmoid to output
        out = torch.sigmoid(out)  # 将输出归一化到 [0, 1] 范围
        return out


if __name__ == '__main__':
    x = torch.randn(1, 1, 16, 16)
    model = SADB_Net()
    out = model(x)
    print(f"Output shape: {out.shape}")  # → torch.Size([1, 1, 128, 128])
    from nni.compression.utils.counter import count_flops_params
    flops, params, _ = count_flops_params(model, x=(x,))
    print(f"FLOPs: {flops / 1e6:.2f} MFLOPs | Params: {params / 1e6:.2f} M")