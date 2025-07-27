import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class HybridAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        residual = x
        x = self.ca(x)
        x = self.sa(x)
        x = self.conv(x)
        return residual + x

class HASRN(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入特征提取
        self.init_conv = nn.Conv2d(1, 64, 3, padding=1)
        
        # 注意力块堆叠
        self.attention_blocks = nn.Sequential(
            *[HybridAttentionBlock(64) for _ in range(8)]
        )
        
        # 多阶段上采样
        self.upscale = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),  # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),  # 64x64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),  # 128x128
            nn.ReLU(inplace=True)
        )
        
        # 输出层
        self.output = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.init_conv(x)          # [B,64,16,16]
        x = self.attention_blocks(x)   # [B,64,16,16]
        x = self.upscale(x)            # [B,64,128,128]
        return self.output(x)           # [B,1,128,128]

def test_model(model_class):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建模型
    model = model_class().to(device)
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数总量: {total_params/1e6:.2f}M")
    
    # 测试输入
    x = torch.rand(4, 1, 16, 16).to(device)
    
    # 前向传播测试
    out = model(x)
    expected_shape = (4, 1, 128, 128)
    assert out.shape == expected_shape, \
        f"形状错误: 预期 {expected_shape}, 实际 {out.shape}"
    print(f"✅ 前向传播: {x.shape} → {out.shape}")
    
    # 梯度测试
    target = torch.rand_like(out)
    loss = F.l1_loss(out, target)
    loss.backward()
    
    # 梯度分析
    max_grad = max(p.grad.abs().max().item() 
                for p in model.parameters() if p.grad is not None)
    print(f"最大梯度值: {max_grad:.6f}")
    
    # FLOPs统计
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(x,), verbose=False)
        print(f"FLOPs: {flops/1e9:.2f}G")
    except ImportError:
        print("安装thop: pip install thop")
    
    # 内存分析
    try:
        mem = torch.cuda.memory_allocated(device) / 1024**2
        print(f"GPU内存: {mem:.2f}MB")
    except:
        pass

if __name__ == '__main__':
    print("="*60)
    print("测试 HASRN 网络:")
    test_model(HASRN)