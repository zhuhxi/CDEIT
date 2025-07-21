import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.ELU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        identity = x
        x = self.conv(x)
        gate = self.gate(x)
        x = x * gate
        return identity + x

class ERSPN(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始特征提取
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.GroupNorm(4, 64),
            nn.ELU()
        )
        
        # 残差块组
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(8)]
        )
        
        # 多阶段上采样
        self.up1 = nn.Sequential(
            nn.Conv2d(64, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 32x32
            nn.ELU()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 64x64
            nn.ELU()
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(64, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 128x128
            nn.ELU()
        )
        
        # 输出层
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.init_conv(x)     # [B,64,16,16]
        x = self.res_blocks(x)    # [B,64,16,16]
        
        # 渐进上采样
        x = self.up1(x)           # [B,64,32,32]
        x = self.up2(x)           # [B,64,64,64]
        x = self.up3(x)           # [B,64,128,128]
        
        return self.final_conv(x)  # [B,1,128,128]


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
    print("测试 ERSPN 网络:")
    test_model(ERSPN)
