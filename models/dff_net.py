import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseFeatureFusion(nn.Module):
    def __init__(self, in_ch, growth_rate=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, growth_rate, 3, padding=1),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch + growth_rate, growth_rate, 3, padding=1),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch + 2*growth_rate, growth_rate, 3, padding=1),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch + 3*growth_rate, in_ch, 1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(torch.cat([x, f1], 1))
        f3 = self.conv3(torch.cat([x, f1, f2], 1))
        return self.conv4(torch.cat([x, f1, f2, f3], 1))

class DFF_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取
        self.feat_extract = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            DenseFeatureFusion(64),
            DenseFeatureFusion(64)
        )
        
        # 高倍率上采样
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * 64, 3, padding=1),  # 64*(8^2)
            nn.PixelShuffle(8),
            nn.ReLU(inplace=True)
        )
        
        # 残差增强重建
        self.reconstruct = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        x = self.feat_extract(x)  # [B,64,16,16]
        x = self.upsample(x)      # [B,64,128,128]
        return self.reconstruct(x)
    

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
    print("测试 DFF_Net 网络:")
    test_model(DFF_Net)