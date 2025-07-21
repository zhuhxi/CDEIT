import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    """带残差连接的深度卷积块"""
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True))
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual

class DenseBlock(nn.Module):
    """密集连接块增强特征复用"""
    def __init__(self, in_ch, growth_rate=32, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_ch
        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(ch),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ch, growth_rate, 3, padding=1)
            )
            self.layers.append(layer)
            ch += growth_rate
        self.final_conv = nn.Conv2d(ch, in_ch, 1)  # 特征压缩

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return self.final_conv(torch.cat(features, dim=1))

class DeepUNet_PS8x_16to128(nn.Module):
    def __init__(self):
        super().__init__()
        # ----------------- 深度编码器 -----------------
        # 输入: [B,1,16,16]
        self.enc1 = nn.Sequential(
            ResidualConvBlock(1, 64),
            DenseBlock(64, growth_rate=16, num_layers=3))
        self.pool1 = nn.MaxPool2d(2)  # [B,64,8,8]
        
        self.enc2 = nn.Sequential(
            ResidualConvBlock(64, 128, dilation=2),
            DenseBlock(128, growth_rate=32, num_layers=3))
        self.pool2 = nn.MaxPool2d(2)  # [B,128,4,4]
        
        self.enc3 = nn.Sequential(
            ResidualConvBlock(128, 256, dilation=2),
            ResidualConvBlock(256, 256),
            DenseBlock(256, growth_rate=32, num_layers=4)
        )
        self.pool3 = nn.MaxPool2d(2)  # [B,256,2,2]
        
        # ----------------- 瓶颈层（带空洞卷积）-----------------
        self.bottleneck = nn.Sequential(
            ResidualConvBlock(256, 512, dilation=3),
            ResidualConvBlock(512, 512),
            DenseBlock(512, growth_rate=64, num_layers=4),
            nn.Conv2d(512, 256, 1)  # 通道压缩
        )  # [B,256,2,2]
        
        # ----------------- 深度解码器 -----------------
        # 上采样1: 2x2 -> 4x4
        self.up1 = nn.Sequential(
            nn.Conv2d(256, 256 * 4, 3, padding=1),
            nn.PixelShuffle(2),  # [B,256,4,4]
            nn.LeakyReLU(0.2)
        )
        self.dec1 = nn.Sequential(
            ResidualConvBlock(256 + 256, 256),  # 跳跃连接
            DenseBlock(256, growth_rate=32, num_layers=3)
        )
        
        # 上采样2: 4x4 -> 8x8
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128 * 4, 3, padding=1),
            nn.PixelShuffle(2),  # [B,128,8,8]
            nn.LeakyReLU(0.2)
        )
        self.dec2 = nn.Sequential(
            ResidualConvBlock(128 + 128, 128),  # 跳跃连接
            DenseBlock(128, growth_rate=32, num_layers=2)
        )
        
        # 上采样3: 8x8 -> 16x16
        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),  # [B,64,16,16]
            nn.LeakyReLU(0.2)
        )
        self.dec3 = nn.Sequential(
            ResidualConvBlock(64 + 64, 64),  # 跳跃连接
            DenseBlock(64, growth_rate=16, num_layers=2)
        )
        
        # ----------------- 最终上采样 (16x16->128x128) -----------------
        self.final_up = nn.Sequential(
            nn.Conv2d(64, 64 * (8**2), 3, padding=1),  # 64*64=4096
            nn.PixelShuffle(8),  # [B,64,128,128]
            nn.LeakyReLU(0.2)
        )
        self.final_conv = nn.Sequential(
            ResidualConvBlock(64, 32),
            nn.Conv2d(32, 1, 1)  # 输出通道
        )

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)       # [B,64,16,16]
        p1 = self.pool1(e1)     # [B,64,8,8]
        e2 = self.enc2(p1)      # [B,128,8,8]
        p2 = self.pool2(e2)     # [B,128,4,4]
        e3 = self.enc3(p2)      # [B,256,4,4]
        p3 = self.pool3(e3)     # [B,256,2,2]
        
        # 瓶颈层
        b = self.bottleneck(p3) # [B,256,2,2]
        
        # 解码器
        d1 = self.up1(b)        # [B,256,4,4]
        d1 = torch.cat([d1, e3], dim=1)  # [B,512,4,4]
        d1 = self.dec1(d1)      # [B,256,4,4]
        
        d2 = self.up2(d1)       # [B,128,8,8]
        d2 = torch.cat([d2, e2], dim=1)  # [B,256,8,8]
        d2 = self.dec2(d2)      # [B,128,8,8]
        
        d3 = self.up3(d2)       # [B,64,16,16]
        d3 = torch.cat([d3, e1], dim=1)  # [B,128,16,16]
        d3 = self.dec3(d3)      # [B,64,16,16]
        
        # 最终上采样
        out = self.final_up(d3) # [B,64,128,128]
        return self.final_conv(out)  # [B,1,128,128]

# ===================== 增强测试脚本 =====================
if __name__ == '__main__':
    # 配置测试参数
    batch_size = 4
    input_size = 16
    target_size = 128
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model = DeepUNet_PS8x_16to128().to(device)
    
    # 计算模型参数总量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"💎 总参数量: {total_params/1e6:.2f}M")
    
    # 测试输入
    x = torch.rand(batch_size, 1, input_size, input_size).to(device)
    
    print("="*60)
    print(f"🔍 测试深度模型: {model.__class__.__name__}")
    print(f"📦 输入尺寸: {x.shape} (batch={batch_size}, channels=1, {input_size}x{input_size})")
    
    # 维度验证
    try:
        out = model(x)
        expected_shape = (batch_size, 1, target_size, target_size)
        assert out.shape == expected_shape, \
            f"❌ 输出尺寸错误: 预期 {expected_shape}, 实际 {out.shape}"
        print(f"✅ 前向传播验证通过: {x.shape} → {out.shape}")
    except Exception as e:
        print(f"❌ 前向传播失败: {str(e)}")
        raise

    # 计算量分析
    try:
        from thop import profile
        flops, params = profile(model, inputs=(x,), verbose=False)
        print(f"📊 FLOPs: {flops/1e9:.2f}G | 参数量: {params/1e6:.2f}M")
    except ImportError:
        print("⚠️ 安装thop以获取计算量统计: pip install thop")
    
    # 内存占用分析
    try:
        mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
        print(f"💾 GPU内存占用: {mem_alloc:.2f} MB (前向传播后)")
    except:
        pass
    
    # 梯度检查
    try:
        loss = F.l1_loss(out, torch.randn_like(out))
        loss.backward()
        print("✅ 梯度反向传播成功")
        
        # 检查梯度爆炸
        max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
        print(f"📈 最大梯度值: {max_grad:.4f}")
    except Exception as e:
        print(f"❌ 梯度反向传播失败: {str(e)}")
    
    # 关键特征图尺寸验证
    print("\n🔬 深度特征验证:")
    with torch.no_grad():
        e1 = model.enc1(x)
        print(f"  enc1 输出: {e1.shape} (预期 [B,64,16,16])")
        e3 = model.enc3(model.pool2(model.enc2(model.pool1(e1))))
        print(f"  enc3 输出: {e3.shape} (预期 [B,256,4,4])")
        b = model.bottleneck(model.pool3(e3))
        print(f"  bottleneck 输出: {b.shape} (预期 [B,256,2,2])")
        d1 = model.dec1(torch.cat([model.up1(b), e3], dim=1))
        print(f"  dec1 输出: {d1.shape} (预期 [B,256,4,4])")
        
    print("="*60)
