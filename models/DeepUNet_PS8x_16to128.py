import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    """带梯度稳定技术的残差块"""
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        # 使用LayerNorm替代BatchNorm以获得更稳定的训练
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, out_ch),  # 使用GroupNorm替代BatchNorm
            nn.ELU(inplace=True)  # 使用ELU激活函数替代LeakyReLU
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, out_ch),
            nn.ELU(inplace=True))
        
        # 添加梯度缩放层
        self.grad_scale = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            nn.Sigmoid()
        )
        
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        
        # 应用梯度缩放
        scale_factor = self.grad_scale(x)
        x = x * scale_factor
        
        return x + residual

class DenseBlock(nn.Module):
    """带梯度控制的密集连接块"""
    def __init__(self, in_ch, growth_rate=32, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_ch
        
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.GroupNorm(4, ch),  # 使用GroupNorm
                nn.ELU(),  # 使用ELU激活函数
                nn.Conv2d(ch, growth_rate, 3, padding=1),
                
                # 添加梯度门控
                nn.Sequential(
                    nn.Conv2d(growth_rate, growth_rate, 1),
                    nn.Sigmoid()
                )
            )
            self.layers.append(layer)
            ch += growth_rate
        
        # 最终卷积层添加权重归一化
        self.final_conv = nn.utils.weight_norm(
            nn.Conv2d(ch, in_ch, 1), name='weight'
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            
            # 应用梯度门控
            conv_out, gate = new_feat[:, :-1], new_feat[:, -1:]
            gated_feat = conv_out * gate
            
            features.append(gated_feat)
        
        return self.final_conv(torch.cat(features, dim=1))

class DeepUNet_PS8x_16to128(nn.Module):
    def __init__(self):
        super().__init__()
        # ----------------- 深度编码器 -----------------
        self.enc1 = nn.Sequential(
            ResidualConvBlock(1, 64),
            DenseBlock(64, growth_rate=16, num_layers=3)
        )
        self.pool1 = nn.AvgPool2d(2)  # 使用AvgPool替代MaxPool
        
        self.enc2 = nn.Sequential(
            ResidualConvBlock(64, 128, dilation=2),
            DenseBlock(128, growth_rate=32, num_layers=3)
        )
        self.pool2 = nn.AvgPool2d(2)
        
        self.enc3 = nn.Sequential(
            ResidualConvBlock(128, 256, dilation=2),
            ResidualConvBlock(256, 256),
            DenseBlock(256, growth_rate=32, num_layers=4)
        )
        self.pool3 = nn.AvgPool2d(2)
        
        # ----------------- 瓶颈层 -----------------
        self.bottleneck = nn.Sequential(
            ResidualConvBlock(256, 512, dilation=3),
            ResidualConvBlock(512, 512),
            DenseBlock(512, growth_rate=64, num_layers=4),
            nn.Conv2d(512, 256, 1)
        )
        
        # ----------------- 深度解码器 -----------------
        # 上采样1: 2x2 -> 4x4
        self.up1 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(256, 256 * 4, 3, padding=1)),
            nn.PixelShuffle(2),
            nn.ELU()
        )
        self.dec1 = nn.Sequential(
            ResidualConvBlock(256 + 256, 256),
            DenseBlock(256, growth_rate=32, num_layers=3)
        )
        
        # 上采样2: 4x4 -> 8x8
        self.up2 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(256, 128 * 4, 3, padding=1)),
            nn.PixelShuffle(2),
            nn.ELU()
        )
        self.dec2 = nn.Sequential(
            ResidualConvBlock(128 + 128, 128),
            DenseBlock(128, growth_rate=32, num_layers=2)
        )
        
        # 上采样3: 8x8 -> 16x16
        self.up3 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(128, 64 * 4, 3, padding=1)),
            nn.PixelShuffle(2),
            nn.ELU()
        )
        self.dec3 = nn.Sequential(
            ResidualConvBlock(64 + 64, 64),
            DenseBlock(64, growth_rate=16, num_layers=2)
        )
        
        # ----------------- 最终上采样 -----------------
        self.final_up = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(64, 64 * (8**2), 3, padding=1)),
            nn.PixelShuffle(8),
            nn.ELU()
        )
        self.final_conv = nn.Sequential(
            ResidualConvBlock(64, 32),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # 瓶颈层
        b = self.bottleneck(p3)
        
        # 解码器
        d1 = self.up1(b)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3)
        
        # 最终上采样
        out = self.final_up(d3)
        return self.final_conv(out)

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
    
    # 梯度稳定性测试
    print("\n🔬 梯度稳定性测试:")
    try:
        # 创建优化器并设置梯度裁剪
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        
        # 模拟训练过程
        max_grad_norm = 1.0  # 梯度裁剪阈值
        for i in range(10):  # 模拟10个训练步骤
            # 前向传播
            out = model(x)
            target = torch.randn_like(out)
            loss = F.l1_loss(out, target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 计算梯度范数
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # 应用梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # 参数更新
            optimizer.step()
            
            # 打印梯度信息
            clipped_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    clipped_norm += p.grad.data.norm(2).item() ** 2
            clipped_norm = clipped_norm ** 0.5
            
            print(f"步骤 {i+1}: 损失={loss.item():.4f} | "
                  f"梯度范数={total_norm:.2f} | "
                  f"裁剪后={clipped_norm:.2f} | "
                  f"缩放比={clipped_norm/max(1e-6, total_norm):.2f}")
        
        print("✅ 梯度稳定性测试通过")
    except Exception as e:
        print(f"❌ 梯度测试失败: {str(e)}")
    
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
