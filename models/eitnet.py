# models/eitnet.py
import torch
import torch.nn as nn

class EITNet(nn.Module):
    def __init__(self):
        super(EITNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),    # (B, 1, 16, 16) → (B, 32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (B, 32, 16, 16) → (B, 64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (B, 64, 8, 8) → (B, 128, 4, 4)
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (B, 128, 4, 4) → (B, 64, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # (B, 64, 8, 8) → (B, 32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # (B, 32, 16, 16) → (B, 16, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),    # (B, 16, 32, 32) → (B, 8, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),     # (B, 8, 64, 64) → (B, 1, 128, 128)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 测试脚本
if __name__ == '__main__':
    b = 1
    input_size = 16
    x = torch.rand(b, 1, input_size, input_size)  # 输入: (B, 1, 16, 16)
    model = EITNet()

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