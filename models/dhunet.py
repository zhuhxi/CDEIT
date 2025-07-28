import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

# ---------------- DeformableConvBlock ----------------
class DeformableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DeformableConvBlock, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding

        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            padding=padding
        )

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        offset = self.offset_conv(x)
        out = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            padding=(self.padding, self.padding)
        )
        return out

# ---------------- SEBlock ----------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ---------------- CoordAttention ----------------
class CoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h_attn = self.conv_h(y[:, :, :h, :])
        x_w_attn = self.conv_w(y[:, :, h:, :]).permute(0, 1, 3, 2)
        a = torch.sigmoid(x_h_attn * x_w_attn)
        return x * a

# ---------------- HyperConv2D ----------------
class HyperConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super(HyperConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        coords = self._make_coords(kernel_size)
        self.register_buffer("coords", coords)

        self.hypernet = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, in_channels * out_channels)
        )

    def _make_coords(self, k):
        x = torch.linspace(-1, 1, steps=k)
        y = torch.linspace(-1, 1, steps=k)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([grid_x, grid_y], dim=-1)
        return coords.view(-1, 2)

    def forward(self, x):
        B, C_in, H, W = x.shape
        weights = self.hypernet(self.coords)
        weights = weights.view(self.kernel_size ** 2, self.in_channels, self.out_channels)
        kernel = weights.permute(2, 1, 0).reshape(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )
        return F.conv2d(x, kernel, bias=None, stride=self.stride, padding=self.padding)

# ---------------- MultiHeadAttention ----------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.norm_Q = nn.LayerNorm(d_model)
        self.norm_K = nn.LayerNorm(d_model)
        self.norm_V = nn.LayerNorm(d_model)

        self.fc_out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        q, k, v = x
        B, L, H, W = q.shape
        d_model = H * W
        Q = self.norm_Q(self.W_Q(q.view(B, L, d_model)))
        K = self.norm_K(self.W_K(k.view(B, L, d_model)))
        V = self.norm_V(self.W_V(v.view(B, L, d_model)))

        Q = Q.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        x = torch.matmul(attn_weights, V)
        x = x.transpose(1, 2).contiguous().view(B, L, self.d_model)
        x = self.fc_out(x)
        out = x + q.view(B, L, d_model)
        out = self.norm(out)
        out2 = self.mlp(out)
        out = out + out2
        return out.view(B, L, H, W)

# ---------------- DHUnet ----------------
class DHUnet(nn.Module):
    def __init__(self):
        channel = 64
        super().__init__()
        self.feature = nn.Conv2d(1, channel, 1)
        self.dcse_block_1 = nn.Sequential(
            DeformableConvBlock(channel, channel),
            SEBlock(channel, 4),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.dcse_block_2 = nn.Sequential(
            DeformableConvBlock(channel, channel),
            SEBlock(channel, 4),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.dcse_block_3 = nn.Sequential(
            DeformableConvBlock(channel, channel),
            SEBlock(channel, 8),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.dcse_block_4 = nn.Sequential(
            DeformableConvBlock(channel, channel),
            SEBlock(channel, 8),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.dcse_block_5 = nn.Sequential(
            DeformableConvBlock(channel, channel),
            SEBlock(channel, 16),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        self.cahc_block_1 = nn.Sequential(
            CoordAttention(channel, 4),
            HyperConv2D(channel, channel),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.cahc_block_2 = nn.Sequential(
            CoordAttention(channel, 4),
            HyperConv2D(channel, channel),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.cahc_block_3 = nn.Sequential(
            CoordAttention(channel, 8),
            HyperConv2D(channel, channel),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.cahc_block_4 = nn.Sequential(
            CoordAttention(channel, 8),
            HyperConv2D(channel, channel),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.cahc_block_5 = nn.Sequential(
            CoordAttention(channel, 16),
            HyperConv2D(channel, channel),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        self.conv1x1 = nn.Conv2d(2 * channel, channel, 1)
        self.cnn1 = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.cnn2 = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))

        self.upsample1 = nn.ConvTranspose2d(channel, channel, 3, 2, 1, 1)
        self.decoder_block_1 = nn.Sequential(
            MultiHeadAttention(16 * 16, 8),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.ConvTranspose2d(channel, channel, 3, 2, 1, 1)
        self.decoder_block_2 = nn.Sequential(
            MultiHeadAttention(32 * 32, 8),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.upsample3 = nn.ConvTranspose2d(channel, channel, 3, 2, 1, 1)
        self.decoder_block_3 = nn.Sequential(
            MultiHeadAttention(64 * 64, 8),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.upsample4 = nn.ConvTranspose2d(channel, channel, 3, 2, 1, 1)
        self.decoder_block_4 = nn.Sequential(
            MultiHeadAttention(128 * 128, 8),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.upsample5 = nn.ConvTranspose2d(1, 1, 9, 8, 1, 1)
        self.conv1x1_2 = nn.Conv2d(channel, 1, 1)

    def forward(self, x):
        # Input x: (B, 1, 16, 16)
        x = self.upsample5(x)  # -> (B, 1, 128, 128)
        x = self.feature(x)    # -> (B, 64, 128, 128)

        # Encoder Branch 1 (dcse blocks)
        fcu_4 = self.dcse_block_1(x)      # -> (B, 64, 128, 128)
        x1 = self.avg_pool(fcu_4)        # -> (B, 64, 64, 64)
        fcu_3 = self.dcse_block_2(x1)    # -> (B, 64, 64, 64)
        x1 = self.avg_pool(fcu_3)        # -> (B, 64, 32, 32)
        fcu_2 = self.dcse_block_3(x1)    # -> (B, 64, 32, 32)
        x1 = self.avg_pool(fcu_2)        # -> (B, 64, 16, 16)
        fcu_1 = self.dcse_block_4(x1)    # -> (B, 64, 16, 16)
        x1 = self.avg_pool(fcu_1)        # -> (B, 64, 8, 8)
        x1 = self.dcse_block_5(x1)       # -> (B, 64, 8, 8)

        # Encoder Branch 2 (cahc blocks)
        fcd_4 = self.cahc_block_1(x)     # -> (B, 64, 128, 128)
        x2 = self.avg_pool(fcd_4)        # -> (B, 64, 64, 64)
        fcd_3 = self.cahc_block_2(x2)    # -> (B, 64, 64, 64)
        x2 = self.avg_pool(fcd_3)        # -> (B, 64, 32, 32)
        fcd_2 = self.cahc_block_3(x2)    # -> (B, 64, 32, 32)
        x2 = self.avg_pool(fcd_2)        # -> (B, 64, 16, 16)
        fcd_1 = self.cahc_block_4(x2)    # -> (B, 64, 16, 16)
        x2 = self.avg_pool(fcd_1)        # -> (B, 64, 8, 8)
        x2 = self.cahc_block_5(x2)       # -> (B, 64, 8, 8)

        # Fusion
        x = torch.cat([x1, x2], dim=1)   # -> (B, 128, 8, 8)
        x = self.conv1x1(x)              # -> (B, 64, 8, 8)
        x = self.cnn1(x)                 # -> (B, 64, 8, 8)
        x = self.cnn2(x)                 # -> (B, 64, 8, 8)

        # Decoder Blocks
        x = self.upsample1(x)                    # -> (B, 64, 16, 16)
        x = self.decoder_block_1([x, fcu_1, fcd_1])  # -> (B, 64, 16, 16)

        x = self.upsample2(x)                    # -> (B, 64, 32, 32)
        x = self.decoder_block_2([x, fcu_2, fcd_2])  # -> (B, 64, 32, 32)

        x = self.upsample3(x)                    # -> (B, 64, 64, 64)
        x = self.decoder_block_3([x, fcu_3, fcd_3])  # -> (B, 64, 64, 64)

        x = self.upsample4(x)                    # -> (B, 64, 128, 128)
        # x = self.decoder_block_4([x, fcu_4, fcd_4])  # -> (B, 64, 128, 128)

        x = self.conv1x1_2(x)                    # -> (B, 1, 128, 128)
        return x

# ---------------- Main ----------------
if __name__ == "__main__":
    b = 1
    input_size = 16
    x = torch.rand(b, 1, input_size, input_size)  # 输入: (B, 1, 16, 16)
    model = DHUnet()

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