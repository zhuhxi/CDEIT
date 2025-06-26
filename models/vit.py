import torch
import torch.nn as nn
import torch.nn.functional as F

class ViTPatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, emb_size=768):
        super(ViTPatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        # Converts input image into patch embeddings using Conv2D (kernel_size=stride=patch_size)

    def forward(self, x):
        x = self.projection(x)         # Shape: (B, emb_size, H/patch, W/patch)
        x = x.flatten(2)               # Shape: (B, emb_size, N), where N = num_patches
        x = x.transpose(1, 2)          # Shape: (B, N, emb_size) - ready for transformer input
        return x


class ViTSelfAttention(nn.Module):
    def __init__(self, emb_size=768):
        super(ViTSelfAttention, self).__init__()
        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)
        self.attn_dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(emb_size, emb_size)
    
    def forward(self, x):
        # x: (B, N, E)
        q = self.query(x)             # Shape: (B, N, E)
        k = self.key(x)               # Shape: (B, N, E)
        v = self.value(x)             # Shape: (B, N, E)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        # Shape: (B, N, N) - attention scores between patches

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)   # Shape: (B, N, E)
        output = self.out_proj(output)           # Shape: (B, N, E)
        return output


class ViTBlock(nn.Module):
    def __init__(self, emb_size=768):
        super(ViTBlock, self).__init__()
        self.attn = ViTSelfAttention(emb_size)
        self.norm1 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.GELU(),
            nn.Linear(emb_size * 4, emb_size)
        )
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))    # Residual connection after attention
        x = x + self.mlp(self.norm2(x))     # Residual connection after MLP
        return x


class Vit(nn.Module):
    def __init__(self, emb_size=768, num_blocks=12, patch_size=16, img_size=16):
        super(Vit, self).__init__()
        
        self.patch_embed = ViTPatchEmbedding(in_channels=1, patch_size=patch_size, emb_size=emb_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))  
        # Class token of shape (1, 1, emb_size)

        self.positional_encoding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, emb_size))
        # Positional encoding for (num_patches + 1), shape: (1, N+1, emb_size)

        self.blocks = nn.ModuleList([ViTBlock(emb_size) for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(emb_size)

        self.to_image = nn.Sequential(
            nn.Flatten(start_dim=1),                     # Shape: (B, (N+1)*E)
            nn.Linear(2 * 768, 128 * 128),               # Assuming N+1=2, output: (B, 16384)
            nn.Unflatten(1, (1, 128, 128))               # Reshape to image: (B, 1, 128, 128)
        )

        self.cnn = nn.Sequential(                        # Simple CNN refinement
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (B, 16, 128, 128)
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)   # (B, 1, 128, 128)
        )

    def forward(self, x):
        batch_size = x.size(0)                # e.g., B = 8

        x = self.patch_embed(x)              # → (B, N=1, E=768) if image 16x16 and patch 16x16
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # → (B, 1, 768)
        x = torch.cat([cls_tokens, x], dim=1)                  # → (B, N+1=2, 768)

        x = x + self.positional_encoding                      # Add positional encoding (B, 2, 768)

        for block in self.blocks:
            x = block(x)                                      # Apply Transformer blocks

        x = self.layer_norm(x)                                # Final LayerNorm → (B, 2, 768)

        x = self.to_image(x)                                  # Flatten + MLP + reshape → (B, 1, 128, 128)
        x = self.cnn(x)                                       # → (B, 1, 128, 128)
        return x


# Test Script
if __name__ == '__main__':
    b = 1
    input_size = 16
    x = torch.rand(b, 1, input_size, input_size)  # Input: (B, 1, 16, 16)
    model = Vit()

    # --- Shape Test ---
    try:
        out = model(x)
        print(f"✅ Forward Pass Success: {x.shape} → {out.shape}")
    except Exception as e:
        print(f"❌ Forward Failed: {e}")

    # --- FLOPs and Parameters ---
    try:
        from nni.compression.utils.counter import count_flops_params
        flops, params, _ = count_flops_params(model, x=(x,))
        print(f"📊 FLOPs: {flops / 1e6:.2f} MFLOPs | Params: {params / 1e6:.2f} M")
    except ImportError:
        print("⚠️ NNI not installed. Run: pip install nni")

