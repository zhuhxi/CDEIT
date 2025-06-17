import torch
from torch import nn

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        
        # Define MLP layers for input size (B, 1, 16, 16)
        # Flatten the input to 1D vector, (B, 1, 16, 16) -> (B, 1*16*16)
        self.mlp1 = nn.Linear(1 * 16 * 16, 128 * 128)  # First layer to compress the dimension
        self.mlp2 = nn.Linear(128 * 128, 1 * 128 * 128)  # Second layer to expand back to original size

    def forward(self, x):
        # Flatten the input for MLP
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the input to (B, -1)
        
        # Pass through the MLP layers
        x = self.relu(self.mlp1(x))
        x = self.relu(self.mlp2(x))
        
        # Reshape the output to (B, 1, 128, 128)
        x = x.view(batch_size, 1, 128, 128)
        
        # Pass through the CNN layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Test Script
if __name__ == '__main__':
    b = 1
    input_size = 16
    x = torch.rand(b, 1, input_size, input_size)  # Input: (B, 1, 16, 16)
    model = SRCNN()

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
