import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C3(nn.Module):  # CSP Bottleneck
    def __init__(self, in_channels, out_channels, n=1):
        super(C3, self).__init__()
        self.cv1 = Conv(in_channels, out_channels, 1)
        self.cv2 = Conv(in_channels, out_channels, 1)
        self.cv3 = Conv(2 * out_channels, out_channels, 1)
        self.m = nn.Sequential(*[Conv(out_channels, out_channels) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):  # Spatial Pyramid Pooling - Fast
    def __init__(self, in_channels, out_channels):
        super(SPPF, self).__init__()
        self.cv1 = Conv(in_channels, out_channels, 1)
        self.spp = nn.ModuleList([nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                                  nn.MaxPool2d(kernel_size=9, stride=1, padding=4),
                                  nn.MaxPool2d(kernel_size=13, stride=1, padding=6)])

    def forward(self, x):
        return self.cv1(torch.cat([x] + [pool(x) for pool in self.spp], 1))


class Yolo5(nn.Module):
    def __init__(self, num_classes=80, anchors=None):
        super(Yolo5, self).__init__()
        # Backbone
        self.backbone = nn.Sequential(
            Conv(3, 32, 3, 2),  # Conv Layer 1
            C3(32, 64, 1),  # CSP Bottleneck 1
            C3(64, 128, 1),  # CSP Bottleneck 2
            SPPF(128, 256),  # Spatial Pyramid Pooling
            C3(256, 512, 2),  # CSP Bottleneck 3
            C3(512, 1024, 3),  # CSP Bottleneck 4
        )
        
        # Head
        self.head = nn.Sequential(
            Conv(1024, 512, 3, 1),
            C3(512, 256, 1),
            Conv(256, num_classes + 5, 1)  # num_classes + 5 (4 for box, 1 for objectness, and num_classes for class score)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


# Create a YOLOv5 model
model = Yolo5(num_classes=80)  # For COCO dataset with 80 classes

# Input tensor (batch size of 1, 3 channels, 640x640 image)
input_tensor = torch.randn(1, 3, 640, 640)

# Forward pass
output = model(input_tensor)
print(output.shape)