import torch
from torch import nn
import torch.nn.functional as F

class DeepDecoder(nn.Module):
    
    def __init__(self,k,k_out):
        """Initialize layers."""
        assert k_out == 1 or k_out == 3, 'k_out must be 1 or 3'
        super().__init__()
        self.conv1 = nn.Conv2d(k, k, 1, bias=False)
        self.conv2 = nn.Conv2d(k, k, 1, bias=False)
        self.conv3 = nn.Conv2d(k, k, 1, bias=False)
        self.conv4 = nn.Conv2d(k, k, 1, bias=False)
        self.conv5 = nn.Conv2d(k, k, 1, bias=False)
        self.conv6 = nn.Conv2d(k, k, 1, bias=False)
        self.conv7 = nn.Conv2d(k, k_out, 1, bias=False)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.norm = nn.BatchNorm2d(k)

    def forward(self, x): # 16x16
        """Forward pass of network."""
        x = self.norm(F.relu(self.upsample(self.conv1(x)))) #32
        x = self.norm(F.relu(self.upsample(self.conv2(x)))) #64
        x = self.norm(F.relu(self.upsample(self.conv3(x)))) #128
        x = self.norm(F.relu(self.upsample(self.conv4(x)))) #256
        x = self.norm(F.relu(self.upsample(self.conv5(x)))) #512
        x = self.norm(F.relu(self.conv6(x))) #512
        x = torch.sigmoid(self.conv7(x))
        return x