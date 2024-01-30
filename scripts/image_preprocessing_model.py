import torch
import torch.nn as nn

# Edge detection class (using conv2d (CNN) on images)
class ContourDetector(nn.Module):
    def __init__(self):

        super(ContourDetector, self).__init__()

        # Pooling layers
        self.Pool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
        self.Flat = nn.Flatten()

    def forward(self, x):
        x = self.Pool(x)
        x = self.Pool(x)
        x = self.Pool(x)
        z = self.Pool(x)
        y = self.Flat(z)

        return z, y
