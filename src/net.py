import torch
from utils.model import try_gpu


class MNISTNet(torch.nn.Module):

    def __init__(self):
        super(MNISTNet, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=5, padding=2),
            torch.nn.Sigmoid(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(6, 16, kernel_size=5),
            torch.nn.Sigmoid(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 5 * 5, 120),
            torch.nn.Sigmoid(),
            torch.nn.Linear(120, 84),
            torch.nn.Sigmoid(),
            torch.nn.Linear(84, 10)
        ).to(device=try_gpu())


# convolutional block, used for building CIFARNet
def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True)
    )


# convulutional net for CIFAR dataset
class CIFARNet(torch.nn.Module):

    def __init__(self):
        super(CIFARNet, self).__init__()
        self.model = torch.nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 32),
            conv_block(32, 64, stride=2),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, 128, stride=2),
            conv_block(128, 128),
            conv_block(128, 256),
            conv_block(256, 256),
            torch.nn.AdaptiveAvgPool2d(1)
        ).to(try_gpu())
        self.classifier = torch.nn.Linear(256, 10).to(try_gpu())

    def forward(self, x):
        h = self.model(x)
        B, C, _, _ = h.shape
        h = h.view(B, C)
        return self.classifier(h)
