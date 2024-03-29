# Discrimiantor (critic) of Neural Network for map generation
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channel: int):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(2 * channel, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv2_ln = nn.LayerNorm([128, 64, 64])
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv3_ln = nn.LayerNorm([256, 32, 32])
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv4_ln = nn.LayerNorm([512, 16, 16])
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor):
        d1 = self.lrelu(self.conv1(x))
        d2 = self.lrelu(self.conv2_ln(self.conv2(d1)))
        d3 = self.lrelu(self.conv3_ln(self.conv3(d2)))
        d4 = self.lrelu(self.conv4_ln(self.conv4(d3)))
        d5 = self.conv5(d4)
        return d5
