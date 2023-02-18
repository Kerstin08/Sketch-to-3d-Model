# Generator of Neural Network for map generation
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.bn = None
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        fx = self.conv(self.lrelu(x))

        if self.bn is not None:
            fx = self.bn(fx)

        return fx


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout2d(p=0.5, inplace=True)

    def forward(self, x):
        fx = self.bn(self.deconv(self.relu(x)))

        if self.dropout is not None:
            fx = self.dropout(fx)

        return fx


class Generator(nn.Module):
    def __init__(self, channel):
        super().__init__()
        # Encoder
        self.e_conv1 = nn.Conv2d(channel, 64, kernel_size=4, stride=2, padding=1)
        self.e_conv2 = Encoder(64, 128)
        self.e_conv3 = Encoder(128, 256)
        self.e_conv4 = Encoder(256, 512)
        self.e_conv5 = Encoder(512, 512)
        self.e_conv6 = Encoder(512, 512)
        self.e_conv7 = Encoder(512, 512)
        self.e_conv8 = Encoder(512, 512, batch_norm=False)

        # Decoder
        self.d_deconv1 = Decoder(512, 512, dropout=True)
        self.d_deconv2 = Decoder(1024, 512, dropout=True)
        self.d_deconv3 = Decoder(1024, 512, dropout=True)
        self.d_deconv4 = Decoder(1024, 512)
        self.d_deconv5 = Decoder(1024, 256)
        self.d_deconv6 = Decoder(512, 128)
        self.d_deconv7 = Decoder(256, 64)
        self.d_deconv8 = nn.ConvTranspose2d(128, channel, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # up: decoder
        # down: encoder
        # outermost: downconv, uprelu, upconv, nn.Tanh
        # innermost: downconv, downrelu, downnorm, uprelu, upconv, upnorm
        # everything inbetween: downrelu, downconv, downnorm, uprelu, upconv, upnorm
        # Encoder
        # outermost
        e1 = self.e_conv1(x)
        e2 = self.e_conv2(e1)
        e3 = self.e_conv3(e2)
        e4 = self.e_conv4(e3)
        e5 = self.e_conv5(e4)
        e6 = self.e_conv6(e5)
        e7 = self.e_conv7(e6)
        # innermost
        e8 = self.e_conv8(e7)

        # Decoder
        # innermost
        d1 = self.d_deconv1(e8)
        d1 = torch.cat([d1, e7], 1)
        d2 = self.d_deconv2(d1)
        d2 = torch.cat([d2, e6], 1)
        d3 = self.d_deconv3(d2)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.d_deconv4(d3)
        d4 = torch.cat([d4, e4], 1)
        d5 = self.d_deconv5(d4)
        d5 = torch.cat([d5, e3], 1)
        d6 = self.d_deconv6(d5)
        d6 = torch.cat([d6, e2], 1)
        d7 = self.d_deconv7(d6)
        d7 = torch.cat([d7, e1], 1)
        # outermost
        d8 = self.d_deconv8(d7)
        return torch.tanh(d8)
