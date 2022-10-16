import os.path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from source.util import OpenEXR_utils
from torchvision.utils import make_grid
from enum import Enum
import shutil
import pytorch_lightning as pl

class Type(Enum):
    normal = 1,
    depth = 2


class MapGen(pl.LightningModule):
    def __init__(self, type, n_critic, channels, batch_size, weight_L1, output_dir, lr):
        super(MapGen, self).__init__()
        self.save_hyperparameters()
        self.G = Generator(channels)
        self.D = Discriminator(channels)
        self.n_critic = n_critic
        self.batch_size = batch_size
        self.type = type
        self.weight_L1 = weight_L1
        self.output_dir = output_dir
        self.lr = lr
        self.L1 = torch.nn.L1Loss()

    def configure_optimizers(self):
        opt_g = torch.optim.RMSprop(self.G.parameters(), lr=(self.lr or self.learning_rate))
        opt_d = torch.optim.RMSprop(self.D.parameters(), lr=(self.lr or self.learning_rate))

        return [{'optimizer': opt_g, 'frequency': 1},
                {'optimizer': opt_d, 'frequency': self.n_critic}]


    def forward(self, sample_batched):
        x = sample_batched['input']
        return self.G(x)

    def generator_step(self, sample_batched, fake_images):
        print("Generator")
        input_predicted = torch.cat((sample_batched['input'], fake_images), 1)
        pred_false = self.D(input_predicted)
        d_loss_fake = torch.mean(pred_false)
        pixelwise_loss = self.L1(sample_batched['target'], fake_images)
        g_loss = -d_loss_fake + pixelwise_loss * self.weight_L1
        self.log("g_Loss", float(g_loss.item()), on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return g_loss

    def discriminator_step(self, sample_batched, fake_images):
        print("Discriminator")
        input_predicted = torch.cat((sample_batched['input'], fake_images), 1)
        pred_false = self.D(input_predicted.detach())
        d_loss_fake = torch.mean(pred_false)
        # train discriminator on real images
        input_target = torch.cat((sample_batched['input'], sample_batched['target']), 1)
        pred_true = self.D(input_target)
        d_loss_real = torch.mean(pred_true)

        # loss as defined by Wasserstein paper
        d_loss = -d_loss_real + d_loss_fake
        self.log("d_loss", float(d_loss.item()), on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("d_loss_real", float(d_loss_real.item()), on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("d_loss_fake", float(d_loss_fake.item()), on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)

        return d_loss

    def training_step(self, sample_batched, batch_idx, optimizer_idx):
        fake_images = self(sample_batched)
        if optimizer_idx == 0:
            loss = self.generator_step(sample_batched, fake_images)

        if optimizer_idx == 1:
            loss = self.discriminator_step(sample_batched, fake_images)
        return loss

    def validation_step(self, sample_batched, batch_idx):
        predicted_image = self(sample_batched)
        pixelwise_loss = self.L1(sample_batched['target'], predicted_image)
        self.log("val_loss", pixelwise_loss.item(), batch_size=self.batch_size)
        grid = torchvision.utils.make_grid(predicted_image[:6])
        logger = self.logger.experiment
        image_name = str(self.global_step) + "generated_images"
        logger.add_image(image_name, grid, 0)

    def test_step(self, sample_batched, batch_idx):
        predicted_image = self(sample_batched)
        imagename = sample_batched['input_path'][0].rsplit("\\", 1)[-1].split("_", 1)[0]
        if self.type == Type.depth:
            OpenEXR_utils.writeDepthImage(predicted_image, os.path.join(self.output_dir, imagename + "_depth.exr"))
        else:
            OpenEXR_utils.writeRGBImage(predicted_image, os.path.join(self.output_dir, imagename + "_normal.exr"))
        transform = torchvision.transforms.ToPILImage()
        img = transform(torch.squeeze(predicted_image))
        image_path = os.path.join(self.output_dir, imagename + ".png")
        img.save(image_path)

# Generator
class Generator(nn.Module):
    def __init__(self, channels):
        super(Generator, self).__init__()
        # Encoder
        self.e_conv1 = nn.Conv2d(channels, 64, 4)
        self.e_conv2 = nn.Conv2d(64, 128, 4)
        self.e_conv2_bn = nn.BatchNorm2d(128)
        self.e_conv3 = nn.Conv2d(128, 256, 4)
        self.e_conv3_bn = nn.BatchNorm2d(256)
        self.e_conv4 = nn.Conv2d(256, 512, 4)
        self.e_conv4_bn = nn.BatchNorm2d(512)
        self.e_conv5 = nn.Conv2d(512, 512, 4)
        self.e_conv5_bn = nn.BatchNorm2d(512)
        self.e_conv6 = nn.Conv2d(512, 512, 4)
        self.e_conv6_bn = nn.BatchNorm2d(512)
        self.e_conv7 = nn.Conv2d(512, 512, 4)
        self.e_conv7_bn = nn.BatchNorm2d(512)
        self.e_conv8 = nn.Conv2d(512, 512, 4)
        self.e_conv8_bn = nn.BatchNorm2d(512)

        #Decoder
        self.d_deconv1 = nn.ConvTranspose2d(512, 512, 4)
        self.d_deconv1_bn = nn.BatchNorm2d(512)
        self.d_deconv1_drop = nn.Dropout(0.5)
        self.d_deconv2 = nn.ConvTranspose2d(1024, 512, 4)
        self.d_deconv2_bn = nn.BatchNorm2d(512)
        self.d_deconv2_drop = nn.Dropout(0.5)
        self.d_deconv3 = nn.ConvTranspose2d(1024, 512, 4)
        self.d_deconv3_bn = nn.BatchNorm2d(512)
        self.d_deconv3_drop = nn.Dropout(0.5)
        self.d_deconv4 = nn.ConvTranspose2d(1024, 512, 4)
        self.d_deconv4_bn = nn.BatchNorm2d(512)
        self.d_deconv5 = nn.ConvTranspose2d(1024, 256, 4)
        self.d_deconv5_bn = nn.BatchNorm2d(256)
        self.d_deconv6 = nn.ConvTranspose2d(512, 128, 4)
        self.d_deconv6_bn = nn.BatchNorm2d(128)
        self.d_deconv7 = nn.ConvTranspose2d(256, 64, 4)
        self.d_deconv7_bn = nn.BatchNorm2d(64)
        self.d_deconv8 = nn.ConvTranspose2d(128, channels, 4)


    def forward(self, x):
        # up: decoder
        # down: encoder
        # outermost: downconv, uprelu, upconv, nn.Tanh
        # innermost: downconv, downrelu, downnorm, uprelu, upconv, upnorm
        # Todo: up for discussion if in innermost downnorm is needed or not
        # everything inbetween: downrelu, downconv, downnorm, uprelu, upconv, upnorm
        # Encoder
        # outermost
        e1 = self.e_conv1(x)
        e2 = self.e_conv2_bn(self.e_conv2(F.leaky_relu(e1)))
        e3 = self.e_conv3_bn(self.e_conv3(F.leaky_relu(e2)))
        e4 = self.e_conv4_bn(self.e_conv4(F.leaky_relu(e3)))
        e5 = self.e_conv5_bn(self.e_conv5(F.leaky_relu(e4)))
        e6 = self.e_conv6_bn(self.e_conv6(F.leaky_relu(e5)))
        e7 = self.e_conv7_bn(self.e_conv7(F.leaky_relu(e6)))
        # innermost
        e8 = self.e_conv8_bn(self.e_conv8(F.leaky_relu(e7)))

        # Decoder
        # innermost
        d1 = self.d_deconv1_drop(self.d_deconv1_bn(self.d_deconv1(F.relu(e8))))
        d1 = torch.cat([d1, e7], 1)
        d2 = self.d_deconv2_drop(self.d_deconv2_bn(self.d_deconv2(F.relu(d1))))
        d2 = torch.cat([d2, e6], 1)
        d3 = self.d_deconv3_drop(self.d_deconv3_bn(self.d_deconv3(F.relu(d2))))
        d3 = torch.cat([d3, e5], 1)
        d4 = self.d_deconv4_bn(self.d_deconv4(F.relu(d3)))
        d4 = torch.cat([d4, e4], 1)
        d5 = self.d_deconv5_bn(self.d_deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e3], 1)
        d6 = self.d_deconv6_bn(self.d_deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.d_deconv7_bn(self.d_deconv7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        # outermost
        d8 = self.d_deconv8(F.relu(d7))
        return torch.tanh(d8)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(2*channels, 64, 4)
        self.conv2 = nn.Conv2d(64, 128, 4)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 4)
        self.conv4_bn = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1, 4)

    def forward(self, x):
        d1 = F.leaky_relu(self.conv1(x))
        d2 = F.leaky_relu(self.conv2_bn(self.conv2(d1)))
        d3 = F.leaky_relu(self.conv3_bn(self.conv3(d2)))
        d4 = F.leaky_relu(self.conv4_bn(self.conv4(d3)))
        d5 = torch.sigmoid(self.conv5(d4))
        return d5

