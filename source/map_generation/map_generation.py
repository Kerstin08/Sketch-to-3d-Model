import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import dataset_generation.DataSet as DataSet
from torch.utils.data import DataLoader
import source.util.save_load_networks as Save_Load_Network
from torchvision.utils import make_grid
from enum import Enum
import shutil

class Type(Enum):
    normal = 1,
    depth = 2


class MapGen():
    def __init__(self, type, epoch, n_critic, weight_L1, batch_size):
        self.G = Generator()
        self.D = Discriminator()
        # Todo: refine learning rate & epochs
        # --> maybe not needed if pytorch lightning is used
        self.learning_rate = 0.001
        self.epochs = epoch
        self.optim_G = torch.optim.RMSprop(self.G.parameters(), self.learning_rate)
        self.optim_D = torch.optim.RMSprop(self.D.parameters(), self.learning_rate)
        self.n_critic = n_critic
        self.type = type
        self.weight_L1 = weight_L1
        self.batch_size = batch_size

    def gen_filenames(self, epoch=-1):
        if epoch>0:
            if self.type == Type.normal:
                filename_G = "Gen_N_" + str(epoch) + ".pth"
                filename_D = "Disc_N_" + str(epoch) + ".pth"
            else:
                filename_G = "Gen_D_" + str(epoch) + ".pth"
                filename_D = "Disc_D_" + str(epoch) + ".pth"
        else:
            if self.type == Type.normal:
                filename_G = "Gen_N_trained.pth"
                filename_D = "Disc_N_trained.pth"
            else:
                filename_G = "Gen_D_trained.pth"
                filename_D = "Disc_D_trained.pth"
        return filename_G, filename_D

    def train(self, input_dir, target_dir, checkpiont_dict):
        dataSet = DataSet.DS(input_dir, target_dir)
        dataloader = DataLoader(dataSet, batch_size=self.batch_size,
                                shuffle=True, num_workers=0)
        self.G.train()
        self.D.train()
        for epoch in range(self.epochs):
            for i_batch, sample_batched in enumerate(dataloader):
                # generate fake images
                fake_images = self.G(sample_batched['input'])
                # train discriminator on fake images
                pred_false = self.D(fake_images.detach())
                d_loss_fake = torch.mean(pred_false)
                # train discriminator on real images
                pred_true = self.D(sample_batched['target'])
                d_loss_real = torch.mean(pred_true)

                # loss as defined by Wasserstein paper
                d_loss = -d_loss_fake + d_loss_real
                self.optim_D.zero_grad()
                d_loss.backward()

                # Train the generator every n_critic batch --> Wasserstein GAN
                if i_batch % self.n_critic == 0:
                    # mix traditional loss (pixelwise_l1 loss with wasserstein loss)
                    # mask loss (as described in Su paper) not needed, since we do not have an user-defined mask
                    # use BCEloss as described in Isola paper in order counteract the blurriness L1 introduces (see Isola et al. 1127)
                    pixelwise_loss = torch.nn.L1Loss(sample_batched['input'], fake_images)
                    g_loss = d_loss_fake + pixelwise_loss * self.weight_L1 #+ torch.nn.BCEWithLogitsLoss(pred_false)
                    self.optim_G.zero_grad()
                    g_loss.backward()
            if (self.epochs/5) % 100 == 0:
                filename_G, filename_D = self.gen_filenames(epoch)
                gen_checkpoint_path = os.path.join(checkpiont_dict, filename_G)
                disc_checkpoint_path = os.path.join(checkpiont_dict, filename_D)
                Save_Load_Network.save_models(self.G, self.optim_G, epoch, gen_checkpoint_path)
                Save_Load_Network.save_models(self.D, self.optim_D, epoch, disc_checkpoint_path)

        filename_G, filename_D = self.gen_filenames()
        gen_checkpoint_path = os.path.join(checkpiont_dict, filename_G)
        disc_checkpoint_path = os.path.join(checkpiont_dict, filename_D)
        Save_Load_Network.save_models(self.G, self.optim_G, self.epochs, gen_checkpoint_path)
        Save_Load_Network.save_models(self.D, self.optim_D, self.epochs, disc_checkpoint_path)

    def test(self, input_dir, target_dir, output_dir, generate_comparison):
        self.G.eval()
        dataSet = DataSet.DS(input_dir, target_dir)
        with torch.no_grad():
            for idx, data in enumerate(dataSet):
                predicted_image = self.G(data['input'])
                imagename = data['input_path'].rsplit("\\", 1)[-1]
                if generate_comparison:
                    if len(output_dir) <= 0:
                        raise RuntimeError("Directory to store comparison images is not given")
                    Grid = make_grid([data['input'], data['target'], predicted_image])
                    img = torchvision.transforms.ToPILImage(Grid)
                    image_path = os.path.join(output_dir, "comparison_" + imagename)
                    img.save(image_path)
                else:
                    if self.type == Type.normal:
                        output_dir_generated = os.path.join(output_dir, "predicted_normals")
                    else:
                        output_dir_generated = os.path.join(output_dir, "predicted_depth")
                    if os.path.exists(output_dir_generated):
                        shutil.rmtree(output_dir_generated)
                    os.mkdir(output_dir_generated)
                    img = torchvision.transforms.ToPILImage(predicted_image)
                    image_path = os.path.join(output_dir, idx + imagename)
                    img.save(image_path)
# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder
        self.e_conv1 = nn.Conv2d(3, 64, 4)
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
        self.d_deconv1_drop = nn.Dropout()
        self.d_deconv2 = nn.ConvTranspose2d(512, 512, 4)
        self.d_deconv2_bn = nn.BatchNorm2d(512)
        self.d_deconv2_drop = nn.Dropout()
        self.d_deconv3 = nn.ConvTranspose2d(512, 512, 4)
        self.d_deconv3_bn = nn.BatchNorm2d(512)
        self.d_deconv3_drop = nn.Dropout()
        self.d_deconv4 = nn.ConvTranspose2d(512, 512, 4)
        self.d_deconv4_bn = nn.BatchNorm2d(512)
        self.d_deconv4_drop = nn.Dropout()
        self.d_deconv5 = nn.ConvTranspose2d(512, 256, 4)
        self.d_deconv5_bn = nn.BatchNorm2d(256)
        self.d_deconv5_drop = nn.Dropout()
        self.d_deconv6 = nn.ConvTranspose2d(256, 128, 4)
        self.d_deconv6_bn = nn.BatchNorm2d(128)
        self.d_deconv6_drop = nn.Dropout()
        self.d_deconv7 = nn.ConvTranspose2d(128, 64, 4)
        self.d_deconv7_bn = nn.BatchNorm2d(64)
        self.d_deconv7_drop = nn.Dropout()
        self.d_deconv8 = nn.ConvTranspose2d(64, 3, 4)


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
        d1 = torch.cat([d1, e7], 3)
        d2 = self.d_deconv2_drop(self.d_deconv2_bn(self.d_deconv2(F.relu(d1))))
        d2 = torch.cat([d2, e6], 3)
        d3 = self.d_deconv3_drop(self.d_deconv3_bn(self.d_deconv3(F.relu(d2))))
        d3 = torch.cat([d3, e5], 3)
        d4 = self.d_deconv4_drop(self.d_deconv4_bn(self.d_deconv4(F.relu(d3))))
        d4 = torch.cat([d4, e5], 3)
        d5 = self.d_deconv5_drop(self.d_deconv5_bn(self.d_deconv5(F.relu(d4))))
        d5 = torch.cat([d5, e4], 3)
        d6 = self.d_deconv6_drop(self.d_deconv6_bn(self.d_deconv6(F.relu(d5))))
        d6 = torch.cat([d6, e3], 3)
        d7 = self.d_deconv7_drop(self.d_deconv7_bn(self.d_deconv7(F.relu(d6))))
        d7 = torch.cat([d7, e2], 3)
        # outermost
        d8 = self.d_deconv8(F.relu(d7))
        d8 = torch.cat([d8, e1], 3)
        return F.tanh(d8)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, 4)
        self.conv2 = nn.Conv2d(64, 128, 4)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 4)
        self.conv4_bn = nn.BatchNorm2d(512)

    def forward(self, x):
        d1 = F.leaky_relu(self.conv1(x))
        d2 = F.leaky_relu(self.conv2_bn(self.conv2(d1)))
        d3 = F.leaky_relu(self.conv3_bn(self.conv3(d2)))
        d4 = F.leaky_relu(self.conv4_bn(self.conv4(d3)))
        # Todo: is this needed?
        d5 = nn.Linear(torch.reshape(d4, [1, -1]))
        return d5

