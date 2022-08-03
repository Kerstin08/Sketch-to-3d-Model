import torch
import torch.nn as nn
import torch.nn.functional as F

class MapGen():
    def __init__(self):
        self.G = Generator()
        self.D = Discriminator()
        # Todo: refine learning rate
        self.learning_rate = 0.001
        # Todo: check if optimizers are okay
        self.optim_G = torch.optim.Adam(self.G.parameters(), self.learning_rate)
        self.optim_D = torch.optim.Adam(self.D.parameters(), self.learning_rate)

    # Todo: check if both models need to be saved and loaded during training and test
    # Todo: depending on max epochs maybe remove older checkpoints to not run out of space
    # Todo: maybe make util class for save and loading of checkpoints, since those will be used for multiple models
    def generate_checkpointData(self, model, optim, epoch, checkpointPath, modelPrefix):
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optim.state_dict()
        }
        checkpointPath = checkpointPath + modelPrefix + str(epoch) + ".pth"
        return checkpoint, checkpointPath

    def save_models(self, epoch, checkpointPath):
        checkpoint_G, checkpointPath_G = self.generate_checkpointData(self.G, self.optim_G, epoch, checkpointPath, "_G_")
        torch.save(checkpoint_G, checkpointPath_G)
        checkpoint_D, checkpointPath_D = self.generate_checkpointData(self.D, self.optim_D, epoch, checkpointPath, "_D_")
        torch.save(checkpoint_D, checkpointPath_D)

    # Todo: something is not right with epoch, need other solution for that depending on how they are implemented in train/test
    # --> either save it in class itself or not within model
    def load_models(self, epoch, checkpointPath):
        checkpointPath_G = checkpointPath + "_G_" + str(epoch) + ".pth"
        loaded_checkpoint_G = torch.load(checkpointPath_G)
        self.G.loaded_checkpoint.load_state_dict(loaded_checkpoint_G["model_state"])
        self.optim_G.load_state_dict(loaded_checkpoint_G["optimizer_state"])
        checkpointPath_D = checkpointPath + "_D_" + str(epoch) + ".pth"
        loaded_checkpoint_D = torch.load(checkpointPath_D)
        self.D.loaded_checkpoint.load_state_dict(loaded_checkpoint_D["model_state"])
        self.optim_D.load_state_dict(loaded_checkpoint_D["optimizer_state"])

    def train(self):
        pass

    def test(self):
        pass


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
        self.conv1 = nn.Conv2d(3, 64, 4)
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

