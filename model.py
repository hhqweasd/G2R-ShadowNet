import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init_normal
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator_S2F(nn.Module):
    def __init__(self,init_weights=False):
        super(Generator_S2F, self).__init__()

        # Initial convolution block
        self.conv1_b=nn.Sequential(nn.ReflectionPad2d(3),
                    nn.Conv2d(3, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True))
        self.downconv2_b=nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                    nn.InstanceNorm2d(128),
                                    nn.ReLU(inplace=True))
        self.downconv3_b=nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                    nn.InstanceNorm2d(256),
                                    nn.ReLU(inplace=True))
        self.conv4_b=nn.Sequential(ResidualBlock(256))
        self.conv5_b=nn.Sequential(ResidualBlock(256))
        self.conv6_b=nn.Sequential(ResidualBlock(256))
        self.conv7_b=nn.Sequential(ResidualBlock(256))
        self.conv8_b=nn.Sequential(ResidualBlock(256))
        self.conv9_b=nn.Sequential(ResidualBlock(256))
        self.conv10_b=nn.Sequential(ResidualBlock(256))
        self.conv11_b=nn.Sequential(ResidualBlock(256))
        self.conv12_b=nn.Sequential(ResidualBlock(256))
        self.upconv13_b=nn.Sequential(nn.ConvTranspose2d(256,128,3,stride=2,padding=1,output_padding=1),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True))
        self.upconv14_b=nn.Sequential(nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1),
                        nn.InstanceNorm2d(64),
                        nn.ReLU(inplace=True))
        self.conv15_b=nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(64, 3, 7))
        
        if init_weights:
            self.apply(weights_init_normal)
    
    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_S2F(init_weights=True)
        return model


    def forward(self,xin):
        x=self.conv1_b(xin)
        x=self.downconv2_b(x)
        x=self.downconv3_b(x)
        x=self.conv4_b(x)
        x=self.conv5_b(x)
        x=self.conv6_b(x)
        x=self.conv7_b(x)
        x=self.conv8_b(x)
        x=self.conv9_b(x)
        x=self.conv10_b(x)
        x=self.conv11_b(x)
        x=self.conv12_b(x)
        x=self.upconv13_b(x)
        x=self.upconv14_b(x)
        x=self.conv15_b(x)
        xout=x+xin
        return xout.tanh()

class Generator_F2S(nn.Module):
    def __init__(self,init_weights=False):
        super(Generator_F2S, self).__init__()

        # Initial convolution block
        self.conv1_b=nn.Sequential(nn.ReflectionPad2d(3),
                    nn.Conv2d(4, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True))
        self.downconv2_b=nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                    nn.InstanceNorm2d(128),
                                    nn.ReLU(inplace=True))
        self.downconv3_b=nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                    nn.InstanceNorm2d(256),
                                    nn.ReLU(inplace=True))
        self.conv4_b=nn.Sequential(ResidualBlock(256))
        self.conv5_b=nn.Sequential(ResidualBlock(256))
        self.conv6_b=nn.Sequential(ResidualBlock(256))
        self.conv7_b=nn.Sequential(ResidualBlock(256))
        self.conv8_b=nn.Sequential(ResidualBlock(256))
        self.conv9_b=nn.Sequential(ResidualBlock(256))
        self.conv10_b=nn.Sequential(ResidualBlock(256))
        self.conv11_b=nn.Sequential(ResidualBlock(256))
        self.conv12_b=nn.Sequential(ResidualBlock(256))
        self.upconv13_b=nn.Sequential(nn.ConvTranspose2d(256,128,3,stride=2,padding=1,output_padding=1),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True))
        self.upconv14_b=nn.Sequential(nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1),
                        nn.InstanceNorm2d(64),
                        nn.ReLU(inplace=True))
        self.conv15_b=nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(64, 3, 7))
        
        if init_weights:
            self.apply(weights_init_normal)
    
    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_F2S(init_weights=True)
        return model

    def forward(self,xin,mask):
        x=torch.cat((xin,mask),1)
        x=self.conv1_b(x)
        x=self.downconv2_b(x)
        x=self.downconv3_b(x)
        x=self.conv4_b(x)
        x=self.conv5_b(x)
        x=self.conv6_b(x)
        x=self.conv7_b(x)
        x=self.conv8_b(x)
        x=self.conv9_b(x)
        x=self.conv10_b(x)
        x=self.conv11_b(x)
        x=self.conv12_b(x)
        x=self.upconv13_b(x)
        x=self.upconv14_b(x)
        x=self.conv15_b(x)
        xout=x+xin
        return xout.tanh()

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(3, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self,x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1).squeeze() #global avg pool
