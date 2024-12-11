import torch
from torch import nn
import numpy as np
from .models import *

torch.backends.cudnn.benchmark = True

class CoordinateUtils(object):
    @staticmethod
    def get_image_coordinates(h, w, normalise):
        x_range = torch.arange(w, dtype=torch.float32)
        y_range = torch.arange(h, dtype=torch.float32)
        if normalise:
            x_range = (x_range / (w - 1)) * 2 - 1
            y_range = (y_range / (h - 1)) * 2 - 1
        image_x = x_range.unsqueeze(0).repeat_interleave(h, 0)
        image_y = y_range.unsqueeze(0).repeat_interleave(w, 0).t()
        return image_x, image_y


class SpatialSoftArgmax(nn.Module):
    def __init__(self, temperature=None, normalise=False):
        """
        Applies a spatial soft argmax over the input images.
        :param temperature: The temperature parameter (float). If None, it is learnt.
        :param normalise: Should spatial features be normalised to range [-1, 1]?
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1), requires_grad = True) if temperature is None else nn.Parameter(torch.ones(1)*temperature.cpu().numpy()[0], requires_grad = False).to(temperature.device)
        self.normalise = normalise

    def forward(self, x):
        """
        Applies Spatial SoftArgmax operation on the input batch of images x.
        :param x: batch of images, of size (N, C, H, W)
        :return: Spatial features (one point per channel), of size (N, C, 2)
        """
        n, c, h, w = x.size()
        spatial_softmax_per_map = nn.functional.softmax(x.contiguous().view(n * c, h * w) / self.temperature, dim=1)
        spatial_softmax = spatial_softmax_per_map.view(n, c, h, w)

        # calculate image coordinate maps
        image_x, image_y = CoordinateUtils.get_image_coordinates(h, w, normalise=self.normalise)
        # size (H, W, 2)
        image_coordinates = torch.cat((image_x.unsqueeze(-1), image_y.unsqueeze(-1)), dim=-1)
        # send to device
        image_coordinates = image_coordinates.to(device=x.device)

        # multiply coordinates by the softmax and sum over height and width, like in [2]
        expanded_spatial_softmax = spatial_softmax.unsqueeze(-1)
        image_coordinates = image_coordinates.unsqueeze(0)
        out = torch.sum(expanded_spatial_softmax * image_coordinates, dim=[2, 3])
        # (N, C, 2)
        return out


class DSAE_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, temperature=None, normalise=False):
        """
        Creates a Deep Spatial Autoencoder encoder
        :param in_channels: Input channels in the input image
        :param out_channels: Output channels for each of the layers. The last output channel corresponds to half the
        size of the low-dimensional latent representation.
        :param temperature: Temperature for spatial soft argmax operation. See SpatialSoftArgmax.
        :param normalise: Normalisation of spatial features. See SpatialSoftArgmax.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=7, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=5)
        self.batch_norm2 = nn.BatchNorm2d(out_channels[1])
        self.conv3 = nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=5)
        self.batch_norm3 = nn.BatchNorm2d(out_channels[2])
        self.conv4 = nn.Conv2d(in_channels=out_channels[2], out_channels=out_channels[3], kernel_size=3, padding = 1)
        self.batch_norm4 = nn.BatchNorm2d(out_channels[3])
        self.conv5 = nn.Conv2d(in_channels=out_channels[3], out_channels=out_channels[4], kernel_size=3, padding = 1)
        self.batch_norm5 = nn.BatchNorm2d(out_channels[4])
        self.activ = nn.ReLU()
        self.spatial_soft_argmax = SpatialSoftArgmax(temperature=temperature, normalise=normalise)

    def forward(self, x):
        out_conv1 = self.activ(self.batch_norm1(self.conv1(x)))
        out_conv2 = self.activ(self.batch_norm2(self.conv2(out_conv1)))
        out_conv3 = self.activ(self.batch_norm3(self.conv3(out_conv2)))
        out_conv4 = self.activ(self.batch_norm4(self.conv4(out_conv3)))
        out_conv5 = self.activ(self.batch_norm5(self.conv5(out_conv4)))
        out = self.spatial_soft_argmax(out_conv5)
        return out
    
# class DSAE_Encoder(nn.Module):
#     def __init__(self, in_channels, out_channels, temperature=None, normalise=False):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )
        
#         self.spatial_soft_argmax = SpatialSoftArgmax(temperature=temperature, normalise=normalise)

#     def forward(self, x):
#         x = self.features(x)
#         out = self.spatial_soft_argmax(x)
#         return out

class DSAE_Encoder_Flexible(nn.Module):
    def __init__(self, in_channels, out_channels, strides, kernels, temperature=None, normalise=True, return_feature_maps = False):
        """
        Creates a Deep Spatial Autoencoder encoder
        :param in_channels: Input channels in the input image
        :param out_channels: Output channels for each of the layers. The last output channel corresponds to half the
        size of the low-dimensional latent representation.
        :param temperature: Temperature for spatial soft argmax operation. See SpatialSoftArgmax.
        :param normalise: Normalisation of spatial features. See SpatialSoftArgmax.
        """
        super().__init__()

        self.convs = []
        self.bns = []

        in_c = in_channels
        for i in range(len(out_channels)):
            self.convs.append(nn.Conv2d(in_channels = in_c, out_channels = out_channels[i], kernel_size = kernels[i], stride = strides[i], padding = kernels[i]//2))
            if not i == (len(out_channels) - 1):
                self.bns.append(nn.BatchNorm2d(out_channels[i]))
            in_c = out_channels[i]
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

        self.activ = nn.ReLU()
        self.spatial_soft_argmax = SpatialSoftArgmax(temperature=temperature, normalise=normalise)
        self.return_feature_maps = return_feature_maps

    def forward(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            if not i == (len(self.convs) - 1):
                x = self.bns[i](x)
            x = self.activ(x)

        out = self.spatial_soft_argmax(x)
        if self.return_feature_maps: return out, x
        return out


class DSAE_Encoder_NoMap(nn.Module):
    def __init__(self, in_channels, out_channels, strides, kernels, temperature=None, normalise=True, return_feature_maps = False):
        """
        Creates a Deep Spatial Autoencoder encoder
        :param in_channels: Input channels in the input image
        :param out_channels: Output channels for each of the layers. The last output channel corresponds to half the
        size of the low-dimensional latent representation.
        :param temperature: Temperature for spatial soft argmax operation. See SpatialSoftArgmax.
        :param normalise: Normalisation of spatial features. See SpatialSoftArgmax.
        """
        super().__init__()

        self.convs = []
        self.bns = []

        in_c = in_channels
        for i in range(len(out_channels)):
            self.convs.append(nn.Conv2d(in_channels = in_c, out_channels = out_channels[i], kernel_size = kernels[i], stride = strides[i], padding = kernels[i]//2))
            if not i == (len(out_channels) - 1):
                self.bns.append(nn.BatchNorm2d(out_channels[i]))
            in_c = out_channels[i]
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

        self.activ = nn.ReLU()
        self.spatial_soft_argmax = SpatialSoftArgmax(temperature=temperature, normalise=normalise)
        self.return_feature_maps = return_feature_maps

    def forward(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            if not i == (len(self.convs) - 1):
                x = self.bns[i](x)
            x = self.activ(x)

        return x



class DSAE_Decoder(nn.Module):
    def __init__(self, image_output_size, latent_dimension, in_channels, normalise=True):
        """
        Creates a Deep Spatial Autoencoder decoder
        :param image_output_size: (height, width) of the output, grayscale image
        :param latent_dimension: dimension of the low-dimensional encoded features.
        :param normalise: True if output in range [-1, 1], False for range [0, 1]
        """
        super().__init__()
        self.height, self.width = image_output_size
        self.latent_dimension = latent_dimension
        self.decoder = nn.Linear(in_features=latent_dimension, out_features=(self.height//8) * (self.width//8))
        self.activ = nn.ReLU() # if normalise else nn.Sigmoid()
        self.activ_f = nn.Sigmoid()
        self.dec1 = nn.ConvTranspose2d(1, 512, 3, stride=2, padding=0)
        self.dec2 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1)
        self.dec4 = nn.Conv2d(128, self.height, 3, stride=1, padding=1)
        self.dec5 = nn.Conv2d(self.height, in_channels, 3, stride=1, padding=1)

        self.batch_norm1 = nn.BatchNorm2d(512)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(self.height)

    def forward(self, x):
        out = self.activ(self.decoder(x))
        out = out.view(-1, 1, self.height//8, self.width//8)
        out = self.activ(self.batch_norm1(self.dec1(out)))
        out = self.activ(self.batch_norm2(self.dec2(out)))
        out = self.activ(self.batch_norm3(self.dec3(out)))
        out = self.activ(self.batch_norm4(self.dec4(out)))
        out = self.activ_f(self.dec5(out))
        return out[:,:,:self.height,:self.width]


class DSAE_Map_Decoder(nn.Module):
    #https://github.com/tomasjakab/imm/blob/dev/imm/models/imm_model.py
    def __init__(self, image_output_size, latent_dimension, in_channels, normalise=True):
        """
        Creates a Deep Spatial Autoencoder decoder
        :param image_output_size: (height, width) of the output, grayscale image
        :param latent_dimension: dimension of the low-dimensional encoded features.
        :param normalise: True if output in range [-1, 1], False for range [0, 1]
        """
        super().__init__()
        self.height, self.width = image_output_size
        self.latent_dimension = latent_dimension
        self.activ = nn.ReLU() # if normalise else nn.Sigmoid()
        self.activ_f = nn.Sigmoid()
        self.dec1 = nn.ConvTranspose2d(latent_dimension//2, 512, 3, stride=2, padding=0)
        self.dec2 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1)
        self.dec4 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.dec5 = nn.Conv2d(64, in_channels, 3, stride=1, padding=1)

        self.dec3_1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1)
        self.dec4_1 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.dec5_1 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

        self.batch_norm1 = nn.BatchNorm2d(512)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(64)

        self.batch_norm3_1 = nn.BatchNorm2d(128)
        self.batch_norm4_1 = nn.BatchNorm2d(64)

    def forward(self, x):
        out = create_gaussian_map(x, 8, 0.3).float()
        #out = self.activ(self.decoder(x))
        #out = out.view(-1, 1, self.height//8, self.width//8)
        out = self.activ(self.batch_norm1(self.dec1(out)))
        out_x = self.activ(self.batch_norm2(self.dec2(out)))
        out = self.activ(self.batch_norm3(self.dec3(out_x)))
        out = self.activ(self.batch_norm4(self.dec4(out)))
        out = self.activ_f(self.dec5(out))

        out_1 = self.activ(self.batch_norm3_1(self.dec3_1(out_x)))
        out_1 = self.activ(self.batch_norm4_1(self.dec4(out_1)))
        out_1 = self.activ_f(self.dec5(out_1))
        return out[:,:,:64,:64], out_1[:,:,:64,:64]


class DSAE_Map_Reg_Decoder(nn.Module):
    #https://github.com/tomasjakab/imm/blob/dev/imm/models/imm_model.py
    def __init__(self, image_output_size, latent_dimension, in_channels, normalise=True):
        """
        Creates a Deep Spatial Autoencoder decoder
        :param image_output_size: (height, width) of the output, grayscale image
        :param latent_dimension: dimension of the low-dimensional encoded features.
        :param normalise: True if output in range [-1, 1], False for range [0, 1]
        """
        super().__init__()
        self.height, self.width = image_output_size
        self.latent_dimension = latent_dimension
        self.activ = nn.ReLU() # if normalise else nn.Sigmoid()
        self.activ_f = nn.Sigmoid()
        self.dec1 = nn.ConvTranspose2d(latent_dimension//2, 64, 3, stride=3, padding=0)
        self.dec2 = nn.ConvTranspose2d(64, 32, 3, stride=3, padding=1)
        #self.dec3 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1)
        #self.dec4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.dec5 = nn.Conv2d(32, in_channels, 3, stride=1, padding=1)

        self.dec3_1 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1)
        #self.dec4_1 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.dec5_1 = nn.Conv2d(16, 1, 3, stride=2, padding=1)

        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(32)
        #self.batch_norm3 = nn.BatchNorm2d(64)
        #self.batch_norm4 = nn.BatchNorm2d(64)

        self.batch_norm3_1 = nn.BatchNorm2d(16)
        #self.batch_norm4_1 = nn.BatchNorm2d(64)

        self.reg_fcl1 = nn.Linear(in_features = latent_dimension, out_features = 128)
        self.reg_fcl2 = nn.Linear(in_features = 128, out_features = 64)
        self.reg_fcl3 = nn.Linear(in_features = 64, out_features = 64)
        self.reg_fcl_vel = nn.Linear(in_features = 64, out_features = 4)
        self.reg_fcl_ang = nn.Linear(in_features = 64, out_features = 1)
        self.reg_fcl_norm = nn.Linear(in_features = 64, out_features = 1)

        self.batch_norml_1 = nn.BatchNorm1d(128)
        self.batch_norml_2 = nn.BatchNorm1d(64)
        self.batch_norml_3 = nn.BatchNorm1d(64)
        #self.reg_cnn = CNNet(in_dim = 8, in_chans = latent_dimension//2, out_feat = 4, filters = [32,64,128,1], kernels = [3,3,1,1], dim_s = [2,1,1,1])

    def forward(self, x):
        out_map = create_gaussian_map(x, 8, 0.3).float()
        #out = self.activ(self.decoder(x))
        #out = out.view(-1, 1, self.height//8, self.width//8)
        out = self.activ(self.batch_norm1(self.dec1(out_map)))
        out_x = self.activ(self.batch_norm2(self.dec2(out)))
        #out = self.activ(self.batch_norm3(self.dec3(out_x)))
        #out = self.activ(self.batch_norm4(self.dec4(out)))
        out = self.activ_f(self.dec5(out_x))

        out_1 = self.activ(self.batch_norm3_1(self.dec3_1(out_x)))
        #out_1 = self.activ(self.batch_norm4_1(self.dec4_1(out_1)))
        out_1 = self.activ_f(self.dec5_1(out_1))

        #reg_out = self.reg_cnn(out_map)
        reg_out = self.activ(self.batch_norml_1(self.reg_fcl1(x)))
        reg_out = self.activ(self.batch_norml_2(self.reg_fcl2(reg_out)))
        reg_out = self.activ(self.batch_norml_3(self.reg_fcl3(reg_out)))
        reg_vel = self.reg_fcl_vel(reg_out)
       # reg_norm = self.reg_fcl_norm(reg_out)
       # reg_ang = self.reg_fcl_ang(reg_out)

        #reg_dir = reg_vel/torch.norm(reg_vel, dim = 1).view((-1,1))

        return out[:,:,:64,:64], out_1[:,:,:64,:64], reg_vel, x#, reg_dir, reg_norm, reg_ang


class DSAE_Map_Reg_Decoder_Flexible(nn.Module):
    #https://github.com/tomasjakab/imm/blob/dev/imm/models/imm_model.py
    def __init__(self, image_output_size, latent_dimension, in_channels, out_channels, kernels, upsamples, gauss_map_dim = 8, gauss_map_var = 0.1,normalise=True):
        """
        Creates a Deep Spatial Autoencoder decoder
        :param image_output_size: (height, width) of the output, grayscale image
        :param latent_dimension: dimension of the low-dimensional encoded features.
        :param normalise: True if output in range [-1, 1], False for range [0, 1]
        """
        super().__init__()
        self.height, self.width = image_output_size, image_output_size
        self.latent_dimension = latent_dimension
        self.activ = nn.ReLU() # if normalise else nn.Sigmoid()
        self.activ_f = nn.Sigmoid()
        self.upsamples = upsamples
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.convs = []
        self.bns = []

        in_c = latent_dimension//2
        for i in range(len(out_channels)):
            self.convs.append(nn.Conv2d(in_channels = in_c, out_channels = out_channels[i], kernel_size = kernels[i], stride = 1, padding = kernels[i]//2))
            if not i == (len(out_channels) -1):
                self.bns.append(nn.BatchNorm2d(out_channels[i]))
            in_c = out_channels[i]

        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

        self.gauss_map_dim = gauss_map_dim
        self.gauss_map_var = gauss_map_var

        #self.reg_cnn = CNNet(in_dim = 8, in_chans = latent_dimension//2, out_feat = 4, filters = [32,64,128,1], kernels = [3,3,1,1], dim_s = [2,1,1,1])

    def forward(self, keyp, output_map = False):
        map = create_gaussian_map(keyp.double(), self.gauss_map_dim, self.gauss_map_var).float()
        x = map
        #out = self.activ(self.decoder(x))
        #out = out.view(-1, 1, self.height//8, self.width//8)

        for i in range(len(self.convs)):
            x = self.convs[i](x)
            if not i == (len(self.convs) - 1):
                x = self.bns[i](x)
                x = self.activ(x)
            if self.upsamples[i] == 1:
                x = self.up(x)

        if output_map: return self.activ_f(x[:,:,:self.height, :self.width]), keyp, map
        return self.activ_f(x[:,:,:self.height, :self.width]), keyp

class DSAE_NoMap_Reg_Decoder_Flexible(nn.Module):
    #https://github.com/tomasjakab/imm/blob/dev/imm/models/imm_model.py
    def __init__(self, image_output_size, latent_dimension, in_channels, out_channels, kernels, upsamples, gauss_map_dim = 8, gauss_map_var = 0.1,normalise=True):
        """
        Creates a Deep Spatial Autoencoder decoder
        :param image_output_size: (height, width) of the output, grayscale image
        :param latent_dimension: dimension of the low-dimensional encoded features.
        :param normalise: True if output in range [-1, 1], False for range [0, 1]
        """
        super().__init__()
        self.height, self.width = image_output_size, image_output_size
        self.latent_dimension = latent_dimension
        self.activ = nn.ReLU() # if normalise else nn.Sigmoid()
        self.activ_f = nn.Sigmoid()
        self.upsamples = upsamples
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.convs = []
        self.bns = []

        in_c = latent_dimension//2
        for i in range(len(out_channels)):
            self.convs.append(nn.Conv2d(in_channels = in_c, out_channels = out_channels[i], kernel_size = kernels[i], stride = 1, padding = kernels[i]//2))
            if not i == (len(out_channels) -1):
                self.bns.append(nn.BatchNorm2d(out_channels[i]))
            in_c = out_channels[i]

        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

        self.gauss_map_dim = gauss_map_dim
        self.gauss_map_var = gauss_map_var

        #self.reg_cnn = CNNet(in_dim = 8, in_chans = latent_dimension//2, out_feat = 4, filters = [32,64,128,1], kernels = [3,3,1,1], dim_s = [2,1,1,1])

    def forward(self, input, output_map = False):
        #out = self.activ(self.decoder(x))
        #out = out.view(-1, 1, self.height//8, self.width//8)
        x = torch.cuda.FloatTensor(input)
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            if not i == (len(self.convs) - 1):
                x = self.bns[i](x)
                x = self.activ(x)
            if self.upsamples[i] == 1:
                x = self.up(x)

        if output_map: return self.activ_f(x[:,:,:self.height, :self.width]), keyp, map
        return self.activ_f(x[:,:,:self.height, :self.width]), input, "_"

class DeepSpatialAutoencoder(nn.Module):
    def __init__(self, image_output_size=(64, 64), in_channels=3, out_channels=(64, 64, 32, 16, 16), latent_dimension=32,
                 temperature=None, normalise=False):
        """
        Creates a deep spatial autoencoder. Default parameters are the ones used in [1], with the original input image
        being 3x240x240. See docs for encoder and decoder.
        :param image_output_size: Reconstructed image size
        :param in_channels: Number of channels of input image
        :param out_channels: Output channels of each conv layer in the encoder.
        :param latent_dimension: Input dimension for decoder
        :param temperature: Temperature parameter, None if it is to be learnt
        :param normalise: Should spatial features be normalised to [-1, 1]?
        """
        super().__init__()
        if out_channels[-1] * 2 != latent_dimension:
            raise ValueError("Spatial SoftArgmax produces a location (x,y) per feature map!")
        self.encoder = DSAE_Encoder(in_channels=in_channels, out_channels=out_channels, temperature=temperature,
                                    normalise=normalise)
        self.decoder = DSAE_Decoder(image_output_size=image_output_size, latent_dimension=latent_dimension, in_channels=in_channels)


    def forward(self, x):
        # (N, C, 2)
        spatial_features = self.encoder(x)
        n, c, _2 = spatial_features.size()
        # (N, C * 2 = latent dimension)
        return self.decoder(spatial_features.view(n, c * 2))





class DeepSpatialMapRegAutoencoder(nn.Module):
    def __init__(self, image_output_size=(64, 64), in_channels=3, out_channels=(64, 128, 256, 512, 16), latent_dimension=32,
                 temperature=None, normalise=False):
        """
        Creates a deep spatial autoencoder. Default parameters are the ones used in [1], with the original input image
        being 3x240x240. See docs for encoder and decoder.
        :param image_output_size: Reconstructed image size
        :param in_channels: Number of channels of input image
        :param out_channels: Output channels of each conv layer in the encoder.
        :param latent_dimension: Input dimension for decoder
        :param temperature: Temperature parameter, None if it is to be learnt
        :param normalise: Should spatial features be normaliseMasterchef, Antonio Lorenzon aprirà un hotel a Bassano del Grappad to [-1, 1]?
        """
        super().__init__()
        if out_channels[-1] * 2 != latent_dimension:
            raise ValueError("Spatial SoftArgmax produces a location (x,y) per feature map!")
        self.encoder = DSAE_Encoder(in_channels=in_channels, out_channels=out_channels, temperature=temperature,
                                    normalise=normalise)
        self.decoder = DSAE_Map_Reg_Decoder(image_output_size=image_output_size, latent_dimension=latent_dimension, in_channels=in_channels)

    def encode(self, x):
        spatial_features = self.encoder(x)
        return spatial_features

    def forward(self, x):
        # (N, C, 2)
        spatial_features = self.encoder(x)
        n, c, _2 = spatial_features.size()
        # (N, C * 2 = latent dimension)
        return self.decoder(spatial_features.view(n, c * 2))



class CustomDeepSpatialAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        """
        Same as DeepSpatialAutoencoder, but with your own custom modules
        :param encoder: Encoder
        :param decoder: Decoder
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        spatial_features = self.encoder(x)
        n, c, _2 = spatial_features.size()
        # (N, C * 2 = latent dimension)
        return self.decoder(spatial_features.view(n, c * 2))

class CustomAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        """
        Same as DeepSpatialAutoencoder, but with your own custom modules
        :param encoder: Encoder
        :param decoder: Decoder
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        features = self.encoder(x)
        # (N, C * 2 = latent dimension)
        return self.decoder(features)

class MultiCustomDeepSpatialAutoencoder(nn.Module):
    def __init__(self, encoder, decoders, output_maps = False, return_feature_maps = False):
        """
        Same as DeepSpatialAutoencoder, but with your own custom modules
        :param encoder: Encoder
        :param decoder: Decoder
        """
        super().__init__()
        self.encoder = encoder
        #self.decoders = decoders
        if not type(decoders) == list:
            self.decoders = nn.ModuleList([decoders])
        else:
            self.decoders = nn.ModuleList(decoders)
        self.output_maps = output_maps
        self.return_feature_maps = return_feature_maps

    def forward(self, x):
        if self.return_feature_maps:
            spatial_features, feature_maps = self.encoder(x)
        else:
            spatial_features = self.encoder(x)
        n, c, _2 = spatial_features.size()
        # (N, C * 2 = latent dimension)
        outs = [d(spatial_features.view(n, c * 2), output_map = self.output_maps) for d in self.decoders]

        if self.return_feature_maps:
            return outs, feature_maps
        return outs

class MultiCustomAutoencoder(nn.Module):
    def __init__(self, encoder, decoders, output_maps = False, return_feature_maps = False):
        """
        Same as DeepSpatialAutoencoder, but with your own custom modules
        :param encoder: Encoder
        :param decoder: Decoder
        """
        super().__init__()
        self.encoder = encoder
        #self.decoders = decoders
        if not type(decoders) == list:
            self.decoders = nn.ModuleList([decoders])
        else:
            self.decoders = nn.ModuleList(decoders)
        self.output_maps = output_maps
        self.return_feature_maps = return_feature_maps

    def forward(self, x):

        features = self.encoder(x)
        # (N, C * 2 = latent dimension)
        outs = [d(features, output_map = self.output_maps) for d in self.decoders]

        return outs

class ParallelCustomDeepSpatialAutoencoder(nn.Module):
    def __init__(self, encoders, decoders, output_maps = False, return_feature_maps = False):
        """
        Same as DeepSpatialAutoencoder, but with your own custom modules
        :param encoder: Encoder
        :param decoder: Decoder
        """
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        #self.decoders = decoders
        if not type(decoders) == list:
            self.decoders = nn.ModuleList([decoders])
        else:
            self.decoders = nn.ModuleList(decoders)
        self.output_maps = output_maps
        self.return_feature_maps = return_feature_maps

    def forward(self, x):
        out_d, keys = [], []
        for i in range(len(x)):

            x_i = x[i] #the n streams are attached on the first dim, the second in batch, etc
            spatial_features = self.encoders[i](x_i)
            n, c, _2 = spatial_features.size()
            outs = self.decoders[i](spatial_features.view(n, c*2), output_map = False)
            out_d.append(outs[0])
            keys.append(outs[1])
        keys = torch.cat((keys[0], keys[1]), -1)
        return out_d, keys


class DSAE_Loss(object):
    def __init__(self, add_g_slow=True):
        """
        Loss for deep spatial autoencoder.
        :param add_g_slow: Should g_slow contribution be added? See [1].
        """
        self.add_g_slow = add_g_slow
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.l1_loss = nn.L1Loss()

    def __call__(self, reconstructed, target, ft_minus1=None, ft=None, ft_plus1=None):
        """
        Performs the loss computation, and returns both loss components.
        For the start of a trajectory, where ft_minus1 = ft, simply pass in ft_minus1=ft, ft=ft
        For the end of a trajectory, where ft_plus1 = ft, simply pass in ft=ft, ft_plus1=ft
        :param reconstructed: Reconstructed, grayscale image
        :param target: Target, grayscale image
        :param ft_minus1: Features produced by the encoder for the previous image in the trajectory to the target one
        :param ft: Features produced by the encoder for the target image
        :param ft_plus1: Features produced by the encoder for the next image in the trajectory to the target one
        :return: A tuple (mse, g_slow) where mse = the MSE reconstruction loss and g_slow = g_slow contribution term ([1])
        """
        loss = self.mse_loss(reconstructed, target)
        loss += self.l1_loss(reconstructed, target)
        g_slow_contrib = torch.zeros(1, device=loss.device)
        if self.add_g_slow:
            g_slow_contrib = self.mse_loss(ft_plus1 - ft, ft - ft_minus1)
        return loss, g_slow_contrib


def create_gaussian_map(mu, im_dim, std):

    mu_x, mu_y = mu[:,::2][:,:,None], mu[:,1::2][:,:,None] #keypoints are (x1,y1,x2,y2,...)
    y = torch.from_numpy(np.linspace(-1,1,im_dim)).to(mu.device).double()
    x = torch.from_numpy(np.linspace(-1,1,im_dim)).to(mu.device).double()

    y = y.view((1,1,im_dim))
    x = x.view((1,1,im_dim))

    gauss_y = torch.exp(-(1e-4 + torch.pow((mu_y - y)*(1/std), 2))).double()
    gauss_x = torch.exp(-(1e-4 + torch.pow((mu_x - x)*(1/std), 2))).double()

#    print("gauss_y shape:", gauss_y.shape)
#    print("gauss_x shape:", gauss_x.shape)

    gauss_y = gauss_y.view((gauss_y.shape[0], gauss_y.shape[1], gauss_y.shape[2], 1)).double()
    gauss_x = gauss_x.view((gauss_x.shape[0], gauss_x.shape[1], 1, gauss_x.shape[2])).double()


    gauss_xy = torch.matmul(gauss_y, gauss_x)
#    print("square gauss_y shape:", gauss_xy.shape)
#    print("square gauss_x shape:", gauss_xy.shape)

    return gauss_xy

class FNN(nn.Module):
    def __init__(self, out_features=(512, 64)):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(in_features=16*2, out_features=out_features[0]),
            nn.BatchNorm1d(out_features[0]),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=out_features[0], out_features=out_features[1]),
            nn.BatchNorm1d(out_features[1]),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=out_features[1], out_features=4)
        )
    
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.sequence(x)
        return x

# class FNN(nn.Module):
#     def __init__(self, out_features=(512, 64)):
#         super().__init__()
       
#         self.l1 = nn.Linear(in_features=16*2, out_features=out_features[0]),
#         self.b1 = nn.BatchNorm1d(out_features[0]),
#         self.r1 = nn.ReLU(inplace=True)
#         self.l2 = nn.Linear(in_features=out_features[0], out_features=out_features[1]),
#         self.b2 = nn.BatchNorm1d(out_features[1]),
#         self.r2 = nn.ReLU(inplace=True)
#         self.l3 = nn.Linear(in_features=out_features[1], out_features=4)
    
    
#     def forward(self, x):
#         x = x.reshape(x.size(0), -1)
#         x = self.r1(self.b1(self.l1(x)))
#         x = self.r2(self.b2(self.l2(x)))
#         x = self.l3(x)
#         return x