import torch.nn as nn
import torch.nn.functional as F


latent_size = 256

generator = nn.Sequential(
    #in: latent_size x 1 x 1

    nn.ConvTranspose2d(latent_size, 512, kernel_size = 4, stride = 1, padding = 0, bias = False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    #out: 128x4x4


    nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    #out: 64x8x8

    

    nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    #out: 32x16x16



    nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    #out: 16x32x32

    

    nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(32),
    nn.ReLU(True),
    #out: 8x64x64 


    nn.ConvTranspose2d(32, 3, kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.Tanh()

    #out: 3 x 128 x 128
)