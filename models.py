"""
Implementation of models in paper:
MVTec,
Uninformed Students: Student–Teacher Anomaly Detection with Discriminative Latent Embeddings.
CVPR, 2020.

Author: Luyao Chen
Date: 2020.10
"""

import torch
import numpy as np
from torch import nn
from fast_dense_feature_extractor import *



class _Teacher17(nn.Module):
    """
    T^ net for patch size 17. 
    """

    def __init__(self):
        super(_Teacher17, self).__init__()
        self.net = nn.Sequential(
            # Input n*3*17*17
            nn.Conv2d(3, 128, kernel_size=6, stride=1), # ???? kernel_size=5????
            nn.LeakyReLU(5e-3),
            # n*128*12*12
            nn.Conv2d(128, 256, kernel_size=5, stride=1),
            nn.LeakyReLU(5e-3),
            # n*256*8*8
            nn.Conv2d(256, 256, kernel_size=5, stride=1),
            nn.LeakyReLU(5e-3),
            # n*256*4*4
            nn.Conv2d(256, 128, kernel_size=4, stride=1),
            # n*128*1*1
        )
        self.decode = nn.Linear(128, 512)
            # nn.Sequential(
            # # nn.LeakyReLU(5e-3),
            # # # n*128*1*1
            # # nn.Conv2d(128, 512, kernel_size=1, stride=1),
            # # output n*512*1*1
            # )

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze().view(-1, 128)
        x = self.decode(x)
        return x


class _Teacher33(nn.Module):
    """
    T^ net for patch size 33.
    """

    def __init__(self):
        super(_Teacher33, self).__init__()
        self.net = nn.Sequential(
            # Input n*3*33*33
            nn.Conv2d(3, 128, kernel_size=3, stride=1),
            nn.LeakyReLU(5e-3),
            # n*128*29*29
            nn.MaxPool2d(kernel_size=2, stride=2),
            # n*128*14*14
            nn.Conv2d(128, 256, kernel_size=5, stride=1),
            nn.LeakyReLU(5e-3),
            # n*256*10*10
            nn.MaxPool2d(kernel_size=2, stride=2),
            # n*256*5*5
            nn.Conv2d(256, 256, kernel_size=2, stride=1),
            nn.LeakyReLU(5e-3),
            # n*256*4*4
            nn.Conv2d(256, 128, kernel_size=4, stride=1),
            # n*128*1*1
        )
        self.decode = nn.Linear(128, 512)

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze().view(-1, 128)
        x = self.decode(x)
        return x


class _Teacher65(nn.Module):
    """
    T^ net for patch size 65.
    """

    def __init__(self):
        super(_Teacher65, self).__init__()
        self.net = nn.Sequential(
            # Input n*3*65*65
            nn.Conv2d(3, 128, kernel_size=5, stride=1),
            nn.LeakyReLU(5e-3),
            # n*128*61*61
            nn.MaxPool2d(kernel_size=2, stride=2),
            # n*128*30*30
            nn.Conv2d(128, 128, kernel_size=5, stride=1),
            nn.LeakyReLU(5e-3),
            # n*128*26*26
            nn.MaxPool2d(kernel_size=2, stride=2),
            # n*128*13*13
            nn.Conv2d(128, 128, kernel_size=5, stride=1),
            nn.LeakyReLU(5e-3),
            # n*128*9*9
            nn.MaxPool2d(kernel_size=2, stride=2),
            # n*256*4*4
            nn.Conv2d(128, 256, kernel_size=4, stride=1),
            nn.LeakyReLU(5e-3),
            # n*256*1*1
            nn.Conv2d(256, 128, kernel_size=1, stride=1), # ???? kernel_size=3????
            # n*128*1*1
        )
        self.decode = nn.Linear(128, 512)

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze().view(-1, 128)
        x = self.decode(x)
        return x


class Teacher17(nn.Module):
    """
    Teacher network with patch size 17.
    It has same architecture as T^17 because with no striding or pooling layers.
    """

    def __init__(self, base_net: _Teacher17):
        super(Teacher17, self).__init__()
        self.multiPoolPrepare = multiPoolPrepare(17,17)
        self.net = base_net.net

    def forward(self, x):
        x = self.multiPoolPrepare(x)
        x = self.net(x)
        return x


class Teacher33(nn.Module):
    """
    Teacher network with patch size 33.
    """

    def __init__(self, base_net: _Teacher33, imH, imW):
        super(Teacher33, self).__init__()
        self.imH = imH
        self.imW = imW
        self.sL1 = 2
        self.sL2 = 2
        # image height and width should be multiples of sL1∗sL2∗sL3...
        # self.imW = int(np.ceil(imW / (self.sL1 * self.sL2)) * self.sL1 * self.sL2)
        # self.imH = int(np.ceil(imH / (self.sL1 * self.sL2)) * self.sL1 * self.sL2)
        assert imH % (self.sL1 * self.sL2) == 0, \
            "image height should be multiples of (sL1∗sL2) which is " + \
            str(self.sL1 * self.sL2)
        assert imW % (self.sL1 * self.sL2) == 0, \
            "image width should be multiples of (sL1∗sL2) which is " + \
            str(self.sL1 * self.sL2)

        self.outChans = base_net.net[-1].out_channels
        self.net = nn.Sequential(
            multiPoolPrepare(33, 33),
            base_net.net[0],
            base_net.net[1],
            multiMaxPooling(self.sL1, self.sL1, self.sL1, self.sL1),
            base_net.net[3],
            base_net.net[4],
            multiMaxPooling(self.sL2, self.sL2, self.sL2, self.sL2),
            base_net.net[6],
            base_net.net[7],
            base_net.net[8],
            unwrapPrepare(),
            unwrapPool(self.outChans, imH / (self.sL1*self.sL2),
                       imW / (self.sL1*self.sL2), self.sL2, self.sL2),
            unwrapPool(self.outChans, imH / self.sL1,
                       imW / self.sL1, self.sL1, self.sL1),
        )
    
    def forward(self, x):
        x = self.net(x)
        x = x.permute(5,0,1,2,3,4)
        x = x.view(x.shape[0], -1, self.imH, self.imW)
        return x


class Teacher65(nn.Module):
    """
    Teacher network with patch size 65.
    """

    def __init__(self, base_net: _Teacher65, imH, imW):
        super(Teacher65, self).__init__()
        self.imH = imH
        self.imW = imW
        self.sL1 = 2
        self.sL2 = 2
        self.sL3 = 2
        # image height and width should be multiples of sL1∗sL2∗sL3...
        # self.imW = int(np.ceil(imW / (self.sL1 * self.sL2)) * self.sL1 * self.sL2)
        # self.imH = int(np.ceil(imH / (self.sL1 * self.sL2)) * self.sL1 * self.sL2)
        assert imH % (self.sL1 * self.sL2 * self.sL3) == 0, \
            'image height should be multiples of (sL1∗sL2*sL3) which is ' + \
            str(self.sL1 * self.sL2 * self.sL3) + '.'
        assert imW % (self.sL1 * self.sL2 * self.sL3) == 0, \
            'image width should be multiples of (sL1∗sL2*sL3) which is ' + \
            str(self.sL1 * self.sL2 * self.sL3) + '.'

        self.outChans = base_net.net[-1].out_channels
        self.net = nn.Sequential(
            multiPoolPrepare(65, 65),
            base_net.net[0],
            base_net.net[1],
            multiMaxPooling(self.sL1, self.sL1, self.sL1, self.sL1),
            base_net.net[3],
            base_net.net[4],
            multiMaxPooling(self.sL2, self.sL2, self.sL2, self.sL2),
            base_net.net[6],
            base_net.net[7],
            multiMaxPooling(self.sL3, self.sL3, self.sL3, self.sL3),
            base_net.net[9],
            base_net.net[10],
            base_net.net[11],
            unwrapPrepare(),
            unwrapPool(self.outChans, imH / (self.sL1 * self.sL2 * self.sL3),
                       imW / (self.sL1 * self.sL2 * self.sL3), self.sL3, self.sL3),
            unwrapPool(self.outChans, imH / (self.sL1 * self.sL2),
                       imW / (self.sL1 * self.sL2), self.sL2, self.sL2),
            unwrapPool(self.outChans, imH / self.sL1,
                       imW / self.sL1, self.sL1, self.sL1),
        )

    def forward(self, x):
        x = self.net(x)
        print(x.shape)
        x = x.permute(5,0,1,2,3,4)
        x = x.view(x.shape[0], -1, self.imH, self.imW)
        return x


if __name__ == "__main__":
    net = _Teacher65()
    imH = 128
    imW = 128
    pH = 65
    pW = 65

    T = Teacher65(net, imH, imW)
    # T = Teacher33(net, imH, imW)
    x = torch.ones((2, 3, imH, imW))

    x_ = torch.ones((1, 3, pH, pW))

    y = T(x)
    y_ = net(x_)

    # print(y)
    print(y.shape)
    print(y_.shape)
    # print(T)
