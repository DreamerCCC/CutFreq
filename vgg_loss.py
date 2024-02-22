
from torchvision import models
from collections import namedtuple
import torch.nn as nn
import torch

CONTENT_LAYER = 'relu_16'
CONV_LAYERS = ['conv_2', 'conv_4', 'conv_8', 'conv_12', 'conv_16']

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for i in range(3):
            self.slice1.add_module(str(i), vgg_pretrained_features[i])
        for i in range(4,8):
            self.slice2.add_module(str(i), vgg_pretrained_features[i])
        for i in range(9,17):
            self.slice3.add_module(str(i), vgg_pretrained_features[i])
        for i in range(18,26):
            self.slice4.add_module(str(i), vgg_pretrained_features[i])
        for i in range(27,35):
            self.slice5.add_module(str(i), vgg_pretrained_features[i])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_out1 = h
        h = self.slice2(h)
        h_out2 = h
        h = self.slice3(h)
        h_out3 = h
        h = self.slice4(h)
        h_out4 = h
        h = self.slice5(h)
        h_out5 = h
        vgg_outputs = namedtuple('VggOutputs', ['conv_2', 'conv_4', 'conv_8', 'conv_12', 'conv_16'])
        out = vgg_outputs(h_out1, h_out2, h_out3, h_out4, h_out5)
        return out