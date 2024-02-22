import torch.nn as nn
import torch


class LWISP(nn.Module):

    def __init__(self, instance_norm=True, instance_norm_level_1=False):
        super(LWISP, self).__init__()

        
        self.conv_l1_d0 = ConvLayer(3, 16, kernel_size=3, stride=1)
        self.conv_l1_d1 = ConvMultiBlock(16, 16, 3, instance_norm=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.ca1 = channel_attention(16)
        self.sa1 = spatial_attention()
        
        self.conv_l1_d2 = ConvMultiBlock(32, 32, 3, instance_norm=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.ca2 = channel_attention(32)
        self.sa2 = spatial_attention()
        
        self.conv_l1_d3 = ConvMultiBlock(64, 64, 3, instance_norm=False)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.ca3 = channel_attention(64)
        self.sa3 = spatial_attention()
        
        self.conv_l1_d4 = ConvMultiBlock(128, 128, 3, instance_norm=False)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.ca4 = channel_attention(128)
        self.sa4 = spatial_attention()
        
        self.conv_l1_d5 = ConvLayer(256, 256, kernel_size=3, stride=1)
        self.fc1 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(True))
        
        self.conv_l1_d6 = ConvMultiBlock(256, 128, 3, instance_norm=False)
        self.conv_u1 = UpsampleConvLayer(128, 64, 3)
        self.conv_l1_d7 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.conv_u2 = UpsampleConvLayer(64, 32, 3)
        self.conv_l1_d8 = ConvLayer(64, 32, kernel_size=3, stride=1)
        self.conv_u3 = UpsampleConvLayer(32, 16, 3)
        self.conv_l1_d9 = ConvLayer(32, 16, kernel_size=3, stride=1)

        #self.conv_u4 = UpsampleConvLayer(16, 16, 3)
        self.conv_l1_d10 = ConvLayer(16, 3, kernel_size=3, stride=1, relu=False)
        self.output = nn.Sigmoid()

    def forward(self, x):

        conv_0 = self.conv_l1_d0(x)
        conv_1 = self.conv_l1_d1(conv_0)
        pool1 = self.pool1(conv_1)
        ca1_branch = self.ca1(pool1)
        medium11 = pool1 + ca1_branch
        sa1_branch = self.sa1(pool1)
        medium12 = pool1 + sa1_branch
        pool1 = torch.cat([medium11, medium12], dim=1)
        
        conv_2 = self.conv_l1_d2(pool1)
        pool2 = self.pool2(conv_2)
        ca2_branch = self.ca2(pool2)
        medium21 = pool2 + ca2_branch
        sa2_branch = self.sa2(pool2)
        medium22 = pool2 + sa2_branch
        pool2 = torch.cat([medium21, medium22], dim=1)
        
        conv_3 = self.conv_l1_d3(pool2)
        pool3 = self.pool3(conv_3)
        ca3_branch = self.ca3(pool3)
        medium31 = pool3 + ca3_branch
        sa3_branch = self.sa3(pool3)
        medium32 = pool3 + sa3_branch
        pool3 = torch.cat([medium31, medium32], dim=1)
        
        conv_4 = self.conv_l1_d4(pool3)
        pool4 = self.pool4(conv_4)
        ca4_branch = self.ca4(pool4)
        medium41 = pool4 + ca4_branch
        sa4_branch = self.sa4(pool4)
        medium42 = pool4 + sa4_branch
        pool4 = torch.cat([medium41, medium42], dim=1)
        
        conv_5 = self.conv_l1_d5(pool4)
        conv_global = torch.mean(conv_5, [2, 3])
        conv_dense = self.fc1(conv_global)
        feature = torch.unsqueeze(conv_dense, 2)
        feature = torch.unsqueeze(feature, 3)
        
        ones = torch.zeros(conv_4.shape).cuda()
        global_feature = feature + ones
        
        up6 = torch.cat([conv_4, global_feature], 1)
        conv_6 = self.conv_l1_d6(up6)
        sup1 = self.conv_u1(conv_4) ###
        up7 = self.conv_u1(conv_6)
        up7 = sup1 + up7 ###
        up7 = torch.cat([up7, conv_3], 1)
        conv_7 = self.conv_l1_d7(up7)
        sup2 = self.conv_u2(conv_3) ###
        up8 = self.conv_u2(conv_7)
        up8 = sup2 + up8 ###
        up8 = torch.cat([up8, conv_2], 1)
        conv_8 = self.conv_l1_d8(up8)
        sup3 = self.conv_u3(conv_2) ###
        up9 = self.conv_u3(conv_8)
        up9 = sup3 + up9 ###
        up9 = torch.cat([up9, conv_1], 1)
        conv_9 = self.conv_l1_d9(up9)
        
        conv_9 = conv_0 * conv_9
        #up10 = self.conv_u4(conv_9)
        conv_10 = self.conv_l1_d10(conv_9)
        enhanced = self.output(conv_10)

        return enhanced

class ConvMultiBlock(nn.Module):

    def __init__(self, in_channels, out_channels, max_conv_size, instance_norm):

        super(ConvMultiBlock, self).__init__()
        self.max_conv_size = max_conv_size

        self.conv_3a = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, instance_norm=instance_norm)
        self.conv_3b = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, instance_norm=instance_norm)

    def forward(self, x):

        out_3 = self.conv_3a(x)
        output_tensor = self.conv_3b(out_3)

        return output_tensor


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True, instance_norm=False):

        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2

        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        self.instance_norm = instance_norm
        self.instance = None
        self.relu = None

        if instance_norm:
            self.instance = nn.InstanceNorm2d(out_channels, affine=True)

        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.reflection_pad(x)
        out = self.conv2d(out)

        if self.instance_norm:
            out = self.instance(out)

        if self.relu:
            out = self.relu(out)

        return out


class UpsampleConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, upsample=2, stride=1, relu=True):

        super(UpsampleConvLayer, self).__init__()
        #self.upsample = nn.Upsample(scale_factor=upsample, mode='bilinear', align_corners=True)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)

        self.conv2d = torch.nn.Conv2d(in_channels, out_channels * 4, kernel_size, stride)

        if relu:
            self.relu = nn.LeakyReLU(0.2)
        
        self.upsample = nn.PixelShuffle(upsample) # in_channel -> in_channel / 4
        
        self.conv_extend = ConvLayer(out_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)

        if self.relu:
            out = self.relu(out)
            
        out = self.upsample(out)
        
        out = self.conv_extend(out)

        return out

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class spatial_attention(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attention, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class channel_attention(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(channel_attention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y