import math
from copy import copy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from models.common import *



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        """
        Initialize the Channel Attention module.

        Args:
            in_planes (int): Number of input channels.
            ratio (int): Reduction ratio for the hidden channels in the channel attention block.
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the Channel Attention module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            out (torch.Tensor): Output tensor after applying channel attention.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
            max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
            out = self.sigmoid(avg_out + max_out)
            return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """
        Initialize the Spatial Attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the Spatial Attention module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            out (torch.Tensor): Output tensor after applying spatial attention.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x = torch.cat([avg_out, max_out], dim=1)
            x = self.conv(x)
            return self.sigmoid(x)


class CBAM(nn.Module):
    # ch_in, ch_out, shortcut, groups, expansion, ratio, kernel_size
    def __init__(self, c1, c2, kernel_size=3, shortcut=True, g=1, e=0.5, ratio=16):
        """
        Initialize the CBAM (Convolutional Block Attention Module) .

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            shortcut (bool): Whether to use a shortcut connection.
            g (int): Number of groups for grouped convolutions.
            e (float): Expansion factor for hidden channels.
            ratio (int): Reduction ratio for the hidden channels in the channel attention block.
        """
        super(CBAM, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.channel_attention = ChannelAttention(c2, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Forward pass of the CBAM .

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            out (torch.Tensor): Output tensor after applying the CBAM bottleneck.
        """
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')
        x2 = self.cv2(self.cv1(x))
        out = self.channel_attention(x2) * x2
        out = self.spatial_attention(out) * out
        return x + out if self.add else out


class Involution(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride):
        """
        Initialize the Involution module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            kernel_size (int): Size of the involution kernel.
            stride (int): Stride for the involution operation.
        """
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.c1 = c1
        reduction_ratio = 1
        self.group_channels = 16
        self.groups = self.c1 // self.group_channels
        # self.conv1 = Conv(c1, c1 // reduction_ratio, 1, 1)
        self.conv2 = Conv(c1 // reduction_ratio, kernel_size ** 2 * self.groups, 1)

        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

    def forward(self, x):
        """
        Forward pass of the Involution module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            out (torch.Tensor): Output tensor after applying the involution operation.
        """
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')
        weight = self.conv2(x)
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.c1, h, w)

        return out

class SENet(nn.Module):
    def __init__(self, in_channel, reduction_ratio):
        super(SENet, self).__init__()
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=in_channel, out_features=round(in_channel/reduction_ratio))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=round(in_channel/reduction_ratio), out_features=in_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.globalAvgPool(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * x
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttn(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttn, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

###

class CFEM(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 3, 5, 7]):
        super(CFEM, self).__init__()
        self.dilated_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rates[0], dilation=dilation_rates[0])
        self.dilated_conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rates[1], dilation=dilation_rates[1])
        self.dilated_conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rates[2], dilation=dilation_rates[2])
        self.dilated_conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rates[3], dilation=dilation_rates[3])
        self.deform_conv = DeformConv2d(len(dilation_rates) * out_channels, out_channels, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(len(dilation_rates) * out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply dilated convolutions in parallel
        conv_out1 = self.deform_conv(self.dilated_conv1(x) * 1)
        conv_out2 = self.deform_conv(self.dilated_conv2(x) * 3)
        conv_out3 = self.deform_conv(self.dilated_conv3(x) * 5)
        conv_out4 = self.deform_conv(self.dilated_conv4(x) * 7)
        # Concatenate the outputs
        combined_features = torch.cat([conv_out1, conv_out2, conv_out3, conv_out4], dim=1)
        # Fuse features using 1x1 convolution
        enhanced_features = self.conv(combined_features)
        return self.relu(enhanced_features)

###############
# ECAP-YOLO
class ECA(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ECABottleneck(nn.Module):
    def __init__(self, channel):
        super(ECABottleneck, self).__init__()
        self.conv = Conv(channel, channel, 1, 1)
        self.eca = ECA(channel)
    def forward(self, x):
        identity = x.clone()
        x = self.conv(x)
        x = self.eca(x)
        x = self.conv(x)
        return x + identity

class EAC3_n(nn.Module):
    def __init__(self, in_channels, out_channels, n, k=1, s=1):
        super(EAC3_n, self).__init__()
        self.conv = Conv(in_channels, in_channels, k, s)
        self.bottleneck = nn.Sequential(*[ECABottleneck(in_channels) for _ in range(n)])
    def forward(self, x):
        out1 = self.conv(x)
        out1 = self.bottleneck(out1)
        out2 = self.conv(x)

        out = torch.cat((out1, out2), dim=1)
        out = self.conv(out)
        return out

class EAC3_nF(nn.Module):
    def __init__(self, in_channels, out_channels, k, s):
        super(EAC3_nF, self).__init__()
        self.conv = Conv(in_channels, in_channels, k, s)
        self.eca = ECA(in_channels)
    def forward(self, x):
        out1 = self.conv(x)
        out1 = self.conv(out1)
        out1 = self.eca(out1)
        out1 = self.conv(out1)
        out2 = self.conv(x)

        out = torch.cat((out1, out2), dim=1)
        out = self.conv(out)
        return out

class MEAC(nn.Module):
    """Constructs a MEAC module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(MEAC, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class MEAP(nn.Module):
    def __init__(self, channel, out_channel):
        super(MEAP, self).__init__()
        self.conv = Conv(channel, channel, 1, 1)
        self.meac1 = MEAC(channel, k_size=5)
        self.meac2 = MEAC(channel, k_size=9)
        self.meac3 = MEAC(channel, k_size=13)
    def forward(self, x):
        x = self.conv(x)
        out1 = self.meac1(x)
        out2 = self.meac2(x)
        out3 = self.meac3(x)
        out = torch.cat((out1, out2, out3), dim=1)
        out = self.conv(out)
        return out

### SP-YOLO
class SPD(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super(SPD, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

### LE-YOLO
class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1):
        super().__init__()
        p = p if isinstance(p, str) else (k - 1) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class HGStem(nn.Module):
    # Stem
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(HGStem, self).__init__()
        c_ = int(c2/2)  # hidden channels
        self.conv1 = ConvBNAct(c1, c_, 3, 2)
        self.conv2 = ConvBNAct(c_, (c_ // 2), 2, 1, p="same")
        self.conv3 = ConvBNAct((c_ // 2), c_, 2, 1, p="same")
        self.conv4 = ConvBNAct(c_*2, c_, 3, 2)
        self.conv5 = ConvBNAct(c_, c2, 1, 1)
        self.pool = torch.nn.MaxPool2d(3, stride=1, ceil_mode=True, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        out = torch.cat((self.conv3(self.conv2(x)), self.pool(x)), dim=1)
        out = self.conv4(out)
        out = self.conv5(out)
        return out

class LHGBlock(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):
        super(LHGBlock, self).__init__()
        c_ = int(c2/2)
        self.dwconv1 = DWConv(c1, c1, k, s, act=False)
        self.dwconv2 = DWConv(c1, c_, k, s, act=False)
        self.conv1 = Conv(c_, c_, k, s)
        self.conv2 = Conv(c_, c2, k, s)
        self.cs = nn.ChannelShuffle(1)

    def forward(self, x):
        identity = x
        out = self.dwconv1(x)
        out = self.dwconv2(out)
        out = torch.cat((out, identity), dim=1)
        out = self.conv1(out)
        out = self.cs(out)
        out = self.conv2(out)
        out += identity
        return out

class ConvMish(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Mish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = ConvMish(c1, c_, k, s, None, g, act)
        self.cv2 = ConvMish(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)

class LGSBottleneck(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):
        super(LGSBottleneck, self).__init__()
        c_ = c2 // 2
        self.gsconv1 = GSConv(c1, c_, k, s)
        self.gsconv1 = GSConv(c_, c_, k, s)
        self.conv1 = Conv(c1, c_, k, s)
        self.conv2 = Conv(c_, c_, k, s)
    def forward(self, x):
        out1 = self.gsconv1(x)
        out1 = self.gsconv2(out1)
        out2 = self.conv1(x)
        out2 = self.conv2(out2)
        out = torch.cat((out1, out2), dim=1)
        return out

class LGSCSP(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):
        super(LGSCSP, self).__init__()
        c_ = c2 // 2
        self.conv1 = Conv(c1, c_, k, s)
        self.conv2 = Conv(c_*2, c2, 3, 2)
        self.bottleneck = LGSBottleneck(c_, c_)
    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bottleneck(out1)
        out2 = self.conv1(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.conv2(out)
        return out

### GCL-YOLO
class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super(GhostBottleneck, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


### YOLO-TLA
class CrossConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1):
        super(CrossConv, self).__init__()
        self.conv1 = Conv(in_channels, in_channels, (1, k), (1, s))
        self.conv2 = Conv(in_channels, in_channels, (k, 1), (s, 1))
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class GAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAM, self).__init__()
        self.fc = nn.Linear(in_channels, in_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        out1 = self.channel_attention(x)
        out1 = out1 * x
        out2 = self.spatial_attention(out1)
        out = out2 * out1
        return out

    def channel_attention(self, x):
        identity = x.clone()
        x = x.permute(0, 3, 2, 1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        x = x.permute(0, 1, 2, 3)
        return x + identity

    def spatial_attention(self, x):
        identity = x.clone()
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv(x)
        return x + identity

### DC-YOLO
class MDC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MDC, self).__init__()
        self.conv1 = Conv(in_channels, in_channels, 1, 1)
        self.conv2 = Conv(in_channels, in_channels, 3, 2)
        self.conv3 = Conv(in_channels*3, out_channels, 1, 1)
        self.dwconv1 = DWConv(in_channels, in_channels, 3, 2)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        x = self.conv1(x)
        out1 = self.conv2(x)
        out2 = self.dwconv1(x)
        out3 = self.pool(x)
        out = torch.cat((out1, out2, out3), dim=1)
        out = self.conv3(out)
        return out

class DC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DC, self).__init__()
        self.conv1 = Conv(in_channels, in_channels, 1, 1)
        self.conv2 = Conv(in_channels, in_channels, 3, 1)
        self.conv3 = Conv(in_channels*2, out_channels, 1, 1)
        self.dwconv1 = DWConv(in_channels, in_channels, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        out1 = self.conv2(x)
        out2 = self.dwconv1(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.conv3(out)
        return out
