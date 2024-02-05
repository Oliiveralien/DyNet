# import torch.nn as nn

# import dynconv
# from .coordatt import  CoordAtt
# from utils.config import BaseConfig
# from sparse_conv2d import SparseConv2d


# def conv3x3(in_planes, out_planes, stride=1, groups=1, 
#             padding=1, dilation = 1, last_conv = False, activation=None):
#     """3x3 sparse convolution with padding"""
#     return SparseConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                         padding=padding,dilation = dilation, groups=groups, bias=True, 
#                         last_conv = last_conv, activation = activation)


# def conv1x1(in_planes, out_planes, stride=1, last_conv = False, activation=None):
#     """1x1 convolution"""
#     return SparseConv2d(in_planes, out_planes, kernel_size=1, stride=stride,bias=True, 
#                         last_conv = last_conv, activation = activation)


# class BasicBlock(nn.Module):
#     """Standard residual block """
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None ,use_ca = True,reduction=0):
#         super(BasicBlock, self).__init__()
#         assert groups == 1
#         assert dilation == 1
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d

#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = conv3x3(inplanes, planes, stride , last_conv=False, activation = self.relu) 
#         self.conv2 = conv3x3(planes, planes , last_conv=True, activation = None) 

#         self.sMasker = dynconv.SpatialMaskUnit(channels=inplanes, stride=stride, dilate_stride=1)
#         self.cMasker1 = dynconv.ChannelMaskUnit(in_channels=inplanes, out_channels=planes,reduction = reduction)
#         self.cMasker2 = dynconv.ChannelMaskUnit(in_channels=planes, out_channels=planes,reduction = reduction)

#         self.downsample = downsample

#         self.ca = None
#         if use_ca:
#             self.ca = CoordAtt(planes,planes)

#     def forward(self, input_x):
#         assert self.training == False , "only support eval mode!"
#         x, meta = input_x
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
        
#         m = self.sMasker(x, meta)
#         mask_dilate, mask = m['dilate'], m['std']

#         m1 = self.cMasker1(x, x.shape[1], meta)
#         x = self.conv1(x,{"prev_c_mask":None,
#                           "c_mask":m1.hard,
#                           "s_mask":mask_dilate.hard})
#         if x.size(1) == 0:
#             return  self.relu(identity), meta              
#         m2 = self.cMasker2(x, m1, meta)
#         x = self.conv2(x, {"prev_c_mask":m1.hard,
#                            "c_mask":m2.hard,
#                            "s_mask":mask.hard})
#         if x.size(1) == 0:
#             return  self.relu(identity),meta 

#         if self.ca is not None:
#             out = self.ca(x)

#         out = out + identity
#         out = self.relu(out)

#         return out,meta


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None, use_ca = True,reduction = 0):
#         super(Bottleneck, self).__init__()

#         assert groups == 1 and dilation == 1

#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups

#         print(f'Bottleneck - sparse: {sparse}: inp {inplanes}, hidden_dim {width}, ' +
#               f'oup {planes * self.expansion}, stride {stride}')

#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = conv1x1(inplanes, width, last_conv=False, activation = self.relu)
#         self.conv2 = conv3x3(width, width, stride, groups, padding = 1, 
#                              dilation = dilation, last_conv=False, activation = self.relu)
#         self.conv3 = conv1x1(width, planes * self.expansion, last_conv=True, activation = None)
#         self.downsample = downsample

#         self.ca = None
#         if use_ca:
#             self.ca = CoordAtt(planes * self.expansion,planes * self.expansion)
       
#         self.sMasker = dynconv.SpatialMaskUnit(channels=inplanes, stride=stride, dilate_stride=stride)
#         self.cMasker1 = dynconv.ChannelMaskUnit(in_channels=inplanes, out_channels=width,reduction = reduction)
#         self.cMasker2 = dynconv.ChannelMaskUnit(in_channels=width, out_channels=width,reduction = reduction)
#         self.cMasker3 = dynconv.ChannelMaskUnit(in_channels=width, out_channels=planes * self.expansion,reduction = reduction)



#     def forward(self, input):
#         x, meta = input
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
        
#         assert x.size(1) == 0, "over pruning"

#         m = self.sMasker(x, meta)
#         mask_dilate, mask = m['dilate'], m['std']

#         m1 = self.cMasker1(x, x.shape[1], meta)
#         x = self.conv1(x,{"prev_c_mask":None,
#                           "c_mask":m1.hard,
#                           "s_mask":mask_dilate.hard})
#         if x.size(1) == 0:
#             return  self.relu(identity)   

#         m2 = self.cMasker2(x, m1, meta)
#         x = self.conv2(x,{"prev_c_mask":m1.hard,
#                           "c_mask":m2.hard,
#                           "s_mask":mask.hard})

#         if x.size(1) == 0:
#             return  self.relu(identity) 

#         m3 = self.cMasker2(x, m2, meta)
#         x = self.conv3(x,{"prev_c_mask":m2.hard,
#                           "c_mask":m3.hard,
#                           "s_mask":mask.hard})
#         if x.size(1) == 0:
#             return  self.relu(identity) 

#         if self.ca is not None:
#             out = self.ca(out)

#         out = self.relu(out+identity)

#         return out, meta


import torch.nn as nn

import dynconv
from .coordatt import  CoordAtt
from utils.config import BaseConfig
from sparse_conv2d import SparseConv2d


def conv3x3(in_planes, out_planes, stride=1, groups=1, 
            padding=1, dilation = 1, last_conv = False, activation=None):
    """3x3 sparse convolution with padding"""
    return SparseConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=padding,dilation = dilation, groups=groups, bias=True, 
                        last_conv = last_conv, activation = activation)


def conv1x1(in_planes, out_planes, stride=1, last_conv = False, activation=None):
    """1x1 convolution"""
    return SparseConv2d(in_planes, out_planes, kernel_size=1, stride=stride,bias=True, 
                        last_conv = last_conv, activation = activation)


class BasicBlock(nn.Module):
    """Standard residual block """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None ,use_ca = True,reduction=0):
        super(BasicBlock, self).__init__()
        assert groups == 1
        assert dilation == 1
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride , last_conv=False, activation = self.relu) 
        self.conv2 = conv3x3(planes, planes , last_conv=True, activation = None) 

        self.sMasker = dynconv.SpatialMaskUnit(channels=inplanes, stride=stride, dilate_stride=1)
        self.cMasker1 = dynconv.ChannelMaskUnit(in_channels=inplanes, out_channels=planes,reduction = reduction)
        self.cMasker2 = dynconv.ChannelMaskUnit(in_channels=planes, out_channels=planes,reduction = reduction)

        self.downsample = downsample

        self.ca = None
        if use_ca:
            self.ca = CoordAtt(planes,planes)

    def forward(self, input_x):
        assert self.training == False , "only support eval mode!"
        x, meta = input_x
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        m = self.sMasker(x, meta)
        mask_dilate, mask = m['dilate'], m['std']

        m1 = self.cMasker1(x, x.shape[1], meta)
        x = self.conv1(x,{"prev_c_mask":None,
                          "c_mask":m1.hard,
                          "s_mask":mask_dilate.hard})
        if len(x.size()) == 1:
            return  self.relu(identity), meta              
        m2 = self.cMasker2(x, m1, meta)
        x = self.conv2(x, {"prev_c_mask":m1.hard,
                           "c_mask":m2.hard,
                           "s_mask":mask.hard})
        if len(x.size()) == 1:
            return  self.relu(identity), meta   

        if self.ca is not None:
            out = self.ca(x)

        out = out + identity
        out = self.relu(out)

        return out,meta


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_ca = True,reduction = 0):
        super(Bottleneck, self).__init__()

        assert groups == 1 and dilation == 1

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        print(f'Bottleneck - sparse: {sparse}: inp {inplanes}, hidden_dim {width}, ' +
              f'oup {planes * self.expansion}, stride {stride}')

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(inplanes, width, last_conv=False, activation = self.relu)
        self.conv2 = conv3x3(width, width, stride, groups, padding = 1, 
                             dilation = dilation, last_conv=False, activation = self.relu)
        self.conv3 = conv1x1(width, planes * self.expansion, last_conv=True, activation = None)
        self.downsample = downsample

        self.ca = None
        if use_ca:
            self.ca = CoordAtt(planes * self.expansion,planes * self.expansion)
       
        self.sMasker = dynconv.SpatialMaskUnit(channels=inplanes, stride=stride, dilate_stride=stride)
        self.cMasker1 = dynconv.ChannelMaskUnit(in_channels=inplanes, out_channels=width,reduction = reduction)
        self.cMasker2 = dynconv.ChannelMaskUnit(in_channels=width, out_channels=width,reduction = reduction)
        self.cMasker3 = dynconv.ChannelMaskUnit(in_channels=width, out_channels=planes * self.expansion,reduction = reduction)



    def forward(self, input):
        x, meta = input
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        assert x.size(1) == 0, "over pruning"

        m = self.sMasker(x, meta)
        mask_dilate, mask = m['dilate'], m['std']

        m1 = self.cMasker1(x, x.shape[1], meta)
        x = self.conv1(x,{"prev_c_mask":None,
                          "c_mask":m1.hard,
                          "s_mask":mask_dilate.hard})
        if x.size(1) == 0:
            return  self.relu(identity)   

        m2 = self.cMasker2(x, m1, meta)
        x = self.conv2(x,{"prev_c_mask":m1.hard,
                          "c_mask":m2.hard,
                          "s_mask":mask.hard})

        if x.size(1) == 0:
            return  self.relu(identity) 

        m3 = self.cMasker2(x, m2, meta)
        x = self.conv3(x,{"prev_c_mask":m2.hard,
                          "c_mask":m3.hard,
                          "s_mask":mask.hard})
        if x.size(1) == 0:
            return  self.relu(identity) 

        if self.ca is not None:
            out = self.ca(out)

        out = self.relu(out+identity)

        return out, meta

