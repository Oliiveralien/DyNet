import torch.nn as nn

import dynconv
from .coordatt import  CoordAtt
from utils.config import BaseConfig
from torch.nn.utils import fuse_conv_bn_eval

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Standard residual block """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, sparse=False, type="spatial",use_ca = True,reduction=0):
        super(BasicBlock, self).__init__()
        assert groups == 1
        assert dilation == 1
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        self.downsample = downsample
        self.stride = stride
        self.sparse = sparse
        self.type = type
        self.use_ca = use_ca
        if use_ca:
            self.ca = CoordAtt(planes,planes)
        if sparse:
            if self.type == "spatial":
                self.sMasker = dynconv.SpatialMaskUnit(channels=inplanes, stride=stride, dilate_stride=1)
            elif self.type == "channel":
                # in the resnet basic block, the first convolution is already strided, so mask_stride = 1
                self.cMasker1 = dynconv.ChannelMaskUnit(in_channels=inplanes, out_channels=planes,reduction = reduction)
                self.cMasker2 = dynconv.ChannelMaskUnit(in_channels=planes, out_channels=planes,reduction = reduction)
            elif self.type == "mix":
                self.sMasker = dynconv.SpatialMaskUnit(channels=inplanes, stride=stride, dilate_stride=1)
                self.cMasker1 = dynconv.ChannelMaskUnit(in_channels=inplanes, out_channels=planes,reduction = reduction)
                self.cMasker2 = dynconv.ChannelMaskUnit(in_channels=planes, out_channels=planes,reduction = reduction)
            else:
                raise NotImplementedError
        self.fast = False

    def show_reduction_ratio(self,layer):
        str = "{}_conv1 cr:{:.3f}  sr:{:.3f}\n".format(layer,
                                                       self.conv1.__channel_reduction__,
                                                       self.conv1.__pixel_reduction__)
        str += "{}_conv2 cr:{:.3f}  sr:{:.3f}".format(layer,
                                                       self.conv2.__channel_reduction__,
                                                       self.conv2.__pixel_reduction__)
        return str

    def fuse_module(self):
        self.conv1 = fuse_conv_bn_eval(self.conv1,self.bn1)
        self.bn1 = None

        self.conv2 = fuse_conv_bn_eval(self.conv2,self.bn2)
        self.bn2 = None

        if self.downsample is not None and \
           isinstance(self.downsample, nn.Sequential) and \
           isinstance(self.downsample[0], nn.Conv2d) and \
           isinstance(self.downsample[1], nn.BatchNorm2d):

           self.downsample = nn.Sequential(fuse_conv_bn_eval(self.downsample[0],
                                                             self.downsample[1]))


    def forward_normal(self,input):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out

    def forward_channel(self, x, meta):
        m1 = self.cMasker1(x, x.shape[1], meta)
        m1_dict = {"channel":m1}
        x = dynconv.conv3x3(self.conv1, x, None, m1_dict)
        x = dynconv.bn_relu(self.bn1, self.relu, x, m1_dict)
        x = dynconv.apply_mask(x, m1_dict)
        m2 = self.cMasker2(x, m1, meta)
        m2_dict = {"channel":m2}
        x = dynconv.conv3x3(self.conv2, x, m1_dict, m2_dict)
        x = dynconv.bn_relu(self.bn2, None, x, m2_dict)
        out = dynconv.apply_mask(x, m2_dict)
        meta['masks'].append({"channel":m1})
        meta['masks'].append({"channel":m2})
        return out

    def forward_spatial(self, x, meta):
        m = self.sMasker(x, meta)
        mask_dilate, mask = m['dilate'], m['std']
        mask_dict = {"spatial":mask}
        dilate_dict = {"spatial":mask_dilate}
        x = dynconv.conv3x3(self.conv1, x, None, dilate_dict)
        x = dynconv.bn_relu(self.bn1, self.relu, x, dilate_dict)
        x = dynconv.conv3x3(self.conv2, x, dilate_dict, mask_dict)
        x = dynconv.bn_relu(self.bn2, None, x, mask_dict)
        x = dynconv.apply_mask(x, mask_dict)
        meta['masks'].append({"spatial":m})
        return x

    def forward_mix(self, x, meta):
        m = self.sMasker(x, meta)
        mask_dilate, mask = m['dilate'], m['std']
        m1 = self.cMasker1(x, x.shape[1], meta)
        mask1_dict = {"spatial":mask_dilate,'channel':m1}
        x = dynconv.conv3x3(self.conv1, x, None, mask1_dict)
        x = dynconv.bn_relu(self.bn1, self.relu, x, mask1_dict)
        x = dynconv.apply_mask(x, {'channel':m1})
        # x = dynconv.apply_mask(x, mask1_dict)
        m2 = self.cMasker2(x, m1, meta)
        mask2_dict = {"spatial":mask,"channel":m2}
        x = dynconv.conv3x3(self.conv2, x, mask1_dict, mask2_dict)
        x = dynconv.bn_relu(self.bn2, None, x, mask2_dict)
        out = dynconv.apply_mask(x, mask2_dict)
        meta['masks'].append({"spatial":[mask_dilate,mask],"channel":[m1,m2]})
        # meta['masks'].append(mask1_dict)
        # meta['masks'].append(mask2_dict)
        return out



    def forward(self, input):
        x, meta = input
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        if not self.sparse:
            out = self.forward_normal(x)
        else:
            assert meta is not None
            if self.type == "spaltial":
                out = self.forward_spatial(x, meta)
            elif self.type == "channel":
                out = self.forward_channel(x, meta)
            elif self.type == "mix":
                out = self.forward_mix(x, meta)
            if self.use_ca:
                out = self.ca(out)

        out = out + identity

        out = self.relu(out)
        return out, meta


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, sparse=False, type="spatial",use_ca = True,reduction = 0):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        print(f'Bottleneck - sparse: {sparse}: inp {inplanes}, hidden_dim {width}, ' +
              f'oup {planes * self.expansion}, stride {stride}')

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.sparse = sparse
        self.fast = False


        self.type = type
        self.use_ca = use_ca
        if use_ca:
            self.ca = CoordAtt(planes * self.expansion,planes * self.expansion)
        if sparse:
            if self.type == "spatial":
                # in the resnet basic block, the first convolution is already strided, so mask_stride = 1
                self.sMasker = dynconv.SpatialMaskUnit(channels=inplanes, stride=stride, dilate_stride=stride)
            elif self.type == "channel":
                self.cMasker1 = dynconv.ChannelMaskUnit(in_channels=inplanes, out_channels=width,reduction = reduction)
                self.cMasker2 = dynconv.ChannelMaskUnit(in_channels=width, out_channels=width,reduction = reduction)
                self.cMasker3 = dynconv.ChannelMaskUnit(in_channels=width, out_channels=planes * self.expansion,reduction = reduction)
            elif self.type == "mix":
                self.sMasker = dynconv.SpatialMaskUnit(channels=inplanes, stride=stride, dilate_stride=stride)
                self.cMasker1 = dynconv.ChannelMaskUnit(in_channels=inplanes, out_channels=width,reduction = reduction)
                self.cMasker2 = dynconv.ChannelMaskUnit(in_channels=width, out_channels=width,reduction = reduction)
                self.cMasker3 = dynconv.ChannelMaskUnit(in_channels=width, out_channels=planes * self.expansion,reduction = reduction)
            else:
                raise NotImplementedError

    def show_reduction_ratio(self,layer):
        str = "{}_conv1 cr:{:.3f}  sr:{:.3f}\n".format(layer,
                                                       self.conv1.__channel_reduction__,
                                                       self.conv1.__pixel_reduction__)
        str += "{}_conv2 cr:{:.3f}  sr:{:.3f}\n".format(layer,
                                                       self.conv2.__channel_reduction__,
                                                       self.conv2.__pixel_reduction__)
        str += "{}_conv3 cr:{:.3f}  sr:{:.3f}".format(layer,
                                                       self.conv3.__channel_reduction__,
                                                       self.conv3.__pixel_reduction__)                                                       
        return str

    def fuse_module(self):
        self.conv1 = fuse_conv_bn_eval(self.conv1,self.bn1)
        self.bn1 = None

        self.conv2 = fuse_conv_bn_eval(self.conv2,self.bn2)
        self.bn2 = None

        self.conv3 = fuse_conv_bn_eval(self.conv3,self.bn3)
        self.bn3 = None

        if self.downsample is not None and \
           isinstance(self.downsample, nn.Sequential) and \
           isinstance(self.downsample[0], nn.Conv2d) and \
           isinstance(self.downsample[1], nn.BatchNorm2d):

           self.downsample = nn.Sequential(fuse_conv_bn_eval(self.downsample[0],
                                                             self.downsample[1]))



    def forward_normal(self,x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        return out

    def forward_mix(self, x, meta):
        m = self.sMasker(x, meta)
        mask_dilate, mask = m['dilate'], m['std']

        m1 = self.cMasker1(x, x.shape[1], meta)
        mask1_dict = {"spatial":mask_dilate,'channel':m1}
        x = dynconv.conv1x1(self.conv1, x, None, mask1_dict)
        x = dynconv.bn_relu(self.bn1, self.relu, x, mask1_dict)
        x = dynconv.apply_mask(x, {'channel':m1})

        m2 = self.cMasker2(x, m1, meta)
        mask2_dict = {"spatial":mask,"channel":m2}
        x = dynconv.conv3x3(self.conv2, x, mask1_dict, mask2_dict)
        x = dynconv.bn_relu(self.bn2, self.relu, x, mask2_dict)
        x = dynconv.apply_mask(x, {'channel':m2})

        m3 = self.cMasker2(x, m2, meta)
        mask3_dict = {"spatial":mask,"channel":m3}
        x = dynconv.conv1x1(self.conv3, x, mask2_dict, mask3_dict)
        x = dynconv.bn_relu(self.bn2, None, x, mask3_dict)
        out = dynconv.apply_mask(x, mask3_dict)
        meta['masks'].append({"spatial":[mask_dilate,mask,mask],"channel":[m1,m2,m3]})
        return out



    def forward(self, input):
        x, meta = input
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if not self.sparse:
            out  = self.forward_normal(x)
        else:
            assert meta is not None
            if self.type == "mix":
                out = self.forward_mix(x, meta)
            if self.use_ca:
                out = self.ca(out)
        out = self.relu(out+identity)
        return out, meta

#
# import torch.nn as nn
#
# import dynconv
# from .coordatt import  CoordAtt
#
#
# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)
#
#
# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#
#
# class BasicBlock(nn.Module):
#     """Standard residual block """
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None, sparse=False, type="spatial",use_ca = True):
#         super(BasicBlock, self).__init__()
#         assert groups == 1
#         assert dilation == 1
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes, affine=True)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes, affine=True)
#         self.downsample = downsample
#         self.stride = stride
#         self.sparse = sparse
#         self.type = type
#         self.use_ca = use_ca
#         if use_ca:
#             self.ca = CoordAtt(planes,planes)
#
#         if sparse:
#             if self.type == "spatial":
#                 self.sMasker = dynconv.SpatialMaskUnit(channels=inplanes, stride=stride, dilate_stride=1)
#             elif self.type == "channel":
#                 # in the resnet basic block, the first convolution is already strided, so mask_stride = 1
#                 self.cMasker1 = dynconv.ChannelMaskUnit(in_channels=inplanes, out_channels=planes)
#                 self.cMasker2 = dynconv.ChannelMaskUnit(in_channels=planes, out_channels=planes)
#             elif self.type == "mix":
#                 self.sMasker1 = dynconv.SpatialMaskUnit(channels=inplanes, stride=stride, dilate_stride=1)
#                 self.sMasker2 = dynconv.SpatialMaskUnit(channels=inplanes, stride=1, dilate_stride=1)
#                 self.cMasker1 = dynconv.ChannelMaskUnit(in_channels=inplanes, out_channels=planes)
#                 self.cMasker2 = dynconv.ChannelMaskUnit(in_channels=planes, out_channels=planes)
#             else:
#                 raise NotImplementedError
#         self.fast = False
#
#     def forward_normal(self,input):
#         out = self.conv1(input)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         return out
#
#     def forward_channel(self, x, meta):
#         m1 = self.cMasker1(x, x.shape[1], meta)
#         m1_dict = {"channel":m1}
#         x = dynconv.conv3x3(self.conv1, x, None, m1_dict)
#         x = dynconv.bn_relu(self.bn1, self.relu, x, m1_dict)
#         x = dynconv.apply_mask(x, m1_dict)
#         m2 = self.cMasker2(x, m1, meta)
#         m2_dict = {"channel":m2}
#         x = dynconv.conv3x3(self.conv2, x, m1_dict, m2_dict)
#         x = dynconv.bn_relu(self.bn2, None, x, m2_dict)
#         out = dynconv.apply_mask(x, m2_dict)
#         meta['masks'].append({"channel":m1})
#         meta['masks'].append({"channel":m2})
#         return out
#
#     def forward_spatial(self, x, meta):
#         m = self.sMasker(x, meta)
#         mask_dilate, mask = m['dilate'], m['std']
#         mask_dict = {"spatial":mask}
#         dilate_dict = {"spatial":mask_dilate}
#         x = dynconv.conv3x3(self.conv1, x, None, dilate_dict)
#         x = dynconv.bn_relu(self.bn1, self.relu, x, dilate_dict)
#         x = dynconv.conv3x3(self.conv2, x, dilate_dict, mask_dict)
#         x = dynconv.bn_relu(self.bn2, None, x, mask_dict)
#         x = dynconv.apply_mask(x, mask_dict)
#         meta['masks'].append({"spatial":m})
#         return x
#
#     def forward_mix(self, x, meta):
#         sm1 = self.sMasker1(x, meta)
#
#         m1 = self.cMasker1(x, x.shape[1], meta)
#         mask1_dict = {"spatial":sm1,'channel':m1}
#         x = dynconv.conv3x3(self.conv1, x, None, mask1_dict)
#         x = dynconv.bn_relu(self.bn1, self.relu, x, mask1_dict)
#         x = dynconv.apply_mask(x, mask1_dict)
#         m2 = self.cMasker2(x, m1, meta)
#         sm2 = self.sMasker2(x, meta)
#         mask2_dict = {"spatial":sm2,"channel":m2}
#         x = dynconv.conv3x3(self.conv2, x, mask1_dict, mask2_dict)
#         x = dynconv.bn_relu(self.bn2, None, x, mask2_dict)
#         out = dynconv.apply_mask(x, mask2_dict)
#         meta['masks'].append(mask1_dict)
#         meta['masks'].append(mask2_dict)
#         return out
#
#     def forward_mix1(self, x, meta):
#         sm1 = self.sMasker1(x, meta)
#
#         m1 = self.cMasker1(x, x.shape[1], meta)
#         mask1_dict = {"spatial":sm1,'channel':m1}
#         x = dynconv.conv3x3(self.conv1, x, None, mask1_dict)
#         x = dynconv.bn_relu(self.bn1, self.relu, x, mask1_dict)
#         x = dynconv.apply_mask(x, mask1_dict)
#         m2 = self.cMasker2(x, m1, meta)
#         sm2 = self.sMasker2(x, meta)
#         mask2_dict = {"spatial":sm2,"channel":m2}
#         x = dynconv.conv3x3(self.conv2, x, mask1_dict, mask2_dict)
#         x = dynconv.bn_relu(self.bn2, None, x, mask2_dict)
#         out = dynconv.apply_mask(x, mask2_dict)
#         meta['masks'].append({"spatial":[sm1,sm2],"channel":[m1,m2]})
#         return out
#
#     def show_reduction_ratio(self,layer):
#         str = "{}_conv1 cr:{:.3f}  sr:{:.3f}\n".format(layer,
#                                                        self.conv1.__channel_reduction__,
#                                                        self.conv1.__pixel_reduction__)
#         str += "{}_conv2 cr:{:.3f}  sr:{:.3f}\n".format(layer,
#                                                        self.conv2.__channel_reduction__,
#                                                        self.conv2.__pixel_reduction__)
#         return str
#
#
#     def forward(self, input):
#         x, meta = input
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#         if not self.sparse:
#             out = self.forward_normal(x)
#         else:
#             assert meta is not None
#             if self.type == "spaltial":
#                 out = self.forward_spatial(x, meta)
#             elif self.type == "channel":
#                 out = self.forward_channel(x, meta)
#             elif self.type == "mix":
#                 out = self.forward_mix1(x, meta)
#
#         if self.use_ca:
#             out = self.ca(out)
#
#         out = out + identity
#
#         out = self.relu(out)
#         return out, meta
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None, sparse=False):
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups
#
#         print(f'Bottleneck - sparse: {sparse}: inp {inplanes}, hidden_dim {width}, ' +
#               f'oup {planes * self.expansion}, stride {stride}')
#
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#         self.sparse = sparse
#         self.fast = True
#
#         if sparse:
#             self.masker = dynconv.MaskUnit(channels=inplanes, stride=stride, dilate_stride=stride)
#
#     def forward(self, input):
#         x, meta = input
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         if not self.sparse:
#             out = self.conv1(x)
#             out = self.bn1(out)
#             out = self.relu(out)
#
#             out = self.conv2(out)
#             out = self.bn2(out)
#             out = self.relu(out)
#
#             out = self.conv3(out)
#             out = self.bn3(out)
#             out += identity
#         else:
#             assert meta is not None
#             m = self.masker(x, meta)
#             mask_dilate, mask = m['dilate'], m['std']
#
#             x = dynconv.conv1x1(self.conv1, x, mask_dilate)
#             x = dynconv.bn_relu(self.bn1, self.relu, x, mask_dilate)
#             x = dynconv.conv3x3(self.conv2, x, mask_dilate, mask)
#             x = dynconv.bn_relu(self.bn2, self.relu, x, mask)
#             x = dynconv.conv1x1(self.conv3, x, mask)
#             x = dynconv.bn_relu(self.bn3, None, x, mask)
#             out = identity + dynconv.apply_mask(x, mask)
#
#         out = self.relu(out)
#         return out, meta
