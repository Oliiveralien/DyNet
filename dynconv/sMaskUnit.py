# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from utils.logger import recoder
#
#
# class SpatialMask():
#     '''
#     Class that holds the mask properties
#
#     hard: the hard/binary mask (1 or 0), 4-dim tensor
#     soft (optional): the float mask, same shape as hard
#     active_positions: the amount of positions where hard == 1
#     total_positions: the total amount of positions
#                         (typically batch_size * output_width * output_height)
#     '''
#     def __init__(self, hard, soft=None):
#         assert hard.dim() == 4
#         assert hard.shape[1] == 1
#         assert soft is None or soft.shape == hard.shape
#
#         self.kernel_size = 0
#         self.batch_size = hard.shape[0]
#         self.hard = hard
#         self.spatial_inte_state = torch.sum(hard,dim = (1,2,3)).float() #--->[N]
#         # print(self.hard.size(),self.spatial_inte_state.size())
#         self.active_positions = torch.sum(hard) # this must be kept backpropagatable!
#         self.total_positions = hard.numel()
#         self.soft = soft
#
#         self.flops_per_position = 0
#
#     def size(self):
#         return self.hard.shape
#
#     def __repr__(self):
#         return 'Mask with {}/{} positions, and {} accumulated FLOPS per position'.format(
#             self.active_positions,self.total_positions,self.flops_per_position)
#
# class SpatialMaskUnit(nn.Module):
#     '''
#     Generates the mask and applies the gumbel softmax trick
#     '''
#
#     def __init__(self, channels, stride=1, dilate_stride=1):
#         super(SpatialMaskUnit, self).__init__()
#         #self.maskconv = Squeeze(channels=channels, stride=stride)
#         self.maskconv = SpatialAttention(stride=stride)
#         self.gumbel = Gumbel()
#         self.expandmask = ExpandMask(stride=dilate_stride)
#
#     def forward(self, x, meta):
#         soft = self.maskconv(x)
#         hard = self.gumbel(soft, meta['gumbel_temp'], meta['gumbel_noise'])
#         mask = SpatialMask(hard, soft)
#
#         hard_dilate = self.expandmask(mask.hard)
#         mask_dilate = SpatialMask(hard_dilate)
#         m = {'std': mask, 'dilate': mask_dilate}
#         return m
#
# # class SpatialMaskUnit(nn.Module):
# #     '''
# #     Generates the mask and applies the gumbel softmax trick
# #     '''
# #
# #     def __init__(self, channels, stride=1, dilate_stride=1):
# #         super(SpatialMaskUnit, self).__init__()
# #         # self.maskconv = Squeeze(channels=channels, stride=stride)
# #         self.maskconv = SpatialAttention(stride=stride)
# #         self.gumbel = Gumbel()
# #
# #     def forward(self, x, meta):
# #         soft = self.maskconv(x)
# #         hard = self.gumbel(soft, meta['gumbel_temp'], meta['gumbel_noise'])
# #         mask = SpatialMask(hard, soft)
# #         return mask
#
# ## Gumbel
#
# class Gumbel(nn.Module):
#     '''
#     Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x.
#     '''
#     def __init__(self, eps=1e-8):
#         super(Gumbel, self).__init__()
#         self.eps = eps
#
#     def forward(self, x, gumbel_temp=1.0, gumbel_noise=True):
#         if not self.training:  # no Gumbel noise during inference
#             return (x >= 0).float()
#
#         recoder.add('gumbel_noise', gumbel_noise)
#         recoder.add('gumbel_temp', gumbel_temp)
#
#         if gumbel_noise:
#             eps = self.eps
#             U1, U2 = torch.rand_like(x), torch.rand_like(x)
#             g1, g2 = -torch.log(-torch.log(U1 + eps)+eps), - \
#                 torch.log(-torch.log(U2 + eps)+eps)
#             x = x + g1 - g2
#
#         soft = torch.sigmoid(x / gumbel_temp)
#         hard = ((soft >= 0.5).float() - soft).detach() + soft
#         assert not torch.any(torch.isnan(hard))
#         return hard
#
# ## Mask convs
# class Squeeze(nn.Module):
#     """
#     Squeeze module to predict masks
#     """
#
#     def __init__(self, channels, stride=1):
#         super(Squeeze, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(channels, 1, bias=True)
#         self.conv = nn.Conv2d(channels, 1, stride=stride,
#                               kernel_size=3, padding=1, bias=True)
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, 1, 1, 1)
#         z = self.conv(x)
#         return z + y.expand_as(z)
#
# def avg_pool_func(x):
#     return torch.mean(x, dim=1, keepdim=True)
#
#
# def max_pool_func(x):
#     x, _ = torch.max(x, dim=1, keepdim=True)
#     return x
#
#
# ## Mask convs
# class SpatialAttention(nn.Module):
#     """
#     Squeeze module to predict masks
#     """
#
#     def __init__(self, stride=1):
#         super(SpatialAttention, self).__init__()
#         kernel_size = 7
#         padding = 3
#         self.conv = nn.Conv2d(2, 1, kernel_size, stride, padding, bias=False)
#
#     def forward(self, x):
#         x = torch.cat([avg_pool_func(x), max_pool_func(x)], dim=1)
#         x = self.conv(x)
#         return x
#
# class ExpandMask(nn.Module):
#     def __init__(self, stride, padding=1):
#         super(ExpandMask, self).__init__()
#         self.stride=stride
#         self.padding = padding
#
#
#     def forward(self, x):
#         assert x.shape[1] == 1
#
#         if self.stride > 1:
#             self.pad_kernel = torch.zeros( (1,1,self.stride, self.stride), device=x.device)
#             self.pad_kernel[0,0,0,0] = 1
#         self.dilate_kernel = torch.ones((1,1,1+2*self.padding,1+2*self.padding), device=x.device)
#
#         x = x.float()
#         if self.stride > 1:
#             x = F.conv_transpose2d(x, self.pad_kernel, stride=self.stride, groups=x.size(1))
#         x = F.conv2d(x, self.dilate_kernel, padding=self.padding, stride=1)
#         return x > 0.5


import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import recoder
from utils.config import BaseConfig


class SpatialMask():
    '''
    Class that holds the mask properties

    hard: the hard/binary mask (1 or 0), 4-dim tensor
    soft (optional): the float mask, same shape as hard
    active_positions: the amount of positions where hard == 1
    total_positions: the total amount of positions
                        (typically batch_size * output_width * output_height)
    '''
    def __init__(self, hard, soft=None):
        assert hard.dim() == 4
        assert hard.shape[1] == 1
        assert soft is None or soft.shape == hard.shape

        self.kernel_size = 0
        self.batch_size = hard.shape[0]
        self.hard = hard
        self.spatial_inte_state = torch.sum(hard,dim = (1,2,3)).float() #--->[N]
        # print(self.hard.size(),self.spatial_inte_state.size())
        self.active_positions = torch.sum(hard) # this must be kept backpropagatable!
        self.total_positions = hard.numel()
        self.soft = soft

        self.flops_per_position = 0

    def size(self):
        return self.hard.shape

    def __repr__(self):
        return 'Mask with {}/{} positions, and {} accumulated FLOPS per position'.format(
            self.active_positions,self.total_positions,self.flops_per_position)

class SpatialMaskUnit(nn.Module):
    '''
    Generates the mask and applies the gumbel softmax trick
    '''

    def __init__(self, channels, stride=1, dilate_stride=1):
        super(SpatialMaskUnit, self).__init__()
        #self.maskconv = Squeeze(channels=channels, stride=stride)
        self.maskconv = SpatialAttention(stride=stride)
        self.attenconv = SpatialAttention(stride=stride)
        self.gumbel = Gumbel()
        self.expandmask = ExpandMask(stride=dilate_stride)

    def forward(self, x, meta):
        soft = self.maskconv(x)
        # atten = torch.sigmoid(self.attenconv(x))
        hard = self.gumbel(soft, meta['gumbel_temp'], meta['gumbel_noise'])
        mask = SpatialMask(hard, soft)

        hard_dilate = self.expandmask(mask.hard)
        mask_dilate = SpatialMask(hard_dilate)
        m = {'std': mask, 'dilate': mask_dilate}
        return m

# class SpatialMaskUnit(nn.Module):
#     '''
#     Generates the mask and applies the gumbel softmax trick
#     '''
#
#     def __init__(self, channels, stride=1, dilate_stride=1):
#         super(SpatialMaskUnit, self).__init__()
#         # self.maskconv = Squeeze(channels=channels, stride=stride)
#         self.maskconv = SpatialAttention(stride=stride)
#         self.gumbel = Gumbel()
#
#     def forward(self, x, meta):
#         soft = self.maskconv(x)
#         hard = self.gumbel(soft, meta['gumbel_temp'], meta['gumbel_noise'])
#         mask = SpatialMask(hard, soft)
#         return mask

## Gumbel

class Gumbel(nn.Module):
    '''
    Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x.
    '''
    def __init__(self, eps=1e-8):
        super(Gumbel, self).__init__()
        self.eps = eps

    def forward(self, x, gumbel_temp=1.0, gumbel_noise=True):
        # no Gumbel noise during inference with sparse conv
        if not self.training and BaseConfig.inference == True:  
            return (x >= 0)  

        if not self.training:  # no Gumbel noise during inference
            return (x >= 0).float()

        recoder.add('gumbel_noise', gumbel_noise)
        recoder.add('gumbel_temp', gumbel_temp)

        if gumbel_noise:
            eps = self.eps
            U1, U2 = torch.rand_like(x), torch.rand_like(x)
            g1, g2 = -torch.log(-torch.log(U1 + eps)+eps), - \
                torch.log(-torch.log(U2 + eps)+eps)
            x = x + g1 - g2

        soft = torch.sigmoid(x / gumbel_temp)
        hard = ((soft >= 0.5).float() - soft).detach() + soft
        assert not torch.any(torch.isnan(hard))
        return hard

## Mask convs
class Squeeze(nn.Module):
    """
    Squeeze module to predict masks
    """

    def __init__(self, channels, stride=1):
        super(Squeeze, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1, bias=True)
        self.conv = nn.Conv2d(channels, 1, stride=stride,
                              kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, 1, 1, 1)
        z = self.conv(x)
        return z + y.expand_as(z)

def avg_pool_func(x):
    return torch.mean(x, dim=1, keepdim=True)


def max_pool_func(x):
    x, _ = torch.max(x, dim=1, keepdim=True)
    return x


## Mask convs
class SpatialAttention(nn.Module):
    """
    Squeeze module to predict masks
    """

    def __init__(self, stride=1):
        super(SpatialAttention, self).__init__()
        kernel_size = 7
        padding = 3
        self.conv = nn.Conv2d(2, 1, kernel_size, stride, padding, bias=False)

    def forward(self, x):
        x = torch.cat([avg_pool_func(x), max_pool_func(x)], dim=1)
        x = self.conv(x)
        return x

class ExpandMask(nn.Module):
    def __init__(self, stride, padding=1):
        super(ExpandMask, self).__init__()
        self.stride=stride
        self.padding = padding


    def forward(self, x):
        assert x.shape[1] == 1

        if self.stride > 1:
            self.pad_kernel = torch.zeros( (1,1,self.stride, self.stride), device=x.device)
            self.pad_kernel[0,0,0,0] = 1
        self.dilate_kernel = torch.ones((1,1,1+2*self.padding,1+2*self.padding), device=x.device)

        x = x.float()
        if self.stride > 1:
            x = F.conv_transpose2d(x, self.pad_kernel, stride=self.stride, groups=x.size(1))
        x = F.conv2d(x, self.dilate_kernel, padding=self.padding, stride=1)
        return x > 0.5
