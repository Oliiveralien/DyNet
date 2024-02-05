import torch
import torch.nn as nn

# from utils import logger
from .sMaskUnit import Gumbel


class ChannelMask():
    '''
    Class that holds the mask properties

    hard: the hard/binary mask (1 or 0), 4-dim tensor
    soft (optional): the float mask, same shape as hard
    active_positions: the amount of positions where hard == 1
    total_positions: the total amount of positions
                        (typically batch_size * output_width * output_height)
    '''

    def __init__(self, hard, soft=None, hard_prev=None):
        assert hard.dim() == 4
        assert hard.shape[2] == 1 and hard.shape[3] == 1
        assert soft is None or soft.shape == hard.shape

        self.hard = hard
        # [N Ci 1 1] & [N Co 1 1] -> [N Ci*Co 1 1]
        if isinstance(hard_prev,ChannelMask):
            # print("------------------------------------------------------------")
            # print(hard.sum(dim=1).squeeze(), hard_prev['channel'].hard.detach().sum(dim=1).squeeze(),hard.numel(),hard_prev['channel'].hard.detach().numel())
            self.channel_inte_state = (hard.sum(dim=1) * hard_prev.hard.sum(dim=1)).view(-1)
            self.active_channels = torch.sum(self.channel_inte_state)  # this must be kept backpropagatable!
            #                     N * Co               Ci
            self.total_channels = hard.numel() * hard_prev.hard.shape[1]
        else:
            # Sci = (channel_mask.total_channels - channel_mask.active_channels) * \
            #       spatial_mask.total_positions * \
            #       spatial_mask.kernel_size
            self.channel_inte_state = hard.sum(dim=(1,2,3)) * hard_prev
            self.active_channels = torch.sum(hard) * hard_prev # this must be kept backpropagatable!
            self.total_channels = hard.numel() * hard_prev

        self.soft = soft

        self.flops_per_channel = 0

    def size(self):
        return self.hard.shape

    def __repr__(self):
        return 'Mask with {}/{} channels, and {} accumulated FLOPS per channel'.format(
            self.active_channels,self.total_channels,self.flops_per_channel)


class ChannelMaskUnit(nn.Module):
    '''
    Generates the mask(channel/spatial/mix) and applies the gumbel softmax trick
    '''

    def __init__(self, in_channels, out_channels,reduction = 0):
        super(ChannelMaskUnit, self).__init__()
        self.channel_maskconv = ChannelAttention(in_channels, out_channels,ratio = reduction)
        self.gumbel = Gumbel()

    def forward(self, x, prev_mask, meta):
        channel_soft = self.channel_maskconv(x)
        channel_hard = self.gumbel(channel_soft, meta['gumbel_temp'], meta['gumbel_noise'])
        channel_mask = ChannelMask(channel_hard, channel_soft,hard_prev=prev_mask)
        # mask = {'channel': channel_mask}
        return channel_mask


## Mask convs
class ChannelAttention(nn.Module):
    """
    Squeeze module to predict masks
    """

    def __init__(self, in_channel, out_channel, ratio=0):
        super(ChannelAttention, self).__init__()
        self.ratio = ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        if in_channel <= ratio:
            print(in_channel)
            raise ValueError("Please check squeeze ratio!")
        if self.ratio <= 0:
            self.conv1 = nn.Conv2d(in_channel, out_channel, 1)
        else:
            self.conv1 = nn.Conv2d(in_channel, in_channel // ratio, 1)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(in_channel // ratio, out_channel, 1)

    def forward(self, x):
        if self.ratio <= 0:            
            x1 = self.conv1(self.avg_pool(x))
            x2 = self.conv1(self.max_pool(x))
        else:
            x1 = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
            x2 = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        x = x1 + x2
        return x


# import torch
# import torch.nn as nn
#
# # from utils import logger
# from .sMaskUnit import Gumbel
#
#
# class ChannelMask():
#     '''
#     Class that holds the mask properties
#
#     hard: the hard/binary mask (1 or 0), 4-dim tensor
#     soft (optional): the float mask, same shape as hard
#     active_positions: the amount of positions where hard == 1
#     total_positions: the total amount of positions
#                         (typically batch_size * output_width * output_height)
#     '''
#
#     def __init__(self, hard, soft=None, hard_prev=None):
#         assert hard.dim() == 4
#         assert hard.shape[2] == 1 and hard.shape[3] == 1
#         assert soft is None or soft.shape == hard.shape
#
#         self.hard = hard
#         # [N Ci 1 1] & [N Co 1 1] -> [N Ci*Co 1 1]
#         if isinstance(hard_prev,ChannelMask):
#             # print("------------------------------------------------------------")
#             # print(hard.sum(dim=1).squeeze(), hard_prev['channel'].hard.detach().sum(dim=1).squeeze(),hard.numel(),hard_prev['channel'].hard.detach().numel())
#             self.channel_inte_state = (hard.sum(dim=1) * hard_prev.hard.sum(dim=1)).view(-1)
#             self.active_channels = torch.sum(self.channel_inte_state)  # this must be kept backpropagatable!
#             #                     N * Co               Ci
#             self.total_channels = hard.numel() * hard_prev.hard.shape[1]
#         else:
#             # Sci = (channel_mask.total_channels - channel_mask.active_channels) * \
#             #       spatial_mask.total_positions * \
#             #       spatial_mask.kernel_size
#             self.channel_inte_state = hard.sum(dim=(1,2,3)) * hard_prev
#             self.active_channels = torch.sum(hard) * hard_prev # this must be kept backpropagatable!
#             self.total_channels = hard.numel() * hard_prev
#
#         self.soft = soft
#
#         self.flops_per_channel = 0
#
#     def size(self):
#         return self.hard.shape
#
#     def __repr__(self):
#         return 'Mask with {}/{} channels, and {} accumulated FLOPS per channel'.format(
#             self.active_channels,self.total_channels,self.flops_per_channel)
#
#
# class ChannelMaskUnit(nn.Module):
#     '''
#     Generates the mask(channel/spatial/mix) and applies the gumbel softmax trick
#     '''
#
#     def __init__(self, in_channels, out_channels,atten_head=False):
#         super(ChannelMaskUnit, self).__init__()
#         self.channel_maskconv = ChannelAttention(in_channels, out_channels)
#         self.atten_conv = None
#         if atten_head == True:
#             self.atten_conv = ChannelAttention(in_channels, out_channels)
#         self.gumbel = Gumbel()
#
#     def forward(self, x, prev_mask, meta):
#         channel_soft = self.channel_maskconv(x)
#         channel_hard = self.gumbel(channel_soft, meta['gumbel_temp'], meta['gumbel_noise'])
#         channel_mask = ChannelMask(channel_hard, channel_soft,hard_prev=prev_mask)
#         if self.atten_conv is not None:
#             atten = torch.sigmoid(self.atten_conv(x))
#             # mask = {'channel': channel_mask}
#             return channel_mask,atten
#         else:
#             return channel_mask
#
#
# ## Mask convs
# class ChannelAttention(nn.Module):
#     """
#     Squeeze module to predict masks
#     """
#
#     def __init__(self, in_channel, out_channel, ratio=8):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         if in_channel <= ratio:
#             print(in_channel)
#             raise ValueError("Please check squeeze ratio!")
#
#         self.conv1 = nn.Conv2d(in_channel, out_channel, 1)
#         # self.conv1 = nn.Conv2d(in_channel, in_channel // ratio, 1)
#         # self.relu = nn.ReLU(inplace=True)
#         # self.conv2 = nn.Conv2d(in_channel // ratio, out_channel, 1)
#
#     def forward(self, x):
#         # x1 = self.conv2(self.conv1(self.avg_pool(x)))
#         # x2 = self.conv2(self.conv1(self.max_pool(x)))
#         x1 = self.conv1(self.avg_pool(x))
#         x2 = self.conv1(self.max_pool(x))
#         x = x1 + x2
#         return x
