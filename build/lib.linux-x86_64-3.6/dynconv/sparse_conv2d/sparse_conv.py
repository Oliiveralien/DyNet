import math

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
import sparse_conv2d_cuda

# 在使用该函数之前可以用torch.nn.utils.fusion.fuse_conv_bn_eval将conv和bn进行融合
# 1. 依据通道和空间位置的mask进行img2col以及kernel2col
# 2. 利用addmm和mm进行矩阵乘法
# 3. 利用两种mask将输出的特征图恢复到原来的位置 (需要加以区分,多个连续的sparse conv，
#                                             第一个的输出通道不需要恢复，最后一个的输出通道需要恢复)
class SparseConv2dFunction(Function):
    @staticmethod
    def forward(ctx,features,prev_c_mask,c_mask,s_mask,weight,bias,
                padding=0,stride=1,last_conv=False,activation=None):
        """
        prev_c_mask: 用来挑选权值
        c_mask: 用来挑选权值
        s_mask: 指导img2col以及将矩阵恢复成Tensor
        """
        assert c_mask.dim() == 4 and c_mask.size(2) == 1 and c_mask.size(3) == 1
        assert s_mask.dim() == 4 and s_mask.size(1) == 1
        assert features.dim() == 4 and features.size(0) == 1
        if prev_c_mask is not None:
            assert features.size(1) == prev_c_mask.size(1)


        pad_h, pad_w = _pair(padding)
        stride_h, stride_w = _pair(stride)
        if not features.is_cuda:
            raise NotImplementedError


        out_channel, in_channel, kernel_h, kernel_w = weight.size()
        batch_size = features.size(0)

        out_h = int(math.floor(
                (features.size(2) - kernel_h + 2 *pad_h)/stride_h + 1));
        out_w = int(math.floor(
                (features.size(3) - kernel_w + 2 *pad_w)/stride_w + 1));

        if prev_c_mask is not None:
            # 对输入通道进行筛选
            weight = torch.masked_select(weight,prev_mask).view(out_channel,-1,kernel_h,kernel_w)
        # 对输出通道进行挑选, 同时完成 kernel2col [cout, cin*k*k]
        weight = torch.masked_select(weight,c_mask.view(-1,1,1,1)).view(-1, weight.size(1)*kernel_w*kernel_h)
        if bias is not None:
            bias = torch.masked_select(bias,c_mask.squeeze())
        out_channel_new, in_channel = weight.size(0), weight.size(1)

        c_mask_idx = None
        if not last_conv:
            output = features.new_zeros(batch_size,out_channel_new,out_h,out_w)
        else:
            c_mask_idx = torch.nonzero(c_mask.squeeze()).squeeze()
            output = features.new_zeros(batch_size,out_channel,out_h,out_w);
        s_mask_idx = torch.nonzero(s_mask.squeeze())
        # 这里只负责实现两个都剪的
        if(s_mask.numel() > 0):
            s_mask_h_idx = s_mask_idx[:, 0].contiguous()
            s_mask_w_idx = s_mask_idx[:, 1].contiguous()
            # [cin*k^2, oH*oW]
            data_col = features.new_zeros(weight.size(1),s_mask_idx.size(0))
            # img2col
            sparse_conv2d_cuda.masked_im2col_forward(features, s_mask_h_idx,
                                                     s_mask_w_idx, kernel_h,
                                                     kernel_w,stride_h,
                                                     stride_w,pad_h,pad_w,
                                                     data_col)
            # addmm
            if bias is None:
                masked_out = torch.mm(weight,data_col);
            else:
                masked_out = torch.addmm(bias,weight,data_col);
            # 这里只支持逐元素的激活函数
            if activation is not None:
                masked_out = activation(masked_out)
            # reshape
            if last_conv:
                sparse_conv2d_cuda.masked_col2im_forward1(masked_output, s_mask_h_idx,
                                                         s_mask_w_idx,c_mask_idx, out_h,
                                                         out_w, out_channel_new, output)
                pass    # 最后一层卷积需要负责将通道恢复
            else:
                sparse_conv2d_cuda.masked_col2im_forward(masked_output, s_mask_h_idx,
                                                         s_mask_w_idx, out_h, out_w,
                                                         out_channel_new, output)
                pass    # 不用恢复通道
            # 

        return output


    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return (None, ) * 5


sparse_conv2d_cuda = SparseConv2dFunction.apply


class SparseConv2d(nn.Conv2d):
    """A MaskedConv2d which inherits the official Conv2d.

    The masked forward doesn't implement the backward function and only
    supports the stride parameter to be 1 currently.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 last_conv=False,
                 activation=None):
        super(SparseConv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias)

        self.activation = activation
        self.last_conv = last_conv

    def forward(self, x, mask_dict=None):

        if mask_dict is None:  # fallback to the normal Conv2d
            return super(MaskedConv2d, self).forward(input)

        return sparse_conv2d_cuda(x, mask_dict['prev_c_mask'],
                                 mask_dict['c_mask'], mask_dict['s_mask'], 
                                 self.weight, self.bias, self.padding,
                                 self.stride, self.last_conv, self.activation)