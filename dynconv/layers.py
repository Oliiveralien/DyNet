# these wrappers register the FLOPS of each layer, which
# will be used in the sparsity criterion to restrict the 
# amount of executed conditoinal operations. 
# In the pose repository, these wrappers are also used to
# efficiently execute sparse layers.
import torch

## CONVOLUTIONS

def outshape(param):
    ih, iw = param['ih'], param['iw']
    kh, kw = param['kh'], param['kw']
    ph, pw = param['ph'], param['pw']
    sh, sw = param['sh'], param['sw']
    oh = int((ih - kh + 2 * ph) / sh) + 1
    ow = int((iw - kw + 2 * pw) / sw) + 1
    return oh, ow

# m1 = self.cMasker1(x, x.shape[1], meta)
# x = dynconv.conv3x3(self.conv1, x, None, m1)
# x = dynconv.bn_relu(self.bn1, self.relu, x, m1)
# x = dynconv.apply_mask(x, m1)
# m2 = self.cMasker1(x, m1, meta)
# x = dynconv.conv3x3(self.conv2, x, m1, m2)
# x = dynconv.bn_relu(self.bn2, None, x, m2)
# out = dynconv.apply_mask(x, m2)
# meta['masks'].append(m1)
# meta['masks'].append(m2)

def conv(conv_module, x, mask_prev ,mask):
    channel_mask = mask.get('channel', None)
    spatial_mask = mask.get('spatial', None)
    w = conv_module.weight.data
    # mix
    if channel_mask is not None and spatial_mask is not None:
        spatial_mask.flops_per_position += w.shape[0] * w.shape[1] * w.shape[2] * w.shape[3]
        spatial_mask.kernel_size = w.shape[2] * w.shape[3]
    # channel
    elif channel_mask is not None:
        oh, ow = outshape({'ih': x.shape[2], 'iw': x.shape[3],
                           'kh': w.shape[2], 'kw': w.shape[3],
                           'ph': conv_module.padding[0], 'pw': conv_module.padding[1],
                           'sh': conv_module.stride[0], 'sw': conv_module.stride[1]})
        # FLOPs = Ho * Wo * Ci * Co * Kw * Kh
        #                                    Ho   Wo      Kh            Kw
        channel_mask.flops_per_channel = oh * ow * w.shape[2] * w.shape[3]
    # spatial
    elif spatial_mask is not None:
        #                                  Co            Ci          Kh             Kw
        spatial_mask.flops_per_position += w.shape[0] * w.shape[1] * w.shape[2] * w.shape[3]

    if "channel" in mask.keys():
        mask["channel"] = channel_mask
    if "spatial" in mask.keys():
        mask["spatial"] = spatial_mask

    conv_module.__mask__ = mask
    conv_module.__prev_mask__ = mask_prev
    return conv_module(x)



# 使用CUDA编写的稀疏卷积时就不进行FLOPs的统计了
def conv1x1(conv_module, x, mask_prev, mask, fast=False):
    if not fast:
        return conv(conv_module, x,mask_prev, mask)
    else:
        raise NotImplementedError


def conv3x3_dw(conv_module, x, mask_prev, mask, fast=False):
    if not fast:
        return conv(conv_module, x,mask_prev, mask)
    else:
        raise NotImplementedError

def conv3x3(conv_module, x, mask_prev, mask, fast=False):
    if not fast:
        return conv(conv_module, x,mask_prev, mask)
    else:
        raise NotImplementedError


## BATCHNORM and RELU
def bn_relu(bn_module, relu_module, x, mask, fast=False):
    bn_module.__mask__ = mask
    if relu_module is not None:
        relu_module.__mask__ = mask
    if not fast:
        x = bn_module(x)
        x = relu_module(x) if relu_module is not None else x
        return x
    else:
        raise NotImplementedError
