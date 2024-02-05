""" 
ResNet for 32 by 32 images (CIFAR)
"""

import math

from torch import nn
from .resnet_util import *
from utils.logger import logger


########################################
# Original ResNet                      #
########################################

class SparseResNet_32x32(nn.Module):
    def __init__(self, layers, num_classes=10, use_ca=True):
        super(SparseResNet_32x32, self).__init__()


        assert len(layers) == 3
        block = BasicBlock

        self.use_ca = use_ca

        self.inplanes = 16
        # Conv 和 BN 融合去掉BN, 融合后存在bias
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias = True)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) and m.affine == True:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # 同理也去掉 BN
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True)
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, use_ca=self.use_ca))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_ca=self.use_ca))

        return nn.Sequential(*layers)


    def forward(self, x, meta=None):
        # Conv 和 BN 融合去掉BN
        x = self.conv1(x)
        x = self.relu(x)

        x, meta = self.layer1((x, meta))
        x, meta = self.layer2((x, meta))
        x, meta = self.layer3((x, meta))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, meta


def sparse_resnet8(use_ca=True, **kwargs):
    return SparseResNet_32x32([1, 1, 1], use_ca=use_ca, **kwargs)


def sparse_resnet14(use_ca=True, **kwargs):
    return SparseResNet_32x32([2, 2, 2], use_ca=use_ca, **kwargs)


def sparse_resnet20(use_ca=True, **kwargs):
    return SparseResNet_32x32([3, 3, 3], use_ca=use_ca, **kwargs)


def sparse_resnet26(use_ca=True, **kwargs):
    return SparseResNet_32x32([4, 4, 4], use_ca=use_ca, **kwargs)


def sparse_resnet32(use_ca=True, **kwargs):
    return SparseResNet_32x32([5, 5, 5], use_ca=use_ca, **kwargs)

def sparse_resnet56(use_ca=True, **kwargs):
    return SparseResNet_32x32([9, 9, 9], use_ca=use_ca, **kwargs)
