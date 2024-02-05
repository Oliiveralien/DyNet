""" 
ResNet for 32 by 32 images (CIFAR)
"""

import math

from models_test.resnet_util import *
from utils.logger import logger
from torch.nn.utils import fuse_conv_bn_eval

########################################
# Original ResNet                      #
########################################

class ResNet_32x32(nn.Module):
    def __init__(self, layers, num_classes=10, pretrained=False, sparse=False, type="spatial", use_ca=True):
        super(ResNet_32x32, self).__init__()

        if pretrained is not False:
            raise NotImplementedError('No pretrained models for 32x32 implemented')

        assert len(layers) == 3
        block = BasicBlock
        self.sparse = sparse
        self.type = None
        if self.sparse and type in ['mix', 'spatial', 'channel']:
            self.type = type
        self.use_ca = use_ca

        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
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
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, sparse=self.sparse, type=self.type, use_ca=self.use_ca))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, sparse=self.sparse, type=self.type, use_ca=self.use_ca))

        return nn.Sequential(*layers)



    def show_reduction_ratio(self):
        index = 1;
        for layer in self.layer1:
            str = "block_{}".format(index)
            index+=1
            logger.info(layer.show_reduction_ratio(str))

        for layer in self.layer2:
            str = "block_{}".format(index)
            index+=1
            logger.info(layer.show_reduction_ratio(str))

        for layer in self.layer3:
            str = "block_{}".format(index)
            index+=1
            logger.info(layer.show_reduction_ratio(str))
    
    def fuse_module(self):
        self.conv1 = fuse_conv_bn_eval(self.conv1,self.bn1)
        self.bn1 = None

    def forward(self, x, meta=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x, meta = self.layer1((x, meta))
        x, meta = self.layer2((x, meta))
        x, meta = self.layer3((x, meta))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, meta


def resnet8(sparse=False, type='sparsity', use_ca=True, **kwargs):
    return ResNet_32x32([1, 1, 1], sparse=sparse, type=type, use_ca=use_ca, **kwargs)


def resnet14(sparse=False, type='sparsity', use_ca=True, **kwargs):
    return ResNet_32x32([2, 2, 2], sparse=sparse, type=type, use_ca=use_ca, **kwargs)


def resnet20(sparse=False, type='sparsity', use_ca=True, **kwargs):
    return ResNet_32x32([3, 3, 3], sparse=sparse, type=type, use_ca=use_ca, **kwargs)


def resnet26(sparse=False, type='sparsity', use_ca=True, **kwargs):
    return ResNet_32x32([4, 4, 4], sparse=sparse, type=type, use_ca=use_ca, **kwargs)


def resnet32(sparse=False, type='sparsity', use_ca=True, **kwargs):
    return ResNet_32x32([5, 5, 5], sparse=sparse, type=type, use_ca=use_ca, **kwargs)

def resnet56(sparse=False, type='sparsity', use_ca=True, **kwargs):
    return ResNet_32x32([9, 9, 9], sparse=sparse, type=type, use_ca=use_ca, **kwargs)
