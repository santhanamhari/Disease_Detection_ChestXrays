import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
import time
from torch.nn.parameter import Parameter
import numpy as np
from torch import Tensor as Tensor
import torch.nn.functional as F


def gen_conv_mask(channel1, channel2, k):
    return np.ones([channel1, channel2, k, k])


class _Conv2d(nn.Module):
    '''
        A customized convoluion module to support growth and pruning
    '''
    def __init__(self, in_chann, out_chann, ksize=3, padding=1, width_ratio=1.1,
            init_fill=1.0, stride=1, bias=True):
        '''
            Args:
                in_chann: number of input channels
                out_chann: number of output channels
                ksize: kernel size
                padding: padding for the input image before conv
                width_ratio: max number of filters for growth.  For example, if width_ratio == 1.1,
                    then there are 10% growth space.
                init_fill: initial filling rate in the convolution kernel
                stride: conv stride
                bias: whether to use bias.  Note: bias has no effect if conv is followed by a BN
        '''
        super(_Conv2d, self).__init__()
        self.out_chann_max = int(width_ratio * out_chann)
        self.in_chann_max = int(width_ratio * in_chann)
        if in_chann == 3:
            self.in_chann_max = 3

        self.weight = Parameter(torch.Tensor(self.out_chann_max, self.in_chann_max,
            ksize, ksize))
        self.use_bias = bias

        if self.use_bias:
            self.bias = Parameter(torch.Tensor(self.out_chann_max))
            self.bias.data.zero_()
        else:
            self.bias = None

        self.in_chann, self.out_chann = in_chann, out_chann
        self.curr_in_chann, self.curr_out_chann = in_chann, out_chann

        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.init_fill = init_fill
        self.use_mask = True

        stdv = math.sqrt(2. / (out_chann * 3. * 3.))

        self.weight.data.normal_(0, stdv)

        mask_data = gen_conv_mask(out_chann, in_chann, ksize)
        self.mask = Parameter(data=Tensor(mask_data), requires_grad=False)

        self.weight.data = self.weight.data * self.mask.data
        self.grad = np.zeros(self.weight.data.size())

    def forward(self, _input):
        '''
            foward pass
        '''
        return F.conv2d(_input, (self.weight * self.mask), self.bias, stride=self.stride, padding=self.padding)

    def show_sparsity(self):
        '''
            print out the sparsity ratio
        '''
        w_np = self.weight.cpu().data.numpy()
        m_np = self.mask.cpu().data.numpy()
        #print (m_np)
        print ('weight fill rate: {}'.format(np.count_nonzero(w_np) / float(w_np.size)))
        print ('mask fill rate: {}'.format(np.count_nonzero(m_np) / float(m_np.size)))


def conv3x3(in_planes, out_planes, stride=1, init_fill=1.0):
    '''
        3x3 convolution with padding
    '''
    # replace the default nn.Conv2d with _Conv2d
    return _Conv2d(in_planes, out_planes, ksize=3, stride=stride,
                     padding=1, width_ratio=1.0)


class BasicBlock(nn.Module):
    '''
        Basic building block for resnet 18 and resnet 34
    '''
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, init_fill=1.0):
        super(BasicBlock, self).__init__()
        '''
            Basic building block initialization
            Args:
                inplanes: number of input channels
                planes: number of output channels
        '''
        self.conv1 = conv3x3(inplanes, planes, stride, init_fill=init_fill)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, init_fill=init_fill)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    '''
        Bottleneck building block for resnet 18 and resnet 34
    '''
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, init_fill=1.0):
        '''
            Bottleneck building block initialization
            Args:
                inplanes: number of input channels
                planes: number of output channels
        '''
        super(Bottleneck, self).__init__()
        self.conv1 = _Conv2d(inplanes, planes, ksize=1, padding=0, width_ratio=1.0,
            init_fill=init_fill)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _Conv2d(planes, planes, ksize=3, stride=stride,
                               padding=1, width_ratio=1.0, init_fill=init_fill)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = _Conv2d(planes, planes * self.expansion, ksize=1, padding=0, width_ratio=1.0,
            init_fill=init_fill)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    '''
        The resnet implementation is modified from the official pytorch implementation
    '''
    def __init__(self, args, block, layers, num_classes=1000, init_fill_conv=1.0,
            scaling=0.8):
        self.inplanes = int(64 * scaling)
        super(ResNet, self).__init__()

        self.scaling = scaling
        self.init_fill_conv = init_fill_conv
        print('conv layer init fill rate {}'.format(self.init_fill_conv))

        # replace the default nn.Conv2d with _Conv2d
        self.conv1 = _Conv2d(3, int(64*scaling), ksize=7, stride=2, padding=3,
            width_ratio=1.0, init_fill=self.init_fill_conv)
        self.bn1 = nn.BatchNorm2d(int(64*scaling))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(64*scaling), layers[0])
        self.layer2 = self._make_layer(block, int(128*scaling), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*scaling), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(512*scaling), layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(int(int(512*scaling)*block.expansion), num_classes)

        for m in self.modules():
            # using kaiming initialization for resnet
            if isinstance(m, _Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:

            # replace the default nn.Conv2d with _Conv2d
            downsample = nn.Sequential(
                _Conv2d(self.inplanes, planes * block.expansion,
                          ksize=1, stride=stride, width_ratio=1.0, padding=0,
                        init_fill=self.init_fill_conv),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
            init_fill=self.init_fill_conv))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, init_fill=self.init_fill_conv))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def set_use_msk(self, use_mask):
        '''
            use the mask in the foward pass or not
            mask is switched on for training, off for gradient accumulation
        '''
        for layer in self.modules():
            if isinstance(layer, _Conv2d):
                layer.set_use_msk(use_mask)


def resnet50(args, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print('arguments {}'.format(kwargs))
    model = ResNet(args, Bottleneck, [3, 4, 6, 3], **kwargs)
    print('model initilization complete!')
    return model


if __name__ == "__main__":
    model = resnet50(None)
    with torch.no_grad():
        x = torch.randn(1,3,224,224)
        y = model(x)
