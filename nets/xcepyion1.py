import torch.nn as nn
from torch.hub import load_state_dict_from_url
from .utils import *
import torch
import torch.nn.functional as F

import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 这是一个用于实现 ResNet 架构的 PyTorch 模块中的 BasicBlock 类。ResNet 是一种深度卷积神经网络结构，具有许多卷积层和残差连接，
# 使得在训练深层网络时可以避免梯度消失问题。BasicBlock 是 ResNet 中的基本块，由两个 3x3 的卷积层、BatchNormalization 和 ReLU 激活函数组成，
# 同时包含跳跃连接。该类的输入为大小为 inplanes 的张量，输出为大小为 planes 的张量，stride 参数控制卷积步幅，downsample 参数控制是否执行下采样。
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


#
# 这段代码定义了 ResNet 的 Bottleneck 模块，用于构建 ResNet-50、ResNet-101 等较深的网络。
# Bottleneck 模块主要包含了 1x1、3x3 和 1x1 三个卷积层，其中 1x1 卷积用于下降通道数，3x3 卷积进行特征提取，1x1 卷积上升通道数。
# 这里的下降和上升通道数指的是卷积核数量，通过这种方式可以增加模型的宽度。该模块还包含了 BatchNorm 和 ReLU 激活函数，以及可选的下采样和步长参数。
# 其中，norm_layer 参数表示规范化层的类型，若不指定则默认为 nn.BatchNorm2d。 forward() 方法实现了前向计算，其中 identity 保存了输入的特征图。
# 通过卷积层、规范化层、激活函数的组合对特征图进行处理，并最终返回特征图加上 identity 的结果。
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,alpha=1.667):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # 利用1x1卷积下降通道数
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.planes=planes
        # 利用3x3卷积进行特征提取
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # 利用1x1卷积上升通道数
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.inpanes=inplanes
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
#xceptiion
        w = alpha * planes

        self.out_channel = int(w * 0.167) + int(w * 0.333) + int(w * 0.5)
        self.conv2d_bn = nn.Conv2d(inplanes, self.out_channel, 1)
        self.conv3x3 = nn.Conv2d(inplanes, int(w * 0.167), 3,1,1)
        self.conv5x5 = nn.Conv2d(int(w * 0.167), int(w * 0.333), 3,1,1)
        self.conv7x7 =nn.Conv2d(int(w * 0.333), int(w * 0.5), 3,1,1)

        self.conv3x3_2 = nn.Conv2d(inplanes, int(w * 0.167), 3,1,1)
        self.conv5x5_2 = nn.Conv2d(int(w * 0.167), int(w * 0.333), 3,1,1)
        self.conv7x7_2 = nn.Conv2d(int(w * 0.333), int(w * 0.5), 3,1,1)
        self.bn_1 = torch.nn.BatchNorm2d(self.out_channel)
        self.bn_1_2 = torch.nn.BatchNorm2d(self.out_channel)
        self.con=nn.Conv2d(self.out_channel, width, 3,1,1)
        self.relu = torch.nn.ReLU()
        self.bn_2 = torch.nn.BatchNorm2d(self.out_channel)

    def forward(self, x):
        identity = x
        print("planes",self.planes)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        print("out0", out.shape)
        print(self.inpanes)
        shortcut = self.conv2d_bn(out)
        conv3x3 = self.conv3x3(out)
        conv5x5 = self.conv5x5(conv3x3)
        conv7x7 = self.conv7x7(conv5x5)
        print("conv3x3",conv3x3.shape,"conv5x5",conv5x5.shape,"conv7x7",conv7x7.shape)
        out = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        out = self.bn_1(out)

        conv3x3_2 = self.conv3x3_2(x)
        conv5x5_2 = self.conv5x5_2(conv3x3_2)
        conv7x7_2 = self.conv7x7_2(conv5x5_2)
        out_2 = torch.cat([conv3x3_2, conv5x5_2, conv7x7_2], dim=1)
        print(out_2.shape,"out2")
        out_2 = self.bn_1_2(out_2)

        out_f = shortcut + out + out_2
        out_f = self.relu(out_f)
        out_f = self.bn_2(out_f)
        print("outf",out_f.shape)
        out=self.con(out_f)
        print("out1", out.shape)
        out = self.conv3(out)
        print("out2", out.shape)
        out = self.bn3(out)
        print("out3",out.shape)

        return out


# 该代码实现了一个ResNet网络，其中包含多个Bottleneck模块，使用的是经典的ResNet结构。其中，block参数指定了使用的是什么类型的Bottleneck块，
# layers参数指定了每个stage中包含的Bottleneck块数量，num_classes参数指定了网络的输出类别数。
# 在初始化过程中，定义了多个模块，包括卷积层、归一化层、池化层等，其中特别注意的是，在Bottleneck模块中，使用了1x1卷积降低通道数，3x3卷积进行特征提取，
# 1x1卷积上升通道数的操作，使得网络在保证特征表达能力的同时减少了参数数量。此外，在初始化过程中还对卷积层和归一化层的权重进行了初始化。
# 在前向传播过程中，首先对输入进行卷积、归一化、激活操作，并对结果进行池化操作。之后，通过调用_make_layer函数，
# 多次堆叠Bottleneck模块形成不同的stage，最终输出网络的特征图。注意，在前向传播过程中，并没有进行全局池化和全连接操作
# 这是因为该代码的实现是用于特征提取，输出的特征图可以用于各种任务，如分类、检测、分割等。
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        # -----------------------------------------------------------#
        #   假设输入图像为600,600,3
        #   当我们使用resnet50的时候
        # -----------------------------------------------------------#
        self.inplanes = 64  # 通道数
        super(ResNet, self).__init__()
        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 38,38,1024 -> 19,19,2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
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
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        x = self.conv1(x)
        x = self.bn1(x)
        feat1 = self.relu(x)

        x = self.maxpool(feat1)
        print(x.shape)
        feat2 = self.layer1(x)

        feat3 = self.layer2(feat2)
        feat4 = self.layer3(feat3)
        feat5 = self.layer4(feat4)
        print(feat1.shape,feat2.shape,feat3.shape,feat4.shape,feat5.shape)
        return [feat1, feat2, feat3, feat4, feat5]


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth', model_dir='model_data'),
            strict=False)

    del model.avgpool
    del model.fc
    return model


def Conv2dSame(in_channels, out_channels, kernel_size, use_bias=True, padding_layer=torch.nn.ReflectionPad2d):
    ka = kernel_size // 2
    kb = ka - 1 if kernel_size % 2 == 0 else ka
    return [
        padding_layer((ka, kb, ka, kb)),
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=use_bias)
    ]



#xception

import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 这是一个用于实现 ResNet 架构的 PyTorch 模块中的 BasicBlock 类。ResNet 是一种深度卷积神经网络结构，具有许多卷积层和残差连接，
# 使得在训练深层网络时可以避免梯度消失问题。BasicBlock 是 ResNet 中的基本块，由两个 3x3 的卷积层、BatchNormalization 和 ReLU 激活函数组成，
# 同时包含跳跃连接。该类的输入为大小为 inplanes 的张量，输出为大小为 planes 的张量，stride 参数控制卷积步幅，downsample 参数控制是否执行下采样。
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


#
# 这段代码定义了 ResNet 的 Bottleneck 模块，用于构建 ResNet-50、ResNet-101 等较深的网络。
# Bottleneck 模块主要包含了 1x1、3x3 和 1x1 三个卷积层，其中 1x1 卷积用于下降通道数，3x3 卷积进行特征提取，1x1 卷积上升通道数。
# 这里的下降和上升通道数指的是卷积核数量，通过这种方式可以增加模型的宽度。该模块还包含了 BatchNorm 和 ReLU 激活函数，以及可选的下采样和步长参数。
# 其中，norm_layer 参数表示规范化层的类型，若不指定则默认为 nn.BatchNorm2d。 forward() 方法实现了前向计算，其中 identity 保存了输入的特征图。
# 通过卷积层、规范化层、激活函数的组合对特征图进行处理，并最终返回特征图加上 identity 的结果。
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # 利用1x1卷积下降通道数
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # 利用3x3卷积进行特征提取
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv11 = nn.Conv2d(width, width, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=width)
        self.bn11 = nn.BatchNorm2d(width)  # 输入为上一层输出的通道数
        # pointwise
        self.conv22 = nn.Conv2d(width, width, (1, 1))  # Stride of the convolution. Default: 1
        self.bn22 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU()







        # 利用1x1卷积上升通道数
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv11(out)
        out = self.bn11(out)
        out = self.relu(out)
        out = self.conv22(out)
        out = self.bn22(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 该代码实现了一个ResNet网络，其中包含多个Bottleneck模块，使用的是经典的ResNet结构。其中，block参数指定了使用的是什么类型的Bottleneck块，
# layers参数指定了每个stage中包含的Bottleneck块数量，num_classes参数指定了网络的输出类别数。
# 在初始化过程中，定义了多个模块，包括卷积层、归一化层、池化层等，其中特别注意的是，在Bottleneck模块中，使用了1x1卷积降低通道数，3x3卷积进行特征提取，
# 1x1卷积上升通道数的操作，使得网络在保证特征表达能力的同时减少了参数数量。此外，在初始化过程中还对卷积层和归一化层的权重进行了初始化。
# 在前向传播过程中，首先对输入进行卷积、归一化、激活操作，并对结果进行池化操作。之后，通过调用_make_layer函数，
# 多次堆叠Bottleneck模块形成不同的stage，最终输出网络的特征图。注意，在前向传播过程中，并没有进行全局池化和全连接操作
# 这是因为该代码的实现是用于特征提取，输出的特征图可以用于各种任务，如分类、检测、分割等。
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        # -----------------------------------------------------------#
        #   假设输入图像为600,600,3
        #   当我们使用resnet50的时候
        # -----------------------------------------------------------#
        self.inplanes = 64  # 通道数
        super(ResNet, self).__init__()
        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 38,38,1024 -> 19,19,2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
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
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        x = self.conv1(x)
        x = self.bn1(x)
        feat1 = self.relu(x)

        x = self.maxpool(feat1)
        feat2 = self.layer1(x)

        feat3 = self.layer2(feat2)
        feat4 = self.layer3(feat3)
        feat5 = self.layer4(feat4)
        return [feat1, feat2, feat3, feat4, feat5]


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth', model_dir='model_data'),
            strict=False)

    del model.avgpool
    del model.fc
    return model
class SelfAttention(Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels):
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels)]
        self.gamma = nn.Parameter(tensor([0.]))

    def _conv(self,n_in,n_out):
        return ConvLayer(n_in, n_out, ks=1, ndim=1, norm_type=NormType.Spectral, act_cls=None, bias=False)

    def forward(self, x):
        #Notation from the paper.
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()




