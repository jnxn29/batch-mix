"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
# def mixt(x, lam, num_merge=2):
#     batch_size = x.size(0)
#     merge_size = torch.div(batch_size, num_merge, rounding_mode='trunc')
#     result_x = lam * x[:merge_size] + (1 - lam) * x[merge_size:]
#     return result_x

def mixt(x, lam, num_merge=2):
    batch_size = x.size(0)
    merge_size = torch.div(batch_size, num_merge, rounding_mode='trunc')
    result_x=x[:merge_size]
    for i in range(1, num_merge):
        result_x =lam*result_x+(1-lam)*x[merge_size*i:merge_size*(i+1)]
    return result_x

# def mixt(x, lam, num_chunks=4):
#     chunks = torch.split(x, split_size_or_sections=torch.div(x.size(0), num_chunks, rounding_mode='trunc'), dim=0)
#
#     # 同时计算 mix_i，其中 i 取值范围为 [0, num_chunks)
#     mix_i = [lam * chunks[2 * i] + (1 - lam) * chunks[2 * i + 1] for i in range(num_chunks // 2)]
#
#     # 使用异步操作确保 mix_i 的计算完成
#     torch.cuda.synchronize()
#
#     # 同时计算 result_x
#     result_x = lam * sum(mix_i)
#
#     return result_x

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class c4mixt_resnet(nn.Module):

    def __init__(self, block, num_block, num_classes=10):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x,lam=0,merge=1):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        if lam> 0:
            # #检验点
            # print("lam is ",lam)
            output = mixt(output, lam, merge)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

def c4mixt_resnet18():
    """ return a ResNet 18 object
    """
    return c4mixt_resnet(BasicBlock, [2, 2, 2, 2])

def c4mixt_resnet34():
    """ return a ResNet 34 object
    """
    return c4mixt_resnet(BasicBlock, [3, 4, 6, 3])

def c4mixt_resnet50():
    """ return a ResNet 50 object
    """
    return c4mixt_resnet(BottleNeck, [3, 4, 6, 3])

def c4mixt_resnet101():
    """ return a ResNet 101 object
    """
    return c4mixt_resnet(BottleNeck, [3, 4, 23, 3])

def c4mixt_resnet152():
    """ return a ResNet 152 objecta
    """
    return c4mixt_resnet(BottleNeck, [3, 8, 36, 3])



