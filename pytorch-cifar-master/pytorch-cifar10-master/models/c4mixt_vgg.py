"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
def mixt(x, lam, num_merge=2):
    if lam==0:
        return x
    else:
        batch_size = x.size(0)
        merge_size = torch.div(batch_size, num_merge, rounding_mode='trunc')#

        # 使用 torch.chunk 将 x 切分为 num_merge 部分
        chunks = torch.split(x, split_size_or_sections=merge_size, dim=0)

        # 选择第一个 chunk
        result_x = chunks[0]

        # 对后续的 chunk 进行混合
        for i in range(1, num_merge):
            result_x = lam * result_x + (1 - lam) * chunks[i]

        return result_x
def mixt(x):
    # 自定义的 mixt 函数逻辑
    return x

class CustomMixtLayer(nn.Module):
    def __init__(self):
        super(CustomMixtLayer, self).__init__()

    def forward(self, x):
        # 在这里调用 custom_mixt 函数进行处理
        output = mixt(x)
        return output

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256,'i', 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class c4mixt_VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x, lam=0, num_merge=2):
        output = self.features(x)

        output = output.view(output.size()[0], -1)


        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        if l == 'i':
            layers += [CustomMixtLayer()]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


    return nn.Sequential(*layers)

def c4mixt_vgg11_bn():
    return c4mixt_VGG(make_layers(cfg['A'], batch_norm=True))

def c4mixt_vgg13_bn():
    return c4mixt_VGG(make_layers(cfg['B'], batch_norm=True))

def c4mixt_vgg16_bn():
    return c4mixt_VGG(make_layers(cfg['D'], batch_norm=True))

def c4mixt_vgg19_bn():
    return c4mixt_VGG(make_layers(cfg['E'], batch_norm=True))


