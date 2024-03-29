'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

def get_network(args):
    """ return given network
    """
    print(args.net)

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    # c2mixt
    elif args.net == 'c2mixt_resnet18':
        from models.c2mixt_resnet import c2mixt_resnet18
        net = c2mixt_resnet18()
    elif args.net == 'c2mixt_resnet34':
        from models.c2mixt_resnet import c2mixt_resnet34
        net = c2mixt_resnet34()
    elif args.net == 'c2mixt_resnet50':
        from models.c2mixt_resnet import c2mixt_resnet50
        net = c2mixt_resnet50()
    elif args.net == 'c2mixt_resnet101':
        from models.c2mixt_resnet import c2mixt_resnet101
        net = c2mixt_resnet101()
    elif args.net == 'c2mixt_resnet152':
        from models.c2mixt_resnet import c2mixt_resnet152
        net = c2mixt_resnet152()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()


    # c4mixt
    elif args.net == 'c4mixt_resnet18':
        from models.c4mixt_resnet import c4mixt_resnet18
        net = c4mixt_resnet18()
    elif args.net == 'c4mixt4_resnet34':
        from models.c4mixt_resnet import c4mixt_resnet34
        net = c4mixt_resnet34()
    elif args.net == 'c4mixt4_resnet50':
        from models.c4mixt_resnet import c4mixt_resnet50
        net = c4mixt_resnet50()
    elif args.net == 'c4mixt4_resnet101':
        from models.c4mixt_resnet import c4mixt_resnet101
        net = c4mixt_resnet101()
    elif args.net == 'c4mixt_resnet152':
        from models.c2mixt_resnet import c4mixt_resnet152
        net = c4mixt_resnet152()


        # c2c4mixt
    elif args.net == 'c2c4mixt_resnet18':
        from models.c2c4mixt_resnet import c2c4mixt_resnet18
        net = c2c4mixt_resnet18()
    elif args.net == 'c2c4mixt_resnet34':
        from models.c2c4mixt_resnet import c2c4mixt_resnet34
        net = c2c4mixt_resnet34()
    elif args.net == 'c2c4mixt_resnet50':
        from models.c2c4mixt_resnet import c2c4mixt_resnet50
        net = c2c4mixt_resnet50()
    elif args.net == 'c2c4mixt_resnet101':
        from models.c2c4mixt_resnet import c2c4mixt_resnet101
        net = c2c4mixt_resnet101()
    elif args.net == 'c2c4mixt_resnet152':
        from models.c2c4mixt_resnet import c2c4mixt_resnet152
        net = c2c4mixt_resnet152()
    elif args.net == 'c14mixt_InceptionV3':
        from models.c14mixt_inceptionv3 import c14mixt_inceptionv3
        net = c14mixt_inceptionv3()
    # #mixtvgg
    # elif args.net == 'c4mixt_vgg16':
    #     from models.c4mixt_vgg import c4mixt_vgg16_bn
    #     net = c4mixt_vgg16_bn()
    # elif args.net == 'c4mixt_vgg13':
    #     from models.c4mixt_vgg import c4mixt_vgg13_bn
    #     net = c4mixt_vgg13_bn()
    # elif args.net == 'c4mixt_vgg11':
    #     from models.c4mixt_vgg import c4mixt_vgg11_bn
    #     net = c4mixt_vgg11_bn()
    # elif args.net == 'c4mixt_vgg19':
    #     from models.c4mixt_vgg import c4mixt_vgg19_bn
    #     net = c4mixt_vgg19_bn()
    elif args.net == 'c3mixt_resnet18':
        from models.c3mixt_resnet import c3mixt_resnet18
        net = c3mixt_resnet18()
    elif args.net == 'c3mixt_resnet34':
        from models.c3mixt_resnet import c3mixt_resnet34
        net = c3mixt_resnet34()
    elif args.net == 'c3mixt_resnet50':
        from models.c3mixt_resnet import c3mixt_resnet50
        net = c3mixt_resnet50()
    elif args.net == 'c3mixt_resnet101':
        from models.c3mixt_resnet import c3mixt_resnet101
        net = c3mixt_resnet101()
    elif args.net == 'c3mixt_resnet152':
        from models.c3mixt_resnet import c3mixt_resnet152
        net = c3mixt_resnet152()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
