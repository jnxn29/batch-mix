'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
import time
from models import *
from utils import progress_bar, get_network

def mixt(x, lam, num_merge=2):
    batch_size = x.size(0)
    merge_size = torch.div(batch_size, num_merge, rounding_mode='trunc')#


    chunks = torch.split(x, split_size_or_sections=merge_size, dim=0)


    result_x = chunks[0]


    for i in range(1, num_merge):
        result_x = lam * result_x + (1 - lam) * chunks[i]

    return result_x

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-net', type=str, required=True, help='net type')
parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('-merge', type=int, default=1, help='degree of merging samples in mixt')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

net = get_network(args)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True



loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_loss = 0
    start = time.time()  # 开始时间
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        labels = nn.functional.one_hot(labels, 10)
        lam = np.random.beta(1, 1)
        #lam = torch.tensor(lam).to(device)
        labels = mixt(labels, lam, args.merge)
        optimizer.zero_grad()
        # print('lam:',lam)
        # print('merge:',args.merge)
        # print('inputs:',inputs.shape)
        # print('targets:',labels.shape)
        merge = torch.tensor(args.merge).clone().detach().to(device)
        outputs = net(inputs,lam,2)
        loss =loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()



    finish = time.time()
    # 参数收集部分
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))  # 打印训练时间
    epoch_time = finish - start
    lr = optimizer.param_groups[0]['lr']
    train_loss = train_loss / len(trainloader.dataset)

    return {'epoch': epoch, 'train loss': train_loss, 'lr': lr, 'time': epoch_time}



def test(epoch):
    global best_acc
    start = time.time()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():#不计算梯度
        for batch_idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = loss_function(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    finish = time.time()  # 结束时间
    test_loss = test_loss / len(testloader.dataset),
    test_acc = correct.float() / len(testloader.dataset),

    print('test time consumed: {:.2f}s'.format(finish - start))


    return {'test loss': test_loss[0], 'test acc': test_acc[0].item()}


timestamp = time.strftime("%Y%m%d_%H%M%S")

if args.gpu:
    filename = f"{args.merge}_{args.net}_lr{args.lr}_{timestamp}_3080cifar10.csv"
else:
    filename = f"{args.merge}_{args.net}_lr{args.lr}_{timestamp}_cpucifar10.csv"
with open(filename, mode='w', newline='') as csvfile:
    fieldnames = ['epoch', 'train loss', 'lr', 'test loss', 'test acc', 'time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

for epoch in range(1, 251):
    text1 = train(epoch)
    text2 = test(epoch)
    text1.update(text2)
    fieldnames = ['epoch', 'train loss', 'lr', 'test loss', 'test acc', 'time']
    reordered_dict = {key: text1[key] for key in fieldnames}
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow(text1)

        print(text1)

    scheduler.step()
