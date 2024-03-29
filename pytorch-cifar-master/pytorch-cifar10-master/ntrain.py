
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

    # 使用 torch.chunk 将 x 切分为 num_merge 部分
    chunks = torch.split(x, split_size_or_sections=merge_size, dim=0)

    # 选择第一个 chunk
    result_x = chunks[0]

    # 对后续的 chunk 进行混合
    for i in range(1, num_merge):
        result_x = lam * result_x + (1 - lam) * chunks[i]

    return result_x

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-net', type=str, required=True, help='net type')#使用的网络
parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('-merge', type=int, default=1, help='degree of merging samples in mixt')  # 合并程度
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')#是否从断点处继续训练
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



loss_function = nn.CrossEntropyLoss()#交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)#优化器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)#学习率调整策略


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
        inputs, labels = inputs.to(device), labels.to(device)#将数据放到GPU上
        labels = nn.functional.one_hot(labels, 10).float()  # one-hot编码
        optimizer.zero_grad()  # 梯度清零
        if args.merge > 1:

            lam = np.random.beta(1, 1)
            labels = mixt(labels, lam, args.merge)
            outputs = net(inputs,lam,2)#前向传播
        else:
            outputs = net(inputs)

        loss =loss_function(outputs, labels)#计算损失
        loss.backward()#反向传播
        optimizer.step()#更新参数
        train_loss += loss.item()#累加损失



    finish = time.time()  # 结束时间
    # 参数收集部分
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))  # 打印训练时间
    epoch_time = finish - start  # 训练时间
    lr = optimizer.param_groups[0]['lr']  # 学习率
    train_loss = train_loss / len(trainloader.dataset)  # 平均损失

    return {'epoch': epoch, 'train loss': train_loss, 'lr': lr, 'time': epoch_time}




def test(epoch):#测试函数
    global best_acc
    start = time.time()  # 开始时间
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():#不计算梯度
        for batch_idx, (inputs, labels) in enumerate(testloader):#遍历测试集
            inputs, labels = inputs.to(device), labels.to(device)#将数据放到GPU上
            outputs = net(inputs)
            loss = loss_function(outputs, labels)

            test_loss += loss.item()  # 累加损失
            _, preds = outputs.max(1)  # 取最大值
            correct += preds.eq(labels).sum()  # 累加正确数

    finish = time.time()  # 结束时间
    test_loss = test_loss / len(testloader.dataset),  # 平均损失
    test_acc = correct.float() / len(testloader.dataset)  # 正确率

    print('test time consumed: {:.2f}s'.format(finish - start))  # 打印测试时间
    print(f'Test set: Epoch: {epoch}, Average loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}')



    return {'test loss': test_loss[0], 'test acc': test_acc[0].item()}


timestamp = time.strftime("%Y%m%d_%H%M%S")

if args.gpu:
    filename = f"{args.merge}_{args.net}_lr{args.lr}_{timestamp}_3080cifar10.csv"
else:
    filename = f"{args.merge}_{args.net}_lr{args.lr}_{timestamp}_cpucifar10.csv"
with open(filename, mode='w', newline='') as csvfile:
    fieldnames = ['epoch', 'train loss', 'lr', 'test loss', 'test acc', 'time']  # 列名
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  # 创建csv文件 写入列名
    writer.writeheader()

for epoch in range(1, 251):
    text1 = train(epoch)  # 训练
    text2 = test(epoch)  # 评估
    text1.update(text2)
    fieldnames = ['epoch', 'train loss', 'lr', 'test loss', 'test acc', 'time']  # 列名
    reordered_dict = {key: text1[key] for key in fieldnames}
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # 加入用时和结果
        writer.writerow(text1)  # 写入行
        # 打印
        print(text1)

    scheduler.step()
