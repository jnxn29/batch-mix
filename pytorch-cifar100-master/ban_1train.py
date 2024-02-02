# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
# from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import csv
import random
import matplotlib.pyplot as plt

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from torch.cuda.amp import autocast, GradScaler

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()
    test_loss = 0.0
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        with autocast():
            outputs = net(images)
            loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()  

    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch, 
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start  
    ))
    test_loss = test_loss / len(cifar100_test_loader.dataset),
    test_acc = correct.float() / len(cifar100_test_loader.dataset),
    print()

    return {'test loss': test_loss[0], 'test acc': test_acc[0].item()}



def train(epoch,filename):

    start = time.time()
    net.train()
    scaler = GradScaler()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        with autocast():
            outputs = net(images)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))
        train_loss=loss.item()

        if epoch <= args.warm:
            warmup_scheduler.step()

    finish = time.time()



    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    epoch_time = finish - start
    lr = optimizer.param_groups[0]['lr']

    return {'epoch': epoch, 'train loss': train_loss, 'lr': lr, 'time': epoch_time}



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')

    args = parser.parse_args()
    net = get_network(args)
    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=6,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=6,
        batch_size=args.b,
        shuffle=True
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.gpu:
        filename = f"fp16{args.net}_{timestamp}_3080.csv"
    else:
        filename = f"fp16{args.net}_{timestamp}_cpu.csv"

    with open(filename, mode='w', newline='') as csvfile:
        fieldnames = ['epoch', 'train loss', 'lr', 'test loss', 'test acc', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)


    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
                       train_scheduler.step()


        text1=train(epoch,filename)
        text2=eval_training(epoch)
        text1.update(text2)
     
        fieldnames = ['epoch', 'train loss', 'lr', 'test loss', 'test acc', 'time']
        reordered_dict = {key: text1[key] for key in fieldnames}
        with open(filename, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(text1)
            print(text1)



