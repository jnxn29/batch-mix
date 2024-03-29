import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import os

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None



    elif args.dataset == "imagenet":

        traindir = os.path.join("../../.imagenet", 'train')

        valdir = os.path.join("../../.imagenet", 'val')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(

            traindir,

            transform=transforms.Compose([

                transforms.RandomResizedCrop(224),

                transforms.RandomHorizontalFlip(),

                transforms.ToTensor(),

                normalize,

            ]))

        test_dataset = datasets.ImageFolder(

            valdir,

            transform=transforms.Compose([

                transforms.Resize(256),

                transforms.CenterCrop(224),

                transforms.ToTensor(),

                normalize,

            ]))

        trainset = train_dataset

        testset = test_dataset if args.local_rank in [-1, 0] else None

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)

    test_sampler = SequentialSampler(testset)

    train_loader = DataLoader(trainset,

                              sampler=train_sampler,

                              batch_size=args.train_batch_size,

                              num_workers=4,

                              pin_memory=True)

    test_loader = DataLoader(testset,

                             sampler=test_sampler,

                             batch_size=args.eval_batch_size,

                             num_workers=4,

                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader

