import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sys, os
sys.path.append(os.path.abspath("model"))

def get_transform(args):
    if args.dataset in ['c10','c100']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(args.size, padding=args.padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)
        ])
    else:
        raise NotImplementedError()

    return train_transform, test_transform

def get_dataset(args):
    data_path = "data"
    if args.dataset=='c10':
        args.in_c=3
        args.num_classes = 10
        args.size=32
        args.padding=4
        args.mean,args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR10(data_path, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR10(data_path, train=False, transform=test_transform, download=True)
    elif args.dataset=='c100':
        args.in_c=3
        args.num_classes = 100
        args.size=32
        args.padding=4
        args.mean,args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR100(data_path, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR100(data_path, train=False, transform=test_transform, download=True)
    else:
        raise NotImplementedError()

    return train_ds, test_ds


def get_model(args):
    if args.model == 'spreact18':
        from model.spreact18 import SPreAct18
        model = SPreAct18(args.in_c, args.num_classes)
    elif args.model == 'preact18':
        from model.preact18 import PreAct18
        model = PreAct18(args.in_c, args.num_classes)
    else:
        raise NotImplementedError()

    return model
