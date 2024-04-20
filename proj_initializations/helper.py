import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset

import pickle

from functools import partial
from einops import rearrange

def normalize(net, typ='myrtle'):
    if typ == 'myrtle':
        for i in range(len(net)):
            if isinstance(net[i], nn.Conv2d):
                mean_norm = torch.mean(net[i].weight.data**2)**.5
                net[i].weight.data = net[i].weight.data / (torch.sqrt(torch.sum(net[i].weight.data**2, dim=1, keepdim=True)))
                net[i].weight.data *= (mean_norm/torch.mean(net[i].weight.data**2)**.5)
                net[i].bias = None


def load_cifar():
    train_ds = datasets.CIFAR10(root='~/tmp/data', train=True,
                           download=True, transform=None)
    test_ds = datasets.CIFAR10(root='~/tmp/data', train=False,
                          download=True, transform=None)

    def to_xy(dataset):
        X = torch.Tensor(np.transpose(dataset.data, (0, 3, 1, 2))).float() / 255.0
        #X = 2.0*X - 1.0 # [0, 1] --> [-1, 1]
        Y = torch.Tensor(dataset.targets).long()
        return X, Y

    X_tr, Y_tr = to_xy(train_ds)
    X_te, Y_te = to_xy(test_ds)
    return X_tr, Y_tr, X_te, Y_te

def load_cifar_5m(parts):
    parts.sort()


    for ind in parts:
        part_ind = np.load(f'/n/holystore01/LABS/barak_lab/Everyone/datasets/cifar-5m/part{ind}.npz')
        print(part_ind['X'].dtype, flush=True)
        X_ind = np.transpose(part_ind['X'], (0, 3, 1, 2))
        Y_ind = part_ind['Y']
        print(X_ind.dtype, flush=True)
        print(ind, flush=True)
        if ind == parts[0]:
            X_all = X_ind
            Y_all = Y_ind
        else:
            X_all = np.concatenate((X_all, X_ind))
            Y_all = np.concatenate((Y_all, Y_ind))
        
    X_tr, Y_tr = torch.ByteTensor(X_all[:-40000]), torch.Tensor(Y_all[:-40000]).long()

    if parts[-1] == 5:
        X_te, Y_te = torch.ByteTensor(X_all[-40000:]), torch.Tensor(Y_all[-40000:]).long()
    else:
        part_ind = np.load(f'/n/holystore01/LABS/barak_lab/Everyone/datasets/cifar-5m/part5.npz')
        X_ind = np.transpose(part_ind['X'], (0, 3, 1, 2))
        Y_ind = part_ind['Y']
        X_te, Y_te = torch.ByteTensor(X_ind[-40000:]), torch.Tensor(Y_ind[-40000:]).long()

    return X_tr, Y_tr, X_te, Y_te

def load_cifar_5m_numpy(parts):
    parts.sort()
    for ind in parts:
        part_ind = np.load(f'/n/holystore01/LABS/barak_lab/Everyone/datasets/cifar-5m/part{ind}.npz')
        print(part_ind['X'].dtype, flush=True)
        X_ind = part_ind['X']
        Y_ind = part_ind['Y']
        print(X_ind.dtype, flush=True)
        print(ind, flush=True)
        if ind == parts[0]:
            X_all = X_ind
            Y_all = Y_ind
        else:
            X_all = np.concatenate((X_all, X_ind))
            Y_all = np.concatenate((Y_all, Y_ind))
        
    X_tr, Y_tr = X_all[:-40000], Y_all[:-40000]

    if parts[-1] == 5:
        X_te, Y_te = X_all[-40000:], Y_all[-40000:]
    else:
        part_ind = np.load(f'/n/holystore01/LABS/barak_lab/Everyone/datasets/cifar-5m/part5.npz')
        X_ind = np.transpose(part_ind['X'], (0, 3, 1, 2))
        Y_ind = part_ind['Y']
        X_te, Y_te = X_ind[-40000:], Y_ind[-40000:]

    return X_tr, Y_tr, X_te, Y_te

class NumpyDataset(Dataset):
    """TensorDataset with support of torchvision transforms.
    """
    def __init__(self, X, Y):
        #assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]

        return x, y

def load_cifar_5m_test():
    part_ind = np.load(f'/n/holystore01/LABS/barak_lab/Everyone/datasets/cifar-5m/part5.npz')
    X_ind = np.transpose(part_ind['X'], (0, 3, 1, 2))
    Y_ind = part_ind['Y']

    X_te, Y_te = torch.ByteTensor(X_ind[-40000:]), torch.Tensor(Y_ind[-40000:]).long()

    return X_te, Y_te



class TransformingTensorDataset(Dataset):
    """TensorDataset with support of torchvision transforms.
    """
    def __init__(self, X, Y, aug, noise=-1):
        #assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.X = X
        self.Y = Y
        self.transform = get_data_aug_transform(aug)
        if noise > 0:
            self.noise = torch.randn_like(X)*noise
        else:
            self.noise = None

    def __getitem__(self, index):
        x = self.X[index].float()/255.0
        if self.transform:
            x = self.transform(x)
        if self.noise is not None:
            x += self.noise[index]
        y = self.Y[index]

        return x, y

    def __len__(self):
        return len(self.X)
    
def find_prob(yM, yB, loss_type):
    y = nn.functional.one_hot(yB, yM.size(1)).double()
    y2 = y.ge(.5)
    if loss_type == 'mse':
        g = torch.masked_select(yM, y2)
        g = torch.reshape(g, [yM.size(0)])
        g2 = 2-g
        g = torch.min(g, g2)
        g = torch.sum(g)
    elif loss_type == 'ce':
        g = torch.masked_select(nn.functional.softmax(yM, dim=1), y2)
        g = torch.reshape(g, [yM.size(0)])
        g = torch.sum(g)
    return g

def get_data_aug_transform(aug):
    """
        Returns a torchvision transform that maps (normalized Tensor) --> (normalized Tensor)
        via a random data augmentation.
    """
    if aug == 'none':
        return transforms.Compose([
            #transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    if aug == 'truenone':
        return None
    if aug == 'standard':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    if aug == 'heavy1':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    if aug == 'heavy2':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(5, sigma=0.4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    print('aug no found')
    exit()

def mse_loss(y, y_hat):
    y_hat = F.one_hot(y_hat, 10)
    y_hat = y_hat.type(y.dtype)
    return torch.mean((y - y_hat)**2)