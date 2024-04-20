'''Train CIFAR10 with PyTorch.'''
import sys
print(sys.version)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import numpy as np

import time

from helper import load_cifar, load_cifar_5m, TransformingTensorDataset, find_prob, mse_loss

from models import mcnn_nobn_mu, resnet20_bn_mu

import wandb

import random

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

from typing import List

from mup import MuReadout, make_base_shapes, set_base_shapes, MuSGD, MuAdamW
import mup

wandb.init(project='underparam-limit-mu-z', allow_val_change=True)
wandb.config.update({'jobid': os.environ["SLURM_JOB_ID"]})

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--opt', '-o', default='adam_z', type=str, choices=['sgd', 'adam', 'adam_old', 'adam_z'])
parser.add_argument('--lr', '-lr', default=3e-4, type=float, help='learning rate')
parser.add_argument('--i_noise', '-i_n', default=-1, type=float)
parser.add_argument('--momentum', '-mo', default=.9, type=float)
parser.add_argument('--model', '-m', default='r20_bn', type=str)
parser.add_argument('--depth', '-d', default=-1, type=int)
parser.add_argument('--tr_sz', '-tr_sz', default=-1, type=int)
parser.add_argument('--epochs', '-e', default=10000000, type=int)
parser.add_argument('--weight_decay', '-wd', default=0.0, type=float)
parser.add_argument('--bsz', '-bsz', default=512, type=int)
parser.add_argument('--sched', '-sched', default=0, type=int)
parser.add_argument('--c', '-c', default=128, type=int)
parser.add_argument('--seed', '-seed', default=-1, type=int)
parser.add_argument('--n_mbatch', '-nm', default=1, type=int)
parser.add_argument('--parts', '-parts', nargs='+', default=[0,1,2,3,4,5], type=int)
parser.add_argument('--no_amp', '-no_amp', default=False, action='store_true')
parser.add_argument('--loss', '-loss', default='ce', type=str)
parser.add_argument('--save', '-save', default=1, type=int)
parser.add_argument('--z', '-z', default=1, type=int)
args = parser.parse_args()

wandb.config.update(args)


device = 'cuda'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('gg')

# Model
print('==> Building model..')

if args.depth == 2:
    widths = [args.fw]
if args.depth == 3:
    widths = [args.fw, args.sw]
if args.depth == 4:
    widths = [args.fw, args.sw, args.tw]


if args.seed >= 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    

print('a', args.model, flush=True)
print('b', flush=True)
if args.model == 'myrtle_inf':
    net = mcnn_nobn_mu(width=args.c)
    base_net = mcnn_nobn_mu(width=2)
    delta_net = mcnn_nobn_mu(width=4)
    set_base_shapes(net, base_net, delta=delta_net)

    for param in net.parameters():
        if len(list(param.shape)) >= 2:
            mup.init.kaiming_uniform_(param)
        else:
            torch.nn.init.zeros_(param)
if args.model == 'r20_bn':
    net = resnet20_bn_mu(args.c)
    base_net = resnet20_bn_mu(1*args.z)
    delta_net = resnet20_bn_mu(2*args.z)
    set_base_shapes(net, base_net, delta=delta_net)

    for param in net.parameters():
        if len(list(param.shape)) >= 2:
            print(param.shape, 'mup')
            mup.init.kaiming_normal_(param)
        else:
            print(param.shape, 'default')

        param.weight.data = param.weight.data*.3





net = net.to(memory_format=torch.channels_last).to(device)
net = net.to(device)

#pytorch_total_params = sum(p.numel() for p in net.parameters())
#print(pytorch_total_params)
#exit()


print('There are unnecessary delays in the code. They are currently required to prevent the code from crashing. See: Ticket 161303')

if args.loss == 'ce':
    criterion = nn.CrossEntropyLoss()
elif args.loss == 'mse':
    criterion = mse_loss

if args.opt == 'sgd':
    optimizer = MuSGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
if args.opt == 'adam':
    optimizer = MuAdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay = args.weight_decay)
if args.opt == 'adam_old':
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay = args.weight_decay)
if args.opt == 'adam_z':
    for param in net.linear.parameters():
        param.requires_grad = False
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay = args.weight_decay)

if args.sched == 1:  
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=1.0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[args.epochs])
if args.sched == 2:
    first_milestone = 2500
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=first_milestone)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-first_milestone+10, eta_min=1e-6)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=1.0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2, scheduler3], milestones=[first_milestone, args.epochs])

epoch = 0
log_idx = 0

best_train_error = 100.0
best_test_error = 100.0

scaler = GradScaler(enabled=not args.no_amp)

def train(loader, testloader):
    global epoch, best_train_error, best_test_error, log_idx
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0.0
    total = 0.0
    prob = 0.0
    outnorm = 0.0
    outnorm_prob = 0.0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=not args.no_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            prob_cur = find_prob(outputs, targets, args.loss)
        scaler.scale(loss/args.n_mbatch).backward()
        if batch_idx % args.n_mbatch == args.n_mbatch-1:
            scaler.step(optimizer)
            scaler.update()
            epoch += 1
            if args.sched >= 1:
                scheduler.step()
        
        train_loss += loss.item()*targets.size(0)
        prob += prob_cur.item()
        outnorm = torch.sum(outputs**2).item()
        if args.loss == 'ce':
            outnorm_prob += torch.sum(F.softmax(outputs, dim=-1)**2).item()
        elif args.loss == 'mse':
            outnorm_prob += torch.sum(torch.abs(outputs)).item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        
        if epoch >= 1+log_idx+.5*(log_idx**2)+(1.1**log_idx) or epoch == args.epochs:
            if log_idx < 25 and False:
                log_idx += 2
            else:
                log_idx += 1
            train_err = 100.0*(1.0-correct/total)
            if train_err < best_train_error:
                best_train_error = train_err

            if args.model == 'myrtle_inf':
                last_layer_norm = torch.sum(net[-1].weight**2).item()
            elif args.model == 'r20_bn':
                last_layer_norm = torch.sum(net.linear.weight**2).item()

            if args.sched >= 1:
                print(scheduler.get_last_lr())
                wandb.log({'train error': 100.0*(1.0-correct/total), 'train loss': train_loss/total, 'train soft error': 100.0*(1.0-prob/total), 'best train error': best_train_error, 'lr': optimizer.param_groups[0]["lr"], 'Last Layer Norm': last_layer_norm, 'step': epoch, 'outnorm': outnorm/total, 'outnorm_prob': outnorm_prob/total}, step = epoch)
            else:
                wandb.log({'train error': 100.0*(1.0-correct/total), 'train loss': train_loss/total, 'train soft error': 100.0*(1.0-prob/total), 'best train error': best_train_error, 'Last Layer Norm': last_layer_norm, 'step': epoch, 'outnorm': outnorm/total, 'outnorm_prob': outnorm_prob/total}, step = epoch)
            
            
            
            
            test(testloader)

            if args.loss == 'mse' and train_loss/total < 1e-5:
                exit()

            train_loss = 0
            correct = 0.0
            total = 0.0
            prob = 0.0
            outnorm = 0.0
            outnorm_prob = 0.0
            
            if epoch >= args.epochs:
                exit()

            
            net.train()

from pathlib import Path
model_save_folder = f'/n/holystore01/LABS/barak_lab/Users/nvyas/models/{wandb.run.project}/{wandb.run.name}/'
Path(model_save_folder).mkdir(parents=True, exist_ok=True)    


output_save_folder = f'/n/holystore01/LABS/barak_lab/Users/nvyas/logs/{wandb.run.project}/{wandb.run.name}/'
Path(output_save_folder).mkdir(parents=True, exist_ok=True) 


def test(loader):
    print('test')

    if len(args.parts) == 1 and args.parts[0] == -2:
        outputs_all = np.zeros((10000, 10))
    else:
        outputs_all = np.zeros((40000, 10))

    global epoch, best_test_error
    net.eval()

    if (epoch > args.epochs or args.c < 500) and epoch > 210 and args.save == 1: 
        torch.save(net.state_dict(), f'{model_save_folder}{epoch}.pt')

    test_loss = 0.0
    prob = 0.0
    correct = 0.0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            #print(batch_idx)
            with autocast(enabled=not args.no_amp):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                prob_cur = find_prob(outputs, targets, args.loss)
            loss = criterion(outputs, targets)

            test_loss += loss.item()*targets.size(0)
            prob += prob_cur.item()
            _, predicted = outputs.max(1)
            outputs_all[total:total+targets.size(0)] = outputs.cpu().numpy()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_err = 100.0*(1.0-correct/total)
    if test_err < best_test_error:
        best_test_error = test_err
    
    wandb.log({'test error': 100.0*(1.0-correct/total), 'test loss': test_loss/total, 'test soft error': 100.0*(1.0-prob/total), 'best test error': best_test_error}, step = epoch)

    np.save(f'{output_save_folder}{epoch}.npy', outputs_all)

    return

print('==> Preparing data..')
print(args.parts)

if args.seed == -1:
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

def make_dataloaders(train_dataset=None, val_dataset=None, batch_size=None, num_workers=None):
    paths = {
        'train': train_dataset,
        'test': val_dataset
    }

    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    loaders = {}

    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        #if name == 'train': #Commented as want to have the same distribution for train and test as well as log outputs to deterministically compare.
        #    image_pipeline.extend([
        #        RandomHorizontalFlip(),
        #        RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
        #        Cutout(4, tuple(map(int, CIFAR_MEAN))),
        #    ])
        if args.no_amp:
            image_pipeline.extend([
                ToTensor(),
                ToDevice('cuda:0', non_blocking=True),
                ToTorchImage(),
                Convert(torch.float32),
                torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
        else:
            image_pipeline.extend([
                ToTensor(),
                ToDevice('cuda:0', non_blocking=True),
                ToTorchImage(),
                Convert(torch.float16),
                torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
        
        ordering = OrderOption.QUASI_RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        if name == 'train':
            os_cache = False
        else:
            os_cache = True

        if args.seed == -1:
            SEED = 1
        else:
            SEED = args.seed

        loaders[name] = Loader(paths[name], batch_size=batch_size, num_workers=num_workers,
                               order=ordering, drop_last=(name == 'train'), os_cache=os_cache,
                               pipelines={'image': image_pipeline, 'label': label_pipeline}, seed=SEED)

    return loaders

test_path = f'/n/holystore01/LABS/barak_lab/Everyone/datasets/cifar-5m/cifar_test.beton'

args.parts.sort()



if len(args.parts) == 6:
    train_name = 'train'
    wandb.log({'epoch': 5000000//args.bsz})
elif len(args.parts) == 1 and args.parts[0] == -1:
    train_name = 'train_small'
    wandb.log({'epoch': 200000//args.bsz})
elif len(args.parts) == 1 and args.parts[0] == -2:
    train_name = '10_train'
    wandb.log({'epoch': 50000//args.bsz})
    test_path = '/n/holystore01/LABS/barak_lab/Everyone/datasets/cifar-5m/cifar_10_test.beton'
else:
    train_parts = ''.join(str(x) for x in args.parts)
    train_name = 'train_'+train_parts
    wandb.log({'epoch': len(args.parts)*5000000//(6*args.bsz)})

train_path = f'/n/holystore01/LABS/barak_lab/Everyone/datasets/cifar-5m/cifar_{train_name}.beton'

loaders = make_dataloaders(train_path, test_path, args.bsz//args.n_mbatch, 1)

trainloader = loaders['train']
testloader = loaders['test']

for epoch2 in range(0, 1000000):

    train(trainloader, testloader)