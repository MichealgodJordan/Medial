'''
Author: Robber swag162534@outlook.com
Date: 2024-12-08 21:13:33
LastEditors: Robber swag162534@outlook.com
LastEditTime: 2025-01-14 21:02:08
FilePath: \research\su\kneecodes\my_supcon.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from torch import randperm

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model, accuracy
from resnet_big import SupConResNet
from mylosses import SupConLoss
from tqdm import tqdm

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

import disfunc


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=5e-3,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='300,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='knee', help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--dataset_size', type = float, default = 0.9, help = 'rate for the size of train set')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--dis_func', type=str, default='Manhattan', choices=['default', 'L2','Norm_L2', 'Cor', 'Manhattan'],
                        help='choose distance function for loss calculation')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = r'D:\research\su\data\Knee\Digital_Knee_X-ray_Images\MedicalExpert-I'
    opt.model_path = '../autodl-tmp/save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = '../autodl-tmp/save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_{}_pre{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
                opt.weight_decay, opt.batch_size, opt.temp, opt.trial, opt.dis_func, opt.dataset_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

class CustomSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[self.indices[index]]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)

def set_loader(opt):
    # construct data loader
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    normalize = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

    train_transform = transforms.Compose([
        transforms.Resize([opt.size,opt.size]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    img_dataset = datasets.ImageFolder(opt.data_folder)
    class_names = img_dataset.classes
    size = len(img_dataset)
    img_dataset = torch.utils.data.Subset(img_dataset, randperm(size).tolist())
    train_indices = range(0, int(size*opt.dataset_size))
    train_set = CustomSubset(img_dataset, train_indices, TwoCropTransform(train_transform))
    print(f'train dataset size : {len(train_set)}')
    print(f'classes names : {class_names}')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, 
                                               shuffle=True,num_workers=opt.num_workers, pin_memory=True)
    return train_loader

def set_sim(opt):
    if opt.dis_func == 'L2':
        print('dis_func:L2')
        return disfunc.L2
    elif opt.dis_func == 'Norm_L2':
        print('dis_func:Norm_L2')
        return disfunc.norm_L2
    elif opt.dis_func == 'Cor':
        print('dis_func:Cor')
        return disfunc.Cor
    elif opt.dis_func == 'Manhattan':
        print('dis_func:Manhattan')
        return disfunc.Manhattan_Distance
    else:
        print('No dis_func applied')
        return torch.matmul

def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp, dis_func=set_sim(opt))

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    torch.cuda.empty_cache()
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    correct = 0
    total = 0

    end = time.time()
    for idx, (images, labels) in tqdm(enumerate(train_loader)):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                            format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # if loss.items() < 0.4:
        #     correct += bsz
        # total += bsz

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'loss {loss.val:.3f} ({loss.avg:.3f}\t'
                'acc {acc:.3f}'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,acc=correct/total if total != 0 else 0))
            sys.stdout.flush()

    return losses.avg

# def val(test_loader, model,criterion,opt):
#     """one epoch testing"""
#     torch.cuda.empty_cache()
#     model.eval()
#     with torch.no_grad():
#         accs = []
#         for i , (images, labels) in tqdm(enumerate(test_loader)):

#             images = torch.cat([images[0], images[1]], dim=0).cuda()
#             features = model(images)

#             bsz = labels.cuda().shape[0]
#             # compute accuracy
#             f1, f2 = torch.split(features, [bsz, bsz], dim=0)
#             features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            
#             acc = accuracy(features, labels)
#             accs.append(acc.item())
#     return accs # list with only one element

def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in 1, opt.epochs + 1:
        print("")
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)



    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
