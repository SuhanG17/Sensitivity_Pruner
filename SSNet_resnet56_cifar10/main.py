import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import sys
import argparse
import numpy as np
import shutil
# import math
# from torchvision import models
from torchvision import datasets, transforms

import SSNet_ResNet
import Comp_ssty

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--batch_size', default=12, type=int, help='batch size')
parser.add_argument('--num_epochs', default=8, type=int, help='number of training epochs')
parser.add_argument('--lr_decay_epoch', default=10, type=int, help='learning rate decay epoch')
parser.add_argument('--data_base', default='/nas/imagenet', type=str, help='the path of dataset')
parser.add_argument('--ft_model_path', default='/nas/laibl/AutoPruner-master/SSNet_1/results/models/model_resnet56.pth',
                    type=str, help='the path of fine tuned model')
parser.add_argument('--gpu_id', default='2', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--group_id', default=2, type=int, help='the id of compressed group, starting from 0')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--compression_rate', default=0.4, type=float, help='the percentage of 1 in compressed model')
parser.add_argument('--channel_index_range', default=20, type=int, help='the range to calculate channel index')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--beta_range', default=100, type=int, help='the range to calculate channel index')
args = parser.parse_args()
print(args)
best_prec1 = -1
resnet_channel_number = [6, 8, 12, 6]
channel_num = [[64, 64]*3, [128, 128]*4, [256, 256]*6, [512, 512]*3]
channel_index_list = None
beta_index = 0
beta_list = None
# threshold = 95 * np.ones(resnet_channel_number[args.group_id])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Pruning group is: '+ str(args.group_id))
print('Pruning group channel num: ' + str(channel_num[args.group_id]))

def main():
    global args, best_prec1, channel_index_list, resnet_channel_number, channel_num, beta_list
    # Phase 1 : Data Upload
    print('Data Preparing...')
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='/nas/laibl/data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, 32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=10, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='/nas/laibl/data', train=False, transform=transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize,
        ]),download=True),
        batch_size=args.batch_size,
        num_workers=8, pin_memory=True)
    print('Data Load Success!')

    # Phase 2 : Model setup
    print('Model setup')
    if args.group_id == 0:
        model_ft = SSNet_ResNet.ResNet50().to(device)
        torch.save(model_ft, 'results/models/model_1.pth')
        model_ft = torch.nn.DataParallel(model_ft).to(device)
        torch.save(model_ft, 'results/models/model.pth')
    model_ft = SSNet_ResNet.ResNet50_ssnet(args.group_id).to(device)
    model_ft = torch.nn.DataParallel(model_ft)
    cudnn.benchmark = True
    org_model = Comp_ssty.init_sensity_model()
    print("Model setup success!")

    # Phase 3: fine_tune model
    print('Model finetuning...')
    # define loss function (criterion) and optimizer
    criterion1 = nn.CrossEntropyLoss().to(device)
    criterion2 = nn.L1Loss().to(device)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    tmp = np.linspace(1, 100, int(args.num_epochs * len(train_loader) / args.beta_range))
    beta_list = np.ones([resnet_channel_number[args.group_id], len(tmp)])
    for tmp_i in range(resnet_channel_number[args.group_id]):
        beta_list[tmp_i, :] = tmp.copy()
    regulization = 0.1 * np.ones(resnet_channel_number[args.group_id])
    for epoch in range(args.start_epoch, args.num_epochs):
        adjust_learning_rate(optimizer, epoch, int(args.num_epochs/2.0))
        # train for one epoch
        channel_index = train(train_loader, model_ft, org_model, criterion1, criterion2, optimizer, epoch, regulization)

        # torch.save(channel_index,'channel.pth')
        # evaluate on validation set
        # channel_index_ = torch.load('channel.pth')
        # channel_index = []
        # for ch in channel_index_:
        #     channel_index.append(np.ones_like(ch))
        prec1 = validate(val_loader, model_ft, criterion1, channel_index)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
            folder_path = 'results/models/group_' + str(args.group_id)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            torch.save(model_ft, folder_path+'/model.pth')
            if args.group_id == 3:
                tmp = channel_index[0].copy()
                tmp[:] = 1.0
                channel_index.append(tmp.copy())
                channel_index.append(tmp.copy())
            torch.save(channel_index, folder_path+'/channel_index.pth')


def train(train_loader, ft_model, org_model, criterion1, criterion2, optimizer, epoch, regulization):
    global resnet_channel_number, beta_list, beta_index, channel_num
    gpu_num = torch.cuda.device_count()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    ft_model.train()
    channel_index_list = list()
    channel_index_binary = list()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if i % args.beta_range == 0:
            if beta_index == beta_list.shape[1]:
                beta_index = beta_index - 1
            beta = beta_list[:, beta_index]
            beta_index = beta_index + 1

        ft_model.module.set_beta_factor(beta)

        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input).to(device)
        target_var = torch.autograd.Variable(target).to(device)

        # compute output
        ssty, target_channel = Comp_ssty.Setting_Sensity(org_model, input_var, target_var, args.compression_rate, args.group_id, gpu_num)
        output, mask_vec = ft_model(input_var, ssty)

        loss = criterion1(output, target_var)
        for idx in range(len(mask_vec)):
            mask_vec[idx] = mask_vec[idx].reshape(-1, channel_num[args.group_id][idx])
            loss += regulization[idx] * criterion2(mask_vec[idx], target_channel[idx])

        # compute channel index
        channel_vector = list()
        k = resnet_channel_number[args.group_id]
        for j in range(k):
            tmp = mask_vec[j].data.cpu().numpy().mean(0)
            channel_vector.append(tmp.copy())
            if i == 0:
                print('first 5 values in layer {0}: [{1:.6f}, {2:.6f}, {3:.6f}, {4:.6f}, {5:.6f}]'.format(j, tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]))
        
        channel_index_list.append(channel_vector.copy())

        if len(channel_index_list) == args.channel_index_range:
            channel_index_list = np.array(channel_index_list)
            channel_index_binary = list()
            for j in range(k):
                tmp = channel_index_list[args.channel_index_range-1][j]
                cur_two_side_rate = (np.sum(tmp > 0.8) + np.sum(tmp < 0.2)) / len(tmp)
                print('first 5 values in layer {0}: [{1:.6f}, {2:.6f}, {3:.6f}, {4:.6f}, {5:.6f}]'.format(j, tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]))
            
                ## 每channel_index_range次计算一次通道向量值，和映射出来的0，1向量
                tmp2 = np.array(channel_index_list[:, j]).sum(axis=0)
                tmp2 = tmp2 / args.channel_index_range
                for tmp_i in range(len(channel_index_list)):
                    channel_index_list[tmp_i,j] = (np.sign(channel_index_list[tmp_i,j] - 0.5) + 1) / 2.0

                ## 计算第channel_index_range次计算出的通道向量值和映射出来的0，1向量
                tmp = np.array(channel_index_list[:, j]).sum(axis=0)
                tmp = tmp / args.channel_index_range
                channel_index = (np.sign(tmp - 0.5) + 1) / 2.0  # to 0-1 binary
                
                channel_index_binary.append(channel_index.copy())

                cur_real_pruning_rate = 100.0 * np.sum(tmp2 < 10**-6) / len(tmp2)
                cur_binary_pruning_rate = 100.0 * np.sum(channel_index < 10**-6) / len(channel_index)

                tmp[tmp == 0] = 1
                cur_channel_inconsistency = 100.0 * np.sum(tmp != 1) / len(tmp)

                ratio = target_channel[j].cpu().norm(1)/float(target_channel[j].cpu().size(0))
                
                print( "layer [{0}] (real/binary rate): {1:.4f}%/{2:.4f}%, (target rate): {3:.4f}%, (inconsistency): {4:.4f}%, (lambda): {5:.4f}, (beta): {6:.4f}, (two side rate): {7:.4f}".format(
                        int(j), cur_real_pruning_rate, cur_binary_pruning_rate, 100.0*(1 - ratio), cur_channel_inconsistency, regulization[j], beta[j], cur_two_side_rate))
                
                if cur_two_side_rate < 0.9 and beta_index >= int(beta_list.shape[1] / args.num_epochs):
                    beta_list[j, :] = beta_list[j, :] + 1

                x = cur_binary_pruning_rate / 100.0
                if 1 - x < ratio:
                    regulization[j] = 0.0
                else:
                    regulization[j] = 5.0 * np.abs(x - 1 + ratio)

                sys.stdout.flush()
            channel_index_list = list()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch[{0}]: [{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   top1=top1, top5=top5))
            print('+---------------------------------------------------------------------------------------------------------------------------------------------------------+')
            sys.stdout.flush()

    return channel_index_binary


def validate(val_loader, model, criterion, channel_index):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        input_var = torch.autograd.Variable(input).to(device)
        target_var = torch.autograd.Variable(target).to(device)

        # compute output
        channel_index_ = list()
        for idx in range(len(channel_index)):
            tmp = torch.repeat_interleave(torch.Tensor(channel_index[idx]).unsqueeze(dim=0), repeats=input_var.shape[0], dim = 0)
            tmp = torch.autograd.Variable(tmp).to(device)
            channel_index_.append(tmp)
        output, _ = model(input_var, None, channel_index_)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, epoch_num):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // epoch_num))
    print('| Learning Rate = %f' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    main()
