
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,3,0'
import sys
import numpy as np
import torch
from torch import nn
import shutil
import argparse
import torchvision
import torch.utils.data
import SSNet
import time
from torchvision import transforms, datasets, models
import torch.backends.cudnn as cudnn
import torch.nn as nn
import Comp_ssty
import copy
import torch.distributed as dist 

parser = argparse.ArgumentParser(description='PyTorch Digital Sensitivity Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--batch_size', default=24, type=int, help='batch size')
parser.add_argument('--num_epochs', default=1, type=int, help='number of training epochs')
parser.add_argument('--lr_decay_epoch', default=10, type=int, help='learning rate decay epoch')
parser.add_argument('--data_base', default='/nas/imagenet', type=str, help='the path of dataset')
parser.add_argument('--train_path', default='datasets_64/prune/train_loader', type=str, help='the path of dataset')
parser.add_argument('--val_path', default='datasets_64/prune/val_loader', type=str, help='the path of dataset')
parser.add_argument('--ft_model_path', default='results/models/model_vgg16.pth',
                    type=str, help='the path of fine tuned model')
parser.add_argument('--gpu_id', default='1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--layer_id', default=13, type=int, help='the id of compressed layer, starting from 0')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--compression_rate', default=0.2, type=float, help='the percentage of 1 in compressed model')
parser.add_argument('--channel_index_range', default=20, type=int, help='the range to calculate channel index')
parser.add_argument('--beta_range', default=100, type=int, help='the range to calculate channel index')
parser.add_argument('--print_freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
args = parser.parse_args()
print(args)
best_prec1 = -1
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
model_path = 'results/models/model_1.pth'
beta_idx = 0
beta_list = list()
n_gpu = torch.cuda.device_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


channel_len = [3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 4096, 4096, 1000]

def main():
    print('Data Preparing...')
    global args, best_prec1, beta_list
    train_loader = torch.load(args.train_path)
    val_loader = torch.load(args.val_path)
    print('Data Load Success!')

    print('Model Loading...')
    if args.layer_id == 0:
        model = SSNet.Vgg16()
        torch.save(model,'results/models/model_1.pth')
        model = nn.DataParallel(model).to(device)
        torch.save(model,'results/models/model.pth')
        # model = models.vgg16_bn(pretrained=True)
    if args.layer_id < 13:
        model = SSNet.SSnet(args.layer_id)
    else:
        model = SSNet.SSnet_FC(args.layer_id)
    model = nn.DataParallel(model).to(device)
    cudnn.benchmark = True # 提升一点训练速度，没什么额外开销
    print('Model Load Success!')

    print('Model Finetuning...')
    criterion1 = nn.CrossEntropyLoss().to(device)
    criterion2 = nn.L1Loss().to(device)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,momentum=0.9, weight_decay=args.weight_decay)
    beta_list = np.linspace(0.1,10, int(args.num_epochs * len(train_loader) / args.beta_range))
    regulization = 0.1
    old_model = Comp_ssty.init_sensity_model()

    for epoch in range(args.start_epoch,args.num_epochs):
        adjust_learning_rate(optimizer, epoch, 2)

        # train the network
        channel_index, regulization = train(train_loader, model, old_model, criterion1, criterion2, optimizer, epoch, regulization)

        # validate
        prec1 = validate(val_loader, model, old_model, criterion1, channel_index)

        # remember best prec@1 and save models
        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
            folder_path = 'results/models/layer_' + str(args.layer_id)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            torch.save(model, folder_path+'/model.pth')
            torch.save(channel_index, folder_path+'/channel_index.pth')




def train(train_loader, model, old_model, criterion1, criterion2, optimizer, epoch, regulization):
    global beta_list, beta_idx

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    channel_vector = []
    channel_index = 0
    end = time.time()
    
    for i,(inputs, targets) in enumerate(train_loader):
        if not i % args.beta_range:
            if beta_idx == len(beta_list):
                beta_idx = len(beta_list) - 1
            beta = beta_list[beta_idx]
            beta_idx += 1
        
        data_time.update(time.time() - end)
        # t1 = time.time()
        input_var = torch.autograd.Variable(inputs).to(device)
        target_var = torch.autograd.Variable(targets).to(device)

        ssty, target_channel = Comp_ssty.Setting_Sensity(old_model, input_var, target_var, args.compression_rate, args.layer_id)
        # t2 = time.time()
        ssty = torch.autograd.Variable(ssty).to(device)
        target_channel = torch.autograd.Variable(target_channel).to(device)

        output, mask_vec = model(input_var, ssty, beta)

        mask_vec = mask_vec.reshape(-1,channel_len[args.layer_id+1])

        # t3 = time.time()
        loss1 = criterion1(output, target_var)
        loss2 = criterion2(mask_vec.float(), target_channel.float())
        loss = loss1 + float(regulization) * loss2
        tmp = mask_vec.data.cpu().numpy().mean(0)
        channel_vector.append(tmp.copy())
        # t4 = time.time()
        if i == 0:
            ## 
            # print(t2-t1,t3-t2,t4-t3)

            print('first 5 values: [{0:.6f}, {1:.6f}, {2:.6f}, {3:.6f}, {4:.6f}]'.format(tmp[0], tmp[1], tmp[2], tmp[3],tmp[4]))

        if len(channel_vector) == args.channel_index_range:
            # print(t2-t1,t3-t2,t4-t3)

            channel_vector = np.array(channel_vector)
            # print(channel_vector.shape)
            
            tmp_value = channel_vector[args.channel_index_range-1, :]
            two_side_rate = (np.sum(tmp_value > 0.8) + np.sum(tmp_value < 0.2)) / len(tmp_value)
            tmp = tmp_value
            print('first 5 values: [{0:.6f}, {1:.6f}, {2:.6f}, {3:.6f}, {4:.6f}]'.format(tmp[0], tmp[1], tmp[2], tmp[3],tmp[4]))
            
            ## 
            tmp2 = channel_vector.sum(axis=0)
            tmp2 = tmp2 / args.channel_index_range
            for tmp_i in range(len(channel_vector)):
                channel_vector[tmp_i] = (np.sign(channel_vector[tmp_i] - 0.5) + 1) / 2.0

            ## 
            tmp = channel_vector.sum(axis=0)
            tmp = tmp / args.channel_index_range
            channel_index = (np.sign(tmp - 0.5) + 1) / 2.0  # to 0-1 binary
            # print(channel_index)
            ## 
            real_pruning_rate = 100.0 * np.sum(tmp2 < 10**-6) / len(tmp2)
            binary_pruning_rate = 100.0 * np.sum(channel_index < 10**-6) / len(channel_index)
            ratio = target_channel.cpu().norm(1)/float(target_channel.cpu().size(0))
            if two_side_rate < 0.9 and beta_idx >= 20:
                beta_list = beta_list + 0.2
                beta = beta + 0.2

            tmp[tmp == 0] = 1
            channel_inconsistency = 100.0 * np.sum(tmp != 1) / len(tmp)

            print("pruning rate (real/binary): {0:.4f}%/{1:.4f}%, index inconsistency: {2:.4f}%, two_side_rate: {3:.3f}".format(real_pruning_rate, binary_pruning_rate, channel_inconsistency, two_side_rate))
            print(ratio)
            x = binary_pruning_rate/100.0

            if 1-x < ratio:
                regulization = 0.0
            else:
                regulization = 5.0 * np.abs(x - 1 + ratio)

            channel_vector = []
            # regulization = 100.0 * np.abs(binary_pruning_rate/100.0 - 1 + args.compression_rate)
            sys.stdout.flush()
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch[{0}]: [{1}/{2}]\t'
                  'beta: {3:.4f}\t'
                  'reg_lambda: {4:.4f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), beta, regulization, batch_time=batch_time,
                   top1=top1, top5=top5))
            sys.stdout.flush()

    return channel_index, regulization


def validate(val_loader, model, old_model, criterion, channel_index):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.eval()

    end = time.time()

    for i, (inputs, targets) in enumerate(val_loader):
        input_var = torch.autograd.Variable(inputs).to(device)
        target_var = torch.autograd.Variable(targets).to(device)

        # 计算输出
        output, _ = model(input_var, None, 1.0, channel_index)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # 测时间
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
        correct_k = correct[:k].contiguous().reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == "__main__":
    main()
