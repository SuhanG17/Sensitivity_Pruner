import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import sys
import argparse
import numpy as np

import SSNet_mobilenetv2
import Comp_ssty

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--num_epochs', default=4, type=int, help='number of training epochs')
parser.add_argument('--lr_decay_epoch', default=10, type=int, help='learning rate decay epoch')
parser.add_argument('--train_path', default='/nas/guosuhan/auto-prune/data/imagenet64/prune/train_loader', type=str, help='path to train loader')
parser.add_argument('--val_path', default='/nas/guosuhan/auto-prune/data/imagenet64/prune/val_loader', type=str, help='path to val loader')
parser.add_argument('--model_dir_path', default='/nas/guosuhan/auto-prune/logs/imagenet/results_4/model',
                    type=str, help='the path of model folder')
parser.add_argument('--log_dir_path', default='/nas/guosuhan/auto-prune/logs/imagenet/results_4/logs',
                    type=str, help='the path of logs folder')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--block_id', default=1, type=int, help='block id to be pruned, range from 1 to 7(inclusive)')
parser.add_argument('--compression_rate', default=0.4, type=float, help='the percentage of 1 in compressed model')
parser.add_argument('--layersort', action='store_true', help='if not speficied, global sort will be used')
parser.add_argument('--channel_index_range', default=20, type=int, help='the range to calculate channel index')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--beta_range', default=100, type=int, help='the range to calculate channel index')
args = parser.parse_args()
print(args)

best_prec1 = -1
channel_index_list = None
beta_index = 0
beta_list = None


block_channel_num = [0, 2, 3, 4, 3, 3, 1, 1] # first 0 is a dummy for indexing
num_layers_to_be_pruned =block_channel_num[args.block_id]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Pruning layer num: {num_layers_to_be_pruned}' )

def main():
    global args, best_prec1, channel_index_list, beta_list, num_layers_to_be_pruned
    # Phase 1 : Data Upload
    print('Data Preparing...')
    train_loader = torch.load(args.train_path)
    val_loader = torch.load(args.val_path)
    print('Data Load Success!')

    # Phase 2 : Model setup
    print('Model setup')
    if args.block_id == 1:
        # save model as backup
        model_ft = SSNet_mobilenetv2.mobile_v2(args.model_dir_path+'/pretrained_mobilev2.pth').to(device)
        torch.save(model_ft, args.model_dir_path+'/model_1.pth')
        # save model as checkpoint
        model_ft = torch.nn.DataParallel(model_ft).to(device)
        torch.save(model_ft, args.model_dir_path+'/model.pth') 
    # initiate the pruning model from backup
    if args.block_id == 1: # nothing to load for first block to be pruned
        channel_dim = None 
    else: # load channel_dim from last pruned block
        channel_dim = torch.load(args.log_dir_path + '/block_' + str(args.block_id-1) + '/channel_dim.pth')
    print(f'current channel dim is {channel_dim}')
    model_ft = SSNet_mobilenetv2.mobile_v2_ssnet(args.block_id, args.model_dir_path+'/model.pth', channel_dim).to(device)
    model_ft = torch.nn.DataParallel(model_ft)
    cudnn.benchmark = True
    # initiate the sensitivity model from backup
    org_model = Comp_ssty.init_sensity_model(args.model_dir_path+'/model_1.pth') # always use the original full model, do not dataparallel
    print("Model setup success!")

    # Phase 3: fine_tune model
    print('Model finetuning...')
    # define loss function (criterion) and optimizer
    criterion1 = nn.CrossEntropyLoss().to(device)
    criterion2 = nn.L1Loss().to(device)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    # initate beta
    # tmp = torch.linspace(1, 100, steps=int(args.num_epochs * len(train_loader) / args.beta_range))
    # tmp = torch.linspace(0.1, 10, steps=int(args.num_epochs * len(train_loader) / args.beta_range)) 
    tmp = torch.linspace(0.05, 5, steps=int(args.num_epochs * len(train_loader) / args.beta_range)) 
    beta_list = torch.ones([num_layers_to_be_pruned, len(tmp)])
    for tmp_i in range(num_layers_to_be_pruned):
        beta_list[tmp_i, :] = tmp
    regulization = 0.1 * torch.ones(num_layers_to_be_pruned)

    for epoch in range(args.start_epoch, args.num_epochs):
        adjust_learning_rate(optimizer, epoch, int(args.num_epochs/2.0))
        # train for one epoch
        channel_index = train(train_loader, model_ft, org_model, criterion1, criterion2, optimizer, epoch, regulization)
        prec1 = validate(val_loader, model_ft, criterion1, channel_index)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
            folder_path = args.model_dir_path + '/block_' + str(args.block_id)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            torch.save(model_ft, folder_path+'/model.pth')
            torch.save(channel_index, folder_path+'/channel_index.pth')

def train(train_loader, ft_model, org_model, criterion1, criterion2, optimizer, epoch, regulization):
    global beta_list, beta_index, num_layers_to_be_pruned

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

        input_var = input.to(device)
        target_var = target.to(device)

        # compute output
        ssty, target_channel = Comp_ssty.Setting_Sensity(org_model, input_var, target_var, args.compression_rate, args.block_id, args.layersort, gpu_num, device)
        output, mask_vec = ft_model(input_var, ssty)

        loss = criterion1(output, target_var)
        ## reshape mask_vec for multi-gpu training
        if gpu_num > 1:
            for idx in range(len(mask_vec)):
                mask_vec[idx] = mask_vec[idx].view(gpu_num, -1).mean(0)
                
        for idx in range(len(mask_vec)):
            loss += regulization[idx] * criterion2(mask_vec[idx], target_channel[idx])
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute channel index
        channel_vector = list()
        for j in range(num_layers_to_be_pruned):
            tmp = mask_vec[j].to('cpu').detach().numpy() # shape of [output channel]
            channel_vector.append(tmp.copy())
            if i == 0:
                print('first 5 values in layer {0}: [{1:.6f}, {2:.6f}, {3:.6f}, {4:.6f}, {5:.6f}]'.format(j, tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]))
        
        channel_index_list.append(channel_vector.copy())

        if len(channel_index_list) == args.channel_index_range:
            channel_index_list = np.array(channel_index_list)
            channel_index_binary = list()
            for j in range(num_layers_to_be_pruned):
                tmp1 = channel_index_list[args.channel_index_range-1][j]
                cur_two_side_rate = (np.sum(tmp1 > 0.8) + np.sum(tmp1 < 0.2)) / len(tmp1)
                print('first 5 values in layer {0}: [{1:.6f}, {2:.6f}, {3:.6f}, {4:.6f}, {5:.6f}]'.format(j, tmp1[0], tmp1[1], tmp1[2], tmp1[3], tmp1[4]))
            
                
                ##  binarizied channel_index, averaged over args.channel_index_range 
                tmp2 = np.array(channel_index_list[:, j]).sum(axis=0)
                tmp2 = tmp2 / args.channel_index_range
                channel_index = (np.sign(tmp2 - 0.5) + 1) / 2.0  # to 0-1 binary
                channel_index_binary.append(channel_index.copy())

                ## pruning rate
                tmp3 = np.array(channel_index_list[:, j]).sum(axis=0)
                tmp3 = tmp3 / args.channel_index_range
                cur_real_pruning_rate = 100.0 * np.sum(tmp3 < 10**-6) / len(tmp3)
                cur_binary_pruning_rate = 100.0 * np.sum(channel_index < 10**-6) / len(channel_index)
                
                ## inconsistency
                target_stratety = target_channel[j].cpu() 
                inconsistency = channel_index[target_stratety == 0] == 1
                if len(inconsistency) > 0:
                    cur_channel_inconsistency = 100.0 * np.sum(inconsistency) / len(inconsistency)
                else:
                    cur_channel_inconsistency = 0.

                ## target ratio (percentage kept)
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

                # sys.stdout.flush()
            channel_index_list = list()

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

        input_var = input.to(device)
        target_var = target.to(device)

        # compute output
        channel_index_ = list()
        for idx in range(len(channel_index)):
            tmp = torch.repeat_interleave(torch.Tensor(channel_index[idx]).unsqueeze(dim=0), repeats=input_var.shape[0], dim = 0)
            tmp = torch.autograd.Variable(tmp).to(device)
            channel_index_.append(tmp)
        output, _ = model(input_var, None, channel_index_) 
        # if dataparallel, cannot retrieve channel_index using model.module.features._modules['2'].channel_index
        # if single gpu or cpu, can do using model.features._modules['2'].channel_index
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
