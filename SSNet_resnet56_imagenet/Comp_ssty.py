import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import types
import os
from torchvision import models

device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
block_nums = [3,4,6,3]

def conv_mask_forward(self,x):
    return F.conv2d(x, self.weight * self.mask_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def linear_mask_forward(self,x):
    return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def init_sensity_model():
    model = models.resnet50(False)
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if 'downsample' not in name and 'layer' in name and 'conv3' not in name:
                layer.mask_weight = nn.Parameter(torch.ones_like(layer.weight))
                if isinstance(layer, nn.Conv2d):
                    layer.forward = types.MethodType(conv_mask_forward, layer)
                elif isinstance(layer, nn.Linear):
                    layer.forward = types.MethodType(linear_mask_forward, layer)
    return model

def Setting_Sensity(model, x, y, ratio, group_id, layersort, gpu_num):
    # model_ = model.module
    #  backpropagate to obtain gradient matrix
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if 'downsample' not in name and 'layer' in name and 'conv3' not in name:
                nn.init.constant_(layer.mask_weight,1.0)
                layer.weight.requires_grad = False
    # if x.shape[1] != y.shape[0]:
    #     return None,None
    x = x.cuda()
    model = model.cuda()
    model.zero_grad()
    outputs = model.forward(x)
    loss = F.nll_loss(outputs, y).cuda()
    loss.backward()

    # save gradient matrix
    grads_abs = []
    sum_channel_grads = []
    cur_grads_conv1 = []
    cur_grads_conv2 = []
    # cur_grads_conv3 = []
    for name, layer in model.named_modules():
        if 'downsample' not in name and 'layer' in name and 'conv3' not in name:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                grads_abs.append(torch.abs(layer.mask_weight.grad))
                if 'layer'+str(group_id+1) in name:
                    c,n,w,h = layer.mask_weight.shape
                    cur_mask_grad = torch.abs(layer.mask_weight.grad).reshape(n,c,w,h)
                    cur_mask_grad = torch.repeat_interleave(cur_mask_grad, repeats=gpu_num, dim=0)
                    if 'conv1' in name:
                        cur_grads_conv1.append(cur_mask_grad)
                    elif 'conv2' in name:
                        cur_grads_conv2.append(cur_mask_grad)

    # sum over gradient matrix for layers to be pruned
    block_sum_abs = []
    left = 2*sum(block_nums[:group_id])
    right = 2*sum(block_nums[:group_id+1])
    for i in range(len(grads_abs)):
        cur_layer_sum = []
        c,_,_,_ = grads_abs[i].shape
        cur = grads_abs[i]
        for j in range(c):
            x = torch.sum(cur[j,:,:,:]).item()
            sum_channel_grads.append(x)
            if i >= left and i < right:
                cur_layer_sum.append(x)
        if cur_layer_sum:
            block_sum_abs.append(cur_layer_sum)

    # global vs layer sort
    if layersort:
        sum_channel_grads = torch.Tensor(sum_channel_grads)
    else:
        sum_channel_grads = torch.Tensor(sum_channel_grads)
        numbers = len(sum_channel_grads)
        k = int(numbers*ratio)
        threshold, _ = torch.topk(sum_channel_grads, k, sorted=True)    
        # print(threshold)
        acceptable_score = threshold[-1]

    # save target strategy
    if layersort:
        target_channel = []
        for i in range(len(block_sum_abs)):
            # calcualte acceptable score for each layer
            numbers = len(block_sum_abs[i])
            k = int(numbers*ratio)
            grad_tensor = torch.Tensor(block_sum_abs[i])
            threshold, _ = torch.topk(grad_tensor, k, sorted=True)
            acceptable_score = threshold[-1]

            cur_channel_idx = []
            flag = True
            for l_c in block_sum_abs[i]:
                cur_channel_idx.append((l_c >= acceptable_score).float())
            if sum(cur_channel_idx) == 0:
                flag = False
                print(f'CATION: a channel is decided to be completely pruned')
            if not flag:
                cur_channel_idx = []
                k = int(len(block_sum_abs[i]) * ratio * 0.2)
                thresh, _ = torch.topk(torch.Tensor(block_sum_abs[i]), k, sorted=True)
                acc_score = thresh[-1]
                for l_c in block_sum_abs[i]:
                    cur_channel_idx.append((l_c >= acc_score).float())
                print(f'thete is {sum(cur_channel_idx)} channels left with thres {acc_score}')

            cur_channel_idx = torch.autograd.Variable(torch.Tensor(cur_channel_idx).squeeze()).to(device)
            target_channel.append(cur_channel_idx)
    else:
        target_channel = []
        for i in range(len(block_sum_abs)):
            cur_channel_idx = []
            flag = True
            for l_c in block_sum_abs[i]:
                cur_channel_idx.append((l_c >= acceptable_score).float())
            if sum(cur_channel_idx) == 0:
                flag = False
                print(f'CATION: a channel is decided to be completely pruned')
            if not flag:
                cur_channel_idx = []
                k = int(len(block_sum_abs[i]) * ratio * 0.2)
                thresh, _ = torch.topk(torch.Tensor(block_sum_abs[i]), k, sorted=True)
                acc_score = thresh[-1]
                for l_c in block_sum_abs[i]:
                    cur_channel_idx.append((l_c >= acc_score).float())
                print(f'thete is {sum(cur_channel_idx)} channels left with thres {acc_score}')

            cur_channel_idx = torch.autograd.Variable(torch.Tensor(cur_channel_idx).squeeze()).to(device)
            target_channel.append(cur_channel_idx)

    
    # normalize sensitivity
    # target_channel = torch.Tensor(target_channel)
    norm_factor = torch.sum(sum_channel_grads)
    for i in range(len(cur_grads_conv1)):
        cur_grads_conv1[i] = cur_grads_conv1[i]/norm_factor
    for i in range(len(cur_grads_conv2)):
        cur_grads_conv2[i] = cur_grads_conv2[i]/norm_factor
    # for i in range(len(cur_grads_conv3)):
    #     cur_grads_conv3[i] = cur_grads_conv3[i]/norm_factor
    # cur_grads_conv1 = cur_grads_conv1/norm_factor
    # cur_grads_conv2 = cur_grads_conv2/norm_factor
    # cur_grads_conv3 = cur_grads_conv3/norm_factor
    return [cur_grads_conv1, cur_grads_conv2], target_channel

