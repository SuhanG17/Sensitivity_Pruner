import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import gradcheck
import copy
import types
import os
import math
import copy

def forward_conv2d(self,x):
    x = x.to('cuda:0')
    return F.conv2d(x,self.weight * self.mask_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def forward_linear(self,x):
    x = x.to('cuda:0')
    return F.linear(x,self.weight * self.mask_weight, self.bias)

def init_sensity_model():
    model = torch.load('results/models/model_1.pth')
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.mask_weight = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(forward_conv2d, layer)
        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(forward_linear, layer)

    return model

def Setting_Sensity(model, x, y, ratio, layer_id):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.constant_(layer.mask_weight,1.0)
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False
    
    x = x.cuda()
    model = model.cuda()
    model.zero_grad()
    outputs = model.forward(x)
    loss = F.nll_loss(outputs, y)
    loss.backward()

    grads_abs = []
    sum_channel_grads = []
    for layer in model.modules():
        if (isinstance(layer, nn.Conv2d) and layer_id < 13) or (isinstance(layer, nn.Linear) and layer_id > 12):
            grads_abs.append(torch.abs(layer.mask_weight.grad))
    if layer_id > 12:
        cur_grads = torch.Tensor(grads_abs[layer_id-13].cpu())
    else:
        cur_grads = torch.Tensor(grads_abs[layer_id].cpu())

    for i in range(len(grads_abs)):
        if len(grads_abs[i].shape) > 2:
            c,_,_,_ = grads_abs[i].shape
            cur = grads_abs[i]
            for j in range(c):
                sum_channel_grads.append(torch.sum(cur[j,:,:,:]).item())
        else:
            c,_ = grads_abs[i].shape
            cur = grads_abs[i]
            for j in range(c):
                sum_channel_grads.append(torch.sum(cur[j,:]).item())
    sum_channel_grads = torch.Tensor(sum_channel_grads)
    numbers = len(sum_channel_grads)
    k = int(numbers*ratio)
    threshold, _ = torch.topk(sum_channel_grads, k, sorted=True)

    ch_num = cur_grads.shape[0]
    layer_channel_ssty = []
    for i in range(ch_num):
        if len(cur_grads.shape)>2:
            layer_channel_ssty.append(torch.sum(cur_grads[i,:,:,:]).item())
        else:
            layer_channel_ssty.append(torch.sum(cur_grads[i,:]).item())

    layer_channel_ssty = torch.Tensor(layer_channel_ssty)
    acceptable_score = threshold[-1]
    target_channel = []
    flag = True
    idxs = 0
    for l_c in layer_channel_ssty:
        idx = l_c >= acceptable_score
        target_channel.append(idx.float())
        idxs += idx
    if idxs >= ch_num*ratio*0.2:
        flag = False
    if flag:
        k = int(ch_num*ratio*0.2)
        threshold, _ = torch.topk(layer_channel_ssty, k, sorted=True)
        acceptable_score = threshold[-1]
        for l_c in range(layer_channel_ssty.shape[0]):
            target_channel[l_c] = (layer_channel_ssty[l_c] >= acceptable_score).float()


    target_channel = torch.Tensor(target_channel)
    norm_factor = torch.sum(cur_grads)
    ssty = (cur_grads/norm_factor)
    ssty = torch.sum(ssty, dim=1).unsqueeze(dim=0)
    ssty = torch.repeat_interleave(ssty, repeats=x.shape[0], dim=0)
    return ssty, target_channel