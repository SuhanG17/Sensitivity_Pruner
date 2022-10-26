# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
import torch.nn as nn
import torch.nn.functional as F
import types

def conv_mask_forward(self,x):
    return F.conv2d(x, self.weight * self.mask_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def init_sensity_model(model_path):
    # has to be model, not state_dict()
    model = torch.load(model_path)
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d): 
            nn.init.xavier_normal_(layer.weight)
            # conv layer in features
            if 'features' and 'conv1' in name:
                if int(name.split('.')[1] )!= 1:
                    layer.mask_weight = nn.Parameter(torch.ones_like(layer.weight))
                    layer.forward = types.MethodType(conv_mask_forward, layer)
            # last conv layer, not in features module
            if 'conv.0' in name:
                layer.mask_weight = nn.Parameter(torch.ones_like(layer.weight))
                layer.forward = types.MethodType(conv_mask_forward, layer)
    return model

def Setting_Sensity(model, x, y, ratio, block_to_be_pruned, layersort, gpu_num, device):
    # backpropagate to obtain gradient matrix
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_normal_(layer.weight)
            # conv layer in features
            if 'features' and 'conv1' in name:
                if int(name.split('.')[1] )!= 1:  # calculate sensitivity and target for first conv layer only
                    nn.init.constant_(layer.mask_weight,1.0)
                    layer.weight.requires_grad = False
            # last conv layer, not in features module
            if 'conv.0' in name:
                nn.init.constant_(layer.mask_weight,1.0)
                layer.weight.requires_grad = False 

    x = x.to(device)
    model = model.to(device)
    model.zero_grad()
    outputs = model.forward(x)
    loss = F.nll_loss(outputs, y)
    loss.backward()


    # save gradient matrix
    grads_abs = [] # record gradient of all mask_weight
    cur_grads_conv = [] # record gradient all mask_weight but reshaped and repeated if multiple gpu
    if block_to_be_pruned != 7: # prune IR block in features
        item_range = model.block_dict[block_to_be_pruned] # indexing refer to SSNet_mobilenetv2 
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) and (('features' in name and int(name.split('.')[1]) > 1 and 'conv1' in name) or 'conv.0' in name):
            grads_abs.append(torch.abs(layer.mask_weight.grad))
            if block_to_be_pruned == 7:
                if 'conv.0' in name:
                    out, inp, w, h = layer.mask_weight.shape
                    cur_mask_grad = torch.abs(layer.mask_weight.grad).reshape(inp, out, w, h) 
                    cur_mask_grad = torch.repeat_interleave(cur_mask_grad, repeats=gpu_num, dim=0) 
                    cur_grads_conv.append(cur_mask_grad)
            else:
                if int(name.split('.')[1]) >= item_range[0] and int(name.split('.')[1]) < item_range[1]: 
                    out, inp, w, h = layer.mask_weight.shape
                    cur_mask_grad = torch.abs(layer.mask_weight.grad).reshape(inp, out, w, h) 
                    cur_mask_grad = torch.repeat_interleave(cur_mask_grad, repeats=gpu_num, dim=0) 
                    cur_grads_conv.append(cur_mask_grad)

    # sum over gradient matrix for layers to be pruned
    sum_channel_grads = [] # record all sum of mask gradient into a single list
    block_sum_abs = [] # record sum of mask gradient into list of list, each sub-list indicate the mask for a block of layers
    if block_to_be_pruned == 7:
        start = model.block_dict[6][1] - model.block_dict[1][0] # re-index from the last feature layer, model.block_dict[1][0]=2
        end = start + 1 
    else:
        start = model.block_dict[block_to_be_pruned][0] - model.block_dict[1][0] # index in grads_abs restarts from zero, hence modify
        end = model.block_dict[block_to_be_pruned][1] - model.block_dict[1][0] 
    for i in range(len(grads_abs)):
        cur_layer_sum = []
        c, _, _, _ = grads_abs[i].shape # out, inp, w, h 
        cur = grads_abs[i]
        for j in range(c):
            x = torch.sum(cur[j,:,:,:]).item() # sum the output channel (to be pruned)
            sum_channel_grads.append(x)
            if i >= start and i < end: # record the output channel sum for current block
                cur_layer_sum.append(x)
        if cur_layer_sum:
            block_sum_abs.append(cur_layer_sum)
    
    # global vs layer sort
    if layersort:
        sum_channel_grads = torch.Tensor(sum_channel_grads)
    else:
        sum_channel_grads = torch.Tensor(sum_channel_grads)
        numbers = len(sum_channel_grads)  # total number of channels
        k = int(numbers*ratio) # number of channels to stay, rest to be pruned
        threshold, _ = torch.topk(sum_channel_grads, k, sorted=True)    
        acceptable_score = threshold[-1]

    # save target strategy
    if layersort:
        target_channel = [] # list of tensors, each tensor repr the ground-truth mask of a layer in the designated block
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
            if not flag: # if a layer's channels are all pruned, save 0.2 of the ratio instead
                cur_channel_idx = []
                k = int(len(block_sum_abs[i]) * ratio * 0.2) 
                thresh, _ = torch.topk(torch.Tensor(block_sum_abs[i]), k, sorted=True)
                acc_score = thresh[-1]
                for l_c in block_sum_abs[i]:
                    cur_channel_idx.append((l_c >= acc_score).float()) 
            cur_channel_idx = torch.autograd.Variable(torch.Tensor(cur_channel_idx).squeeze()).to(device)
            target_channel.append(cur_channel_idx)
    else:
        target_channel = [] # list of tensors, each tensor repr the ground-truth mask of a layer in the designated block
        for i in range(len(block_sum_abs)):
            cur_channel_idx = []
            flag = True
            for l_c in block_sum_abs[i]:
                cur_channel_idx.append((l_c >= acceptable_score).float())
            if sum(cur_channel_idx) == 0:
                flag = False
            if not flag: # if a layer's channels are all pruned, save 0.2 of the ratio instead
                cur_channel_idx = []
                k = int(len(block_sum_abs[i]) * ratio * 0.2) 
                thresh, _ = torch.topk(torch.Tensor(block_sum_abs[i]), k, sorted=True)
                acc_score = thresh[-1]
                for l_c in block_sum_abs[i]:
                    cur_channel_idx.append((l_c >= acc_score).float()) 
            cur_channel_idx = torch.autograd.Variable(torch.Tensor(cur_channel_idx).squeeze()).to(device)
            target_channel.append(cur_channel_idx)
    
    # normalize sensitivity
    norm_factor = torch.sum(sum_channel_grads)
    for i in range(len(cur_grads_conv)):
        cur_grads_conv[i] = cur_grads_conv[i]/norm_factor
    
    return cur_grads_conv, target_channel # sensitivity to be trained, ground truth
