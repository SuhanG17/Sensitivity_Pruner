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

class Use_Mask_Conv(torch.autograd.Function):
    '''
    input: x: 64*512*7*7, scale:512 ==> x[:, i, :, :]*scale[i]
    '''
    @staticmethod
    def forward(self, input_data, mask_vector):
        self.save_for_backward(input_data, mask_vector)
        # my_weight = self.state_dict()
        input_data2 = input_data.clone()
        for i in range(mask_vector.shape[0]):
            input_data2[:, i, :, :] = input_data[:, i, :, :] * mask_vector[i]
        return input_data2

    @staticmethod
    def backward(self, grad_output):
        input_data, mask_vector = self.saved_tensors
        grad_input = input_data.clone()
        for i in range(mask_vector.shape[0]):
            grad_input[:, i, :, :] = grad_output.data[:, i, :, :] * mask_vector[i]

        grad_vec = mask_vector.clone()
        for i in range(mask_vector.shape[0]):
            grad_vec[i] = torch.sum(grad_output.data[:, i, :, :]*input_data[:, i, :, :])

        return Variable(grad_input), Variable(grad_vec)

class Use_Mask_FC(torch.autograd.Function):
    '''
    input: x: 64*512*7*7, scale:512 ==> x[:, i, :, :]*scale[i]
    '''
    @staticmethod
    def forward(self, input_data, mask_vector):
        self.save_for_backward(input_data, mask_vector)
        input_data2 = input_data.clone()
        for i in range(mask_vector.shape[0]):
            input_data2[:, i] = input_data[:, i] * mask_vector[i]
        return input_data2

    @staticmethod
    def backward(self, grad_output):
        input_data, mask_vector = self.saved_tensors
        grad_input = input_data.clone()
        for i in range(mask_vector.shape[0]):
            grad_input[:, i] = grad_output.data[:, i] * mask_vector[i]

        grad_vec = mask_vector.clone()
        for i in range(mask_vector.shape[0]):
            grad_vec[i] = torch.sum(grad_output.data[:, i]*input_data[:, i])

        return Variable(grad_input), Variable(grad_vec)


class Use_Sensity(nn.Module):
    def __init__(self,layer_id,channel_num):
        super(Use_Sensity,self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.layer_id = layer_id
        self.conv = nn.Conv2d(channel_num,channel_num,kernel_size=3,stride=1, padding=0)
        # self.conv2 = nn.Conv2d(channel_num, channel_num, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(4096,4096,bias=True)

        n = 3 * 3 * channel_num
        self.conv.weight.data.normal_(0, 10*math.sqrt(2.0/n))
        self.fc.weight.data.normal_(mean=0,std=0.01)


    def forward(self, x, ssty, beta, channel_index=None):
        if self.training:
            ssty = torch.sum(ssty,dim=0).cuda().unsqueeze(dim=0)
            if self.layer_id < 13:
                new_mask = self.conv(ssty)
            else:
                new_mask = self.fc(ssty)
            new_mask = torch.squeeze(new_mask)
            new_mask = new_mask * beta
            new_mask = self.sigmoid(new_mask).clone()

        else:
            new_mask = torch.randn(10)
            new_mask.data = torch.FloatTensor(channel_index.mean(0).squeeze().cpu()).cuda()
        if self.layer_id < 13:
            x = Use_Mask_Conv.apply(x, new_mask)
        else:
            x = Use_Mask_FC.apply(x, new_mask)
        new_mask.cuda()

        return x, new_mask

    
    
    







