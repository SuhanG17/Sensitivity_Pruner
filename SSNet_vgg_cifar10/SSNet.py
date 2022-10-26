import torch
from torch import nn
import Sensity_Filter
import copy

class Vgg16(torch.nn.Module):
    def __init__(self, model_path):
        torch.nn.Module.__init__(self)
        self.feature_1 = nn.Sequential()
        self.classifier = nn.Sequential()

        # add feature layers
        self.feature_1.add_module('conv1_1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('bn1_1',nn.BatchNorm2d(64))
        self.feature_1.add_module('relu1_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv1_2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('bn1_2',nn.BatchNorm2d(64))
        self.feature_1.add_module('relu1_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_1.add_module('conv2_1', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('bn2_1',nn.BatchNorm2d(128))
        self.feature_1.add_module('relu2_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv2_2', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('bn2_2',nn.BatchNorm2d(128))
        self.feature_1.add_module('relu2_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_1.add_module('conv3_1', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('bn3_1',nn.BatchNorm2d(256))
        self.feature_1.add_module('relu3_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv3_2', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('bn3_2',nn.BatchNorm2d(256))
        self.feature_1.add_module('relu3_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv3_3', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('bn3_3',nn.BatchNorm2d(256))
        self.feature_1.add_module('relu3_3', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool3', nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_1.add_module('conv4_1', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('bn4_1',nn.BatchNorm2d(512))
        self.feature_1.add_module('relu4_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv4_2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('bn4_2',nn.BatchNorm2d(512))
        self.feature_1.add_module('relu4_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv4_3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('bn4_3',nn.BatchNorm2d(512))
        self.feature_1.add_module('relu4_3', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool4', nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_1.add_module('conv5_1', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('bn5_1',nn.BatchNorm2d(512))
        self.feature_1.add_module('relu5_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv5_2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('bn5_2',nn.BatchNorm2d(512))
        self.feature_1.add_module('relu5_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv5_3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('bn5_3',nn.BatchNorm2d(512))
        self.feature_1.add_module('relu5_3', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool5', nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier.add_module('fc6', nn.Linear(512, 10))

        model = torch.load(model_path)
        model_weight = model.state_dict()
        my_weight = self.state_dict()
        my_keys = list(my_weight.keys())
        count = 0
        for k, v in model_weight.items():
            my_weight[my_keys[count]] = v
            count += 1
        self.load_state_dict(my_weight)

    def forward(self,x):
        x = self.feature_1(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)

        return x


class SSnet(torch.nn.Module):
    def __init__(self, layer_id=0):
        torch.nn.Module.__init__(self)
        model_path = 'results/models/model.pth'
        model = torch.load(model_path)
        model_weight = model.state_dict()
        self.model_path = model_path
        channel_len = [3]
        for k,v in model_weight.items():
            if 'bias' in k and 'bn' not in k:
                channel_len.append(v.size()[0])
        self.layer_id = layer_id
        self.feature_1 = nn.Sequential()
        self.feature_2 = nn.Sequential()
        self.classifier = nn.Sequential()

        conv_names = {0: 'conv1_1', 1: 'conv1_2', 2: 'conv2_1', 3: 'conv2_2', 4: 'conv3_1', 5: 'conv3_2', 6: 'conv3_3',
                      7: 'conv4_1', 8: 'conv4_2', 9: 'conv4_3', 10: 'conv5_1', 11: 'conv5_2', 12: 'conv5_3'}
        relu_names = {0: 'relu1_1', 1: 'relu1_2', 2: 'relu2_1', 3: 'relu2_2', 4: 'relu3_1', 5: 'relu3_2', 6: 'relu3_3',
                      7: 'relu4_1', 8: 'relu4_2', 9: 'relu4_3', 10: 'relu5_1', 11: 'relu5_2', 12: 'relu5_3'}
        bn_names = {0: 'bn1_1', 1: 'bn1_2', 2: 'bn2_1', 3: 'bn2_2', 4: 'bn3_1', 5: 'bn3_2', 6: 'bn3_3',
                      7: 'bn4_1', 8: 'bn4_2', 9: 'bn4_3', 10: 'bn5_1', 11: 'bn5_2', 12: 'bn5_3'}  
        pool_names = {1: 'pool1', 3: 'pool2', 6: 'pool3', 9: 'pool4', 12: 'pool5'}
        pooling_layer_id = [1, 3, 6, 9, 12]
        pic_size = {0: 32, 1: 32, 2: 16, 3: 16, 4: 8, 5: 8, 6: 8, 7: 4, 8: 4, 9: 4, 10: 2, 11: 2, 12: 2}
        print(channel_len)
        self.Sensity_Filter = Sensity_Filter.Use_Sensity(layer_id,channel_len[layer_id+1])
        for layer in range(13):
            if layer <= layer_id:
                self.feature_1.add_module(conv_names[layer],nn.Conv2d(channel_len[layer],channel_len[layer+1],kernel_size=3,stride=1,padding=1))
                self.feature_1.add_module(bn_names[layer],nn.BatchNorm2d(channel_len[layer+1]))
                self.feature_1.add_module(relu_names[layer],nn.ReLU(inplace=True))
                if layer in pooling_layer_id:
                    if layer == layer_id:
                        self.feature_2.add_module(pool_names[layer],nn.MaxPool2d(kernel_size=2,stride=2))
                    else:
                        self.feature_1.add_module(pool_names[layer],nn.MaxPool2d(kernel_size=2,stride=2))
            else:
                self.feature_2.add_module(conv_names[layer],nn.Conv2d(channel_len[layer],channel_len[layer+1],kernel_size=3,stride=1,padding=1))
                self.feature_2.add_module(bn_names[layer],nn.BatchNorm2d(channel_len[layer+1]))
                self.feature_2.add_module(relu_names[layer],nn.ReLU(inplace=True))
                if layer in pooling_layer_id:
                    self.feature_2.add_module(pool_names[layer],nn.MaxPool2d(kernel_size=2,stride=2))
        self.classifier.add_module('fc6', nn.Linear(channel_len[13], channel_len[14]))

        my_weight = self.state_dict()
        my_keys = list(my_weight.keys())
        for k, v in model_weight.items():
            # print(k)
            name = k.split('.')
            name = 'feature_1.'+name[2]+'.'+name[3]
            if name in my_keys:
                my_weight[name] = v

            name = k.split('.')
            name = 'feature_2.' + name[2] + '.' + name[3]
            if name in my_keys:
                my_weight[name] = v

            name = k[7:]
            if name in my_keys:
                my_weight[name] = v
        self.load_state_dict(my_weight)
    

    def forward(self, x, ssty, beta=0.1, channel_index=None):
        x = self.feature_1(x)
        x, scale_vector= self.Sensity_Filter(x, ssty, beta, channel_index)
        x = self.feature_2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, scale_vector
