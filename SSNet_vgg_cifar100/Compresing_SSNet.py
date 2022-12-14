import torch
from torch import nn
import numpy as np
import os
import torch.nn.init as init
import sys
import SSNet


class SsNet_compressed(torch.nn.Module):
    def __init__(self, layer_id=0, model_path=None):
        torch.nn.Module.__init__(self)
        # print(model_path)
        model = torch.load(model_path + 'model.pth')
        # model = torch.load('results/models/model.pth')
        model_weight = model.state_dict()
        channel_index = torch.load(model_path + 'channel_index.pth')
        # print(channel_index.shape)
        channel_index = np.where(channel_index != 0)[0]
        new_num = int(channel_index.shape[0])
        channel_length = list()
        channel_length.append(3)
        for k, v in model_weight.items():
            if 'bias' in k and 'bn' not in k:
                channel_length.append(v.size()[0])
        channel_length[layer_id + 1] = new_num

        self.feature_1 = nn.Sequential()
        self.classifier = nn.Sequential()
        # add channel selection layers
        conv_names = {0: 'conv1_1', 1: 'conv1_2', 2: 'conv2_1', 3: 'conv2_2', 4: 'conv3_1', 5: 'conv3_2', 6: 'conv3_3',
                      7: 'conv4_1', 8: 'conv4_2', 9: 'conv4_3', 10: 'conv5_1', 11: 'conv5_2', 12: 'conv5_3'}
        relu_names = {0: 'relu1_1', 1: 'relu1_2', 2: 'relu2_1', 3: 'relu2_2', 4: 'relu3_1', 5: 'relu3_2', 6: 'relu3_3',
                      7: 'relu4_1', 8: 'relu4_2', 9: 'relu4_3', 10: 'relu5_1', 11: 'relu5_2', 12: 'relu5_3', 13: 'relu6', 14: 'relu7'}
        fc_names = {13: 'fc6', 14: 'fc7', 15: 'fc8'}
        bn_names = {0: 'bn1_1', 1: 'bn1_2', 2: 'bn2_1', 3: 'bn2_2', 4: 'bn3_1', 5: 'bn3_2', 6: 'bn3_3',
                      7: 'bn4_1', 8: 'bn4_2', 9: 'bn4_3', 10: 'bn5_1', 11: 'bn5_2', 12: 'bn5_3'}  
        pool_names = {1: 'pool1', 3: 'pool2', 6: 'pool3', 9: 'pool4', 12: 'pool5'}
        drop_names = {13: 'dropout6', 14: 'dropout7'}
        pooling_layer_id = [1, 3, 6, 9, 12]

        # add feature_1 and feature_2 layers
        print(channel_length)
        for i in range(13):
            self.feature_1.add_module(conv_names[i],nn.Conv2d(channel_length[i], channel_length[i + 1], kernel_size=3, stride=1,padding=1))
            self.feature_1.add_module(bn_names[i],nn.BatchNorm2d(channel_length[i + 1]))
            self.feature_1.add_module(relu_names[i], nn.ReLU(inplace=True))
            if i in pooling_layer_id:
                self.feature_1.add_module(pool_names[i], nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier.add_module(fc_names[13], nn.Linear(channel_length[13], channel_length[14]))
        self.classifier.add_module(relu_names[13], nn.ReLU(inplace=True))
        self.classifier.add_module(drop_names[13], nn.Dropout())
        self.classifier.add_module(fc_names[14], nn.Linear(channel_length[14], channel_length[15]))
        self.classifier.add_module(relu_names[14], nn.ReLU(inplace=True))
        self.classifier.add_module(drop_names[14], nn.Dropout())
        self.classifier.add_module(fc_names[15], nn.Linear(channel_length[15], channel_length[16]))

        # load pretrain model weights
        my_weight = self.state_dict()
        my_keys = list(my_weight.keys())
        channel_index = torch.cuda.LongTensor(channel_index)
        if layer_id < 12:
            #  conv1_1 to conv5_2
            for k, v in model_weight.items():
                name = k.split('.')
                if name[2] == conv_names[layer_id]:
                    if name[3] == 'weight':
                        name = 'feature_1.' + name[2] + '.' + name[3]
                        my_weight[name] = v[channel_index, :, :, :]
                    else:
                        name = 'feature_1.' + name[2] + '.' + name[3]
                        my_weight[name] = v[channel_index]
                elif name[2] == bn_names[layer_id]:
                    if 'num_batches_tracked' not in name:
                        name = 'feature_1.' + name[2] + '.' + name[3]
                        my_weight[name] = v[channel_index]
                elif name[2] == conv_names[layer_id + 1] or name[2] == bn_names[layer_id + 1]:
                    if name[3] == 'weight' and name[2] == conv_names[layer_id + 1]:
                        name = 'feature_1.' + name[2] + '.' + name[3]
                        my_weight[name] = v[:, channel_index, :, :]
                    else:
                        name = 'feature_1.' + name[2] + '.' + name[3]
                        my_weight[name] = v
                else:
                    if name[1] in ['feature_1', 'feature_2']:
                        name = 'feature_1.' + name[2] + '.' + name[3]
                    else:
                        name = name[1] + '.' + name[2] + '.' + name[3]
                    if name in my_keys:
                        my_weight[name] = v
        elif layer_id == 12:
            for k, v in model_weight.items():
                name = k.split('.')
                if name[2] == conv_names[layer_id]:
                    if name[3] == 'weight':
                        name = 'feature_1.' + name[2] + '.' + name[3]
                        my_weight[name] = v[channel_index, :, :, :]
                    else:
                        name = 'feature_1.' + name[2] + '.' + name[3]
                        my_weight[name] = v[channel_index]
                elif name[2] == bn_names[layer_id]:
                    if 'num_batches_tracked' not in name:
                        name = 'feature_1.' + name[2] + '.' + name[3]
                        my_weight[name] = v[channel_index]
                elif name[2] == 'fc6':
                    if name[3] == 'weight':
                        name = 'classifier.' + name[2] + '.' + name[3]
                        tmp = v.view(4096, 512, 1, 1)
                        tmp = tmp[:, channel_index, :, :]
                        my_weight[name] = tmp.view(4096, -1)
                    else:
                        name = 'classifier.' + name[2] + '.' + name[3]
                        my_weight[name] = v
                else:
                    name = 'feature_1.' + name[2] + '.' + name[3]
                    if name in my_keys:
                        my_weight[name] = v
        else:
            for k, v in model_weight.items():
                name = k.split('.')
                if name[2] == fc_names[layer_id]:
                    if name[3] == 'weight':
                        name = 'classifier.' + name[2] + '.' + name[3]
                        my_weight[name] = v[channel_index, :]
                    else:
                        name = 'classifier.' + name[2] + '.' + name[3]
                        my_weight[name] = v[channel_index]
                elif name[2] == fc_names[layer_id + 1]:
                    if name[3] == 'weight':
                        name = 'classifier.' + name[2] + '.' + name[3]
                        my_weight[name] = v[:, channel_index]
                    else:
                        name = 'classifier.' + name[2] + '.' + name[3]
                        my_weight[name] = v
                else:
                    name = k[7:]
                    if name in my_keys:
                        my_weight[name] = v

        self.load_state_dict(my_weight)

    def forward(self, x):
        x = self.feature_1(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SsNet_test(torch.nn.Module):
    def __init__(self, model_path):
        torch.nn.Module.__init__(self)
        model = torch.load(model_path)
        model_weight = model.state_dict()
        channel_length = list()
        channel_length.append(3)
        for k, v in model_weight.items():
            if 'bias' in k and 'bn' not in k:
                channel_length.append(v.size()[0])

        self.feature_1 = nn.Sequential()
        self.classifier = nn.Sequential()
        # add channel selection layers
        conv_names = {0: 'conv1_1', 1: 'conv1_2', 2: 'conv2_1', 3: 'conv2_2', 4: 'conv3_1', 5: 'conv3_2', 6: 'conv3_3',
                      7: 'conv4_1', 8: 'conv4_2', 9: 'conv4_3', 10: 'conv5_1', 11: 'conv5_2', 12: 'conv5_3'}
        relu_names = {0: 'relu1_1', 1: 'relu1_2', 2: 'relu2_1', 3: 'relu2_2', 4: 'relu3_1', 5: 'relu3_2', 6: 'relu3_3',
                      7: 'relu4_1', 8: 'relu4_2', 9: 'relu4_3', 10: 'relu5_1', 11: 'relu5_2', 12: 'relu5_3', 13: 'relu6', 14: 'relu7'}
        fc_names = {13: 'fc6', 14: 'fc7', 15: 'fc8'}
        bn_names = {0: 'bn1_1', 1: 'bn1_2', 2: 'bn2_1', 3: 'bn2_2', 4: 'bn3_1', 5: 'bn3_2', 6: 'bn3_3',
                      7: 'bn4_1', 8: 'bn4_2', 9: 'bn4_3', 10: 'bn5_1', 11: 'bn5_2', 12: 'bn5_3'}  
        pool_names = {1: 'pool1', 3: 'pool2', 6: 'pool3', 9: 'pool4', 12: 'pool5'}
        drop_names = {13: 'dropout6', 14: 'dropout7'}
        pooling_layer_id = [1, 3, 6, 9, 12]

        # add feature_1 and feature_2 layers
        for i in range(13):
            self.feature_1.add_module(conv_names[i],nn.Conv2d(channel_length[i], channel_length[i + 1], kernel_size=3, stride=1,padding=1))
            self.feature_1.add_module(bn_names[i],nn.BatchNorm2d(channel_length[i + 1]))
            self.feature_1.add_module(relu_names[i], nn.ReLU(inplace=True))
            if i in pooling_layer_id:
                self.feature_1.add_module(pool_names[i], nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier.add_module(fc_names[13], nn.Linear(channel_length[13], channel_length[14]))
        self.classifier.add_module(relu_names[13], nn.ReLU(inplace=True))
        self.classifier.add_module(drop_names[13], nn.Dropout())
        self.classifier.add_module(fc_names[14], nn.Linear(channel_length[14], channel_length[15]))
        self.classifier.add_module(relu_names[14], nn.ReLU(inplace=True))
        self.classifier.add_module(drop_names[14], nn.Dropout())
        self.classifier.add_module(fc_names[15], nn.Linear(channel_length[15], channel_length[16]))

        # load pretrain model weights
        my_weight = self.state_dict()
        my_keys = list(my_weight.keys())
        for k, v in model_weight.items():
            name = k.split('.')
            name = name[1] + '.' + name[2] + '.' + name[3]
            # name = k
            if name in my_keys:
                my_weight[name] = v
            else:
                print('error')
                os.exit(0)
        self.load_state_dict(my_weight)

    def forward(self, x):
        x = self.feature_1(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class vgg16_GAP(torch.nn.Module):
    def __init__(self, model_path):
        torch.nn.Module.__init__(self)
        model= torch.load(model_path)
        model_weight = model.state_dict()
        channel_length = list()
        channel_length.append(3)
        for k, v in model_weight.items():
            if 'bias' in k and 'bn' not in k:
                channel_length.append(v.size()[0])

        self.feature_1 = nn.Sequential()
        self.classifier = nn.Sequential()
        # add channel selection layers
        conv_names = {0: 'conv1_1', 1: 'conv1_2', 2: 'conv2_1', 3: 'conv2_2', 4: 'conv3_1', 5: 'conv3_2', 6: 'conv3_3',
                      7: 'conv4_1', 8: 'conv4_2', 9: 'conv4_3', 10: 'conv5_1', 11: 'conv5_2', 12: 'conv5_3'}
        relu_names = {0: 'relu1_1', 1: 'relu1_2', 2: 'relu2_1', 3: 'relu2_2', 4: 'relu3_1', 5: 'relu3_2', 6: 'relu3_3',
                      7: 'relu4_1', 8: 'relu4_2', 9: 'relu4_3', 10: 'relu5_1', 11: 'relu5_2', 12: 'relu5_3'}
        bn_names = {0: 'bn1_1', 1: 'bn1_2', 2: 'bn2_1', 3: 'bn2_2', 4: 'bn3_1', 5: 'bn3_2', 6: 'bn3_3',
                      7: 'bn4_1', 8: 'bn4_2', 9: 'bn4_3', 10: 'bn5_1', 11: 'bn5_2', 12: 'bn5_3'}  
        pool_names = {1: 'pool1', 3: 'pool2', 6: 'pool3', 9: 'pool4', 12: 'pool5'}
        pooling_layer_id = [1, 3, 6, 9]

        # add feature_1 and feature_2 layers
        for i in range(13):
            self.feature_1.add_module(conv_names[i],nn.Conv2d(channel_length[i], channel_length[i + 1], kernel_size=3, stride=1,padding=1))
            self.feature_1.add_module(bn_names[i],nn.BatchNorm2d(channel_length[i + 1]))
            self.feature_1.add_module(relu_names[i], nn.ReLU(inplace=True))
            if i in pooling_layer_id:
                self.feature_1.add_module(pool_names[i], nn.MaxPool2d(kernel_size=2, stride=2))
            if i == 12:
                self.feature_1.add_module(pool_names[i], nn.AvgPool2d(kernel_size=14, stride=1))

        # add classifier
        self.classifier.add_module('fc', nn.Linear(channel_length[13], channel_length[14]))

        init.xavier_uniform(self.classifier.fc.weight, gain=np.sqrt(2.0))
        init.constant(self.classifier.fc.bias, 0)

        # load pretrain model weights
        my_weight = self.state_dict()
        my_keys = list(my_weight.keys())
        for k, v in model_weight.items():
            name = k.split('.')
            name = name[1] + '.' + name[2] + '.' + name[3]
            if name in my_keys:
                my_weight[name] = v
            else:
                print(name)
        self.load_state_dict(my_weight)

    def forward(self, x):
        x = self.feature_1(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = vgg16_GAP('../checkpoint/fine_tune/model.pth')
    print(model)
