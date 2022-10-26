import torch.nn as nn
import torch
import math
import Sensity_Filter
# from torchvision import models
from torchvision import models
ssty = None

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, number_list, stride=1, downsample=None):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(number_list[1], number_list[0], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(number_list[0])
        self.conv2 = nn.Conv2d(number_list[3], number_list[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(number_list[2])
        self.conv3 = nn.Conv2d(number_list[5], number_list[4], kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(number_list[4])
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck_Sensity(nn.Module):
    expansion = 4

    def __init__(self, number_list, channel_num, op_id=0, stride=1, downsample=None):
        super(Bottleneck_Sensity, self).__init__()

        global ssty
        self.conv1 = nn.Conv2d(number_list[1], number_list[0], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(number_list[0])
        self.conv2 = nn.Conv2d(number_list[3], number_list[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(number_list[2])
        self.conv3 = nn.Conv2d(number_list[5], number_list[4], kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(number_list[4])
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.op_id = op_id
        self.channel_num = channel_num

        self.channel_index1 = None
        self.channel_index2 = None

        self.Sensity_Filter_1 = Sensity_Filter.Use_Sensity(1, channel_num[self.op_id * 2])
        self.Sensity_Filter_2 = Sensity_Filter.Use_Sensity(3, channel_num[self.op_id * 2 + 1])

        self.beta1 = 1.0
        self.beta2 = 1.0

        self.vec1 = None
        self.vec2 = None

    def forward(self, x):
        global ssty
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.training:
            out, self.vec1 = self.Sensity_Filter_1(out, ssty[0][self.op_id], self.beta1)
        else:
            out, self.vec1 = self.Sensity_Filter_1(out, None, self.beta1, self.channel_index1)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.training:
            out, self.vec2 = self.Sensity_Filter_2(out, ssty[1][self.op_id], self.beta2)
        else:
            out, self.vec2 = self.Sensity_Filter_2(out, None, self.beta2, self.channel_index2)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
            
        return out

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=1000):
        super().__init__()

        # old_model = models.resnet50(True)
        old_model = torch.load('results/models/resnet50_pretrained.pth')
        old_weight = old_model.state_dict()
        channel_number_list = analyse_number(old_weight)
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(channel_number_list[0], block, 64, num_block[0], 1)
        self.layer2 = self._make_layer(channel_number_list[1], block, 128, num_block[1], 2)
        self.layer3 = self._make_layer(channel_number_list[2], block, 256, num_block[2], 2)
        self.layer4 = self._make_layer(channel_number_list[3], block, 512, num_block[3], 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        my_weight = self.state_dict()
        for k, v in old_weight.items():
            my_weight[k] = v
        self.load_state_dict(my_weight)

    def _make_layer(self, number_list, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            
        layers = []
        layers.append(block(number_list[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(number_list[i]))

        return nn.Sequential(*layers)

    def forward(self, x, ssty=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)  
        x = self.layer3(x)  
        x = self.layer4(x)  

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



class ResNet_SSNet(nn.Module):
    def __init__(self, group_id, block, layers, num_classes=1000):
        super(ResNet_SSNet, self).__init__()

        global ssty
        old_model = torch.load('results/models/model.pth')
        old_weight = old_model.state_dict()
        channel_number_list = analyse_number(old_weight)
        self.inplanes = 64
        self.g_id = group_id
        self.layer_channel_num = None
        self.beta = 0.1
        self.channel_num = [ [64, 64]*layers[0] , [128, 128]*layers[1] , [256, 256]*layers[2] , [512, 512]*layers[3] ]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(channel_number_list[0], 0, block, 64, layers[0])
        self.layer2 = self._make_layer(channel_number_list[1], 1, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(channel_number_list[2], 2, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(channel_number_list[3], 3, block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        my_weight = self.state_dict()
        my_keys = list(my_weight.keys())
        for k, v in old_weight.items():
            name = ''.join(list(k)[7:])
            if name in my_keys:
                my_weight[name] = v
        self.load_state_dict(my_weight)

    def _make_layer(self, number_list, group_id, block, planes, blocks, stride=1):

        
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            
        layers = []
        if group_id == self.g_id:
            layers.append(Bottleneck_Sensity(number_list[0], self.channel_num[self.g_id], op_id=0, stride=stride, downsample=downsample))
        else:
            layers.append(block(number_list[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if group_id == self.g_id:
                layers.append(Bottleneck_Sensity(number_list[i], self.channel_num[self.g_id], op_id=i))
            else:
                layers.append(block(number_list[i]))

        return nn.Sequential(*layers)


    def forward(self, x, in_ssty, channel_index=None):
        global ssty
        ssty = in_ssty
        if not self.training:
            
            self.set_channel_index(channel_index)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)  # 128, 512, 28, 28
        x = self.layer3(x)  # 128, 1024, 14, 14
        x = self.layer4(x)  # 128, 2048, 7, 7

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        mask_vec = self.get_mask_vector()
        return x, mask_vec

    def set_channel_index(self, channel_index):
        if self.g_id == 0:
            self.layer1[0].channel_index1 = channel_index[0]
            self.layer1[0].channel_index2 = channel_index[1]
            self.layer1[1].channel_index1 = channel_index[2]
            self.layer1[1].channel_index2 = channel_index[3]
            self.layer1[2].channel_index1 = channel_index[4]
            self.layer1[2].channel_index2 = channel_index[5]
        elif self.g_id == 1:
            self.layer2[0].channel_index1 = channel_index[0]
            self.layer2[0].channel_index2 = channel_index[1]
            self.layer2[1].channel_index1 = channel_index[2]
            self.layer2[1].channel_index2 = channel_index[3]
            self.layer2[2].channel_index1 = channel_index[4]
            self.layer2[2].channel_index2 = channel_index[5]
            self.layer2[3].channel_index1 = channel_index[6]
            self.layer2[3].channel_index2 = channel_index[7]
        elif self.g_id == 2:
            self.layer3[0].channel_index1 = channel_index[0]
            self.layer3[0].channel_index2 = channel_index[1]
            self.layer3[1].channel_index1 = channel_index[2]
            self.layer3[1].channel_index2 = channel_index[3]
            self.layer3[2].channel_index1 = channel_index[4]
            self.layer3[2].channel_index2 = channel_index[5]
            self.layer3[3].channel_index1 = channel_index[6]
            self.layer3[3].channel_index2 = channel_index[7]
            self.layer3[4].channel_index1 = channel_index[8]
            self.layer3[4].channel_index2 = channel_index[9]
            self.layer3[5].channel_index1 = channel_index[10]
            self.layer3[5].channel_index2 = channel_index[11]
        else:
            self.layer4[0].channel_index1 = channel_index[0]
            self.layer4[0].channel_index2 = channel_index[1]
            self.layer4[1].channel_index1 = channel_index[2]
            self.layer4[1].channel_index2 = channel_index[3]
            self.layer4[2].channel_index1 = channel_index[4]
            self.layer4[2].channel_index2 = channel_index[5]

    def get_mask_vector(self):
        vector_list = list()
        if self.g_id == 0:
            vector_list.append(self.layer1[0].vec1)
            vector_list.append(self.layer1[0].vec2)
            vector_list.append(self.layer1[1].vec1)
            vector_list.append(self.layer1[1].vec2)
            vector_list.append(self.layer1[2].vec1)
            vector_list.append(self.layer1[2].vec2)

        elif self.g_id == 1:
            vector_list.append(self.layer2[0].vec1)
            vector_list.append(self.layer2[0].vec2)
            vector_list.append(self.layer2[1].vec1)
            vector_list.append(self.layer2[1].vec2)
            vector_list.append(self.layer2[2].vec1)
            vector_list.append(self.layer2[2].vec2)
            vector_list.append(self.layer2[3].vec1)
            vector_list.append(self.layer2[3].vec2)
        elif self.g_id == 2:
            vector_list.append(self.layer3[0].vec1)
            vector_list.append(self.layer3[0].vec2)
            vector_list.append(self.layer3[1].vec1)
            vector_list.append(self.layer3[1].vec2)
            vector_list.append(self.layer3[2].vec1)
            vector_list.append(self.layer3[2].vec2)
            vector_list.append(self.layer3[3].vec1)
            vector_list.append(self.layer3[3].vec2)
            vector_list.append(self.layer3[4].vec1)
            vector_list.append(self.layer3[4].vec2)
            vector_list.append(self.layer3[5].vec1)
            vector_list.append(self.layer3[5].vec2)
        else:
            vector_list.append(self.layer4[0].vec1)
            vector_list.append(self.layer4[0].vec2)
            vector_list.append(self.layer4[1].vec1)
            vector_list.append(self.layer4[1].vec2)
            vector_list.append(self.layer4[2].vec1)
            vector_list.append(self.layer4[2].vec2)
        return vector_list

    
    def set_beta_factor(self, sf):
        if self.g_id == 0:
            self.layer1[0].beta1 = sf[0]
            self.layer1[0].beta2 = sf[1]
            self.layer1[1].beta1 = sf[2]
            self.layer1[1].beta2 = sf[3]
            self.layer1[2].beta1 = sf[4]
            self.layer1[2].beta2 = sf[5]
        elif self.g_id == 1:
            self.layer2[0].beta1 = sf[0]
            self.layer2[0].beta2 = sf[1]
            self.layer2[1].beta1 = sf[2]
            self.layer2[1].beta2 = sf[3]
            self.layer2[2].beta1 = sf[4]
            self.layer2[2].beta2 = sf[5]
            self.layer2[3].beta1 = sf[6]
            self.layer2[3].beta2 = sf[7]
        elif self.g_id == 2:
            self.layer3[0].beta1 = sf[0]
            self.layer3[0].beta2 = sf[1]
            self.layer3[1].beta1 = sf[2]
            self.layer3[1].beta2 = sf[3]
            self.layer3[2].beta1 = sf[4]
            self.layer3[2].beta2 = sf[5]
            self.layer3[3].beta1 = sf[6]
            self.layer3[3].beta2 = sf[7]
            self.layer3[4].beta1 = sf[8]
            self.layer3[4].beta2 = sf[9]
            self.layer3[5].beta1 = sf[10]
            self.layer3[5].beta2 = sf[11]
        else:
            self.layer4[0].beta1 = sf[0]
            self.layer4[0].beta2 = sf[1]
            self.layer4[1].beta1 = sf[2]
            self.layer4[1].beta2 = sf[3]
            self.layer4[2].beta1 = sf[4]
            self.layer4[2].beta2 = sf[5]

def analyse_number(weight):
    number_list = list()
    group_list = list()
    layer_list = list()
    old_name = '1.0.'
    old_group = '1'
    # channel_num = []
    for k, v in weight.items():
        if 'layer' in k and'conv' in k:
            current_name = k.split('layer')[1].split('conv')[0]
            current_group = current_name.split('.')[0]
            if current_name != old_name:
                old_name = current_name
                group_list.append(layer_list.copy())
                layer_list = list()
            if current_group != old_group:
                old_group = current_group
                number_list.append(group_list.copy())
                group_list = list()
            layer_list.append(v.size()[0])
            layer_list.append(v.size()[1])
            # if 'downsample' not in k:
            #     channel_num.append(v.size()[1])
    group_list.append(layer_list.copy())
    number_list.append(group_list.copy())
    # print(number_list)
    return number_list


def ResNet50_ssnet(group_id):
    model = ResNet_SSNet(group_id, Bottleneck, [3, 4, 6, 3])
    return model

def ResNet50():
    """ return a ResNet 50 object
    """
    return ResNet(Bottleneck, [3, 4, 6, 3])