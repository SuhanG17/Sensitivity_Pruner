import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import torch
import torch.nn as nn
import numpy as np
# device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, layer_id):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.expand_ratio = expand_ratio
        self.identity = stride == 1 and inp == oup
        self.ReLU = nn.ReLU6(inplace=True)

        self.beta1 = 1.0
        self.channel_index1 = None
        self.layer_id = layer_id # added for code writing, not used

        if expand_ratio == 1:
            # dw
            self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            # pw-linear
            self.conv2 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
            self.bn2 = nn.BatchNorm2d(oup)
        else:
            # pw
            self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
           # dw
            self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_dim)
            # pw-linear
            self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
            self.bn3 = nn.BatchNorm2d(oup)

    def forward(self, x):
        output = x
        if self.expand_ratio == 1:
            x = self.ReLU(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
        else:
            x = self.ReLU(self.bn1(self.conv1(x)))
            x = self.ReLU(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))

        if self.identity:
            return x + output
        else:
            return x

class InvertedResidual_Pruned(nn.Module):
    def __init__(self, inp, oup, hidden_dim, stride, expand_ratio, layer_id):
        super(InvertedResidual_Pruned, self).__init__()
        assert stride in [1, 2]

        # hidden_dim = round(inp * expand_ratio)
        self.expand_ratio = expand_ratio
        self.identity = stride == 1 and inp == oup
        self.ReLU = nn.ReLU6(inplace=True)

        self.beta1 = 1.0
        self.channel_index1 = None
        self.layer_id = layer_id # added for code writing, not used

        if expand_ratio == 1:
            # dw
            self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            # pw-linear
            self.conv2 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
            self.bn2 = nn.BatchNorm2d(oup)
        else:
            # pw
            self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
           # dw
            self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_dim)
            # pw-linear
            self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
            self.bn3 = nn.BatchNorm2d(oup)

    def forward(self, x):
        output = x
        if self.expand_ratio == 1:
            x = self.ReLU(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
        else:
            x = self.ReLU(self.bn1(self.conv1(x)))
            x = self.ReLU(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))

        if self.identity:
            return x + output
        else:
            return x


class MobileNetV2Compress(nn.Module):
    def __init__(self, block_to_be_pruned, model_path, channel_path, channel_dim, num_classes=1000, width_mult=1.):
        super(MobileNetV2Compress, self).__init__()

        # load channel dim for pruned block
        channel_index = torch.load(channel_path)
        channel_dim_prune = []
        for index in channel_index:
            channel_dim_prune.append(np.where(index != 0)[0].shape[0])
        
        # load channel dim for compressed block
        self.channel_dim = channel_dim

        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # set block id to be pruned
        self.block_to_be_pruned = block_to_be_pruned
        # set block index for mask retrieval and application
        self.block_dict = self.generate_block_dict()

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]

        # building inverted residual blocks
        for i, (t, c, n, s) in enumerate(self.cfgs): 
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8) 
            if i > 0 and i < self.block_to_be_pruned:
                print(f'compress block {i}')
                block = InvertedResidual_Pruned
                for j in range(n):
                    hidden_channel = self.channel_dim[i][j]
                    if j == 0:
                        layers.append(block(input_channel, output_channel, hidden_channel, s, expand_ratio=t, layer_id=j))
                    else:
                        layers.append(block(input_channel, output_channel, hidden_channel, 1, expand_ratio=t, layer_id=j)) 
                    input_channel = output_channel
            else:
                if i == self.block_to_be_pruned:
                    print('prune current block') 
                    block = InvertedResidual_Pruned
                    for j in range(n):
                        hidden_channel = channel_dim_prune[j]
                        if j == 0:
                            layers.append(block(input_channel, output_channel, hidden_channel, s, expand_ratio=t, layer_id=j))
                        else:
                            layers.append(block(input_channel, output_channel, hidden_channel, 1, expand_ratio=t, layer_id=j)) 
                        input_channel = output_channel 
                else:
                    print(f'not prune block {i}') 
                    block = InvertedResidual
                    for j in range(n):
                        if j == 0:
                            layers.append(block(input_channel, output_channel, s, expand_ratio=t, layer_id=j))
                        else:
                            layers.append(block(input_channel, output_channel, 1, expand_ratio=t, layer_id=j)) 
                        input_channel = output_channel 

        self.features = nn.Sequential(*layers)

        # building last several layers
        if self.block_to_be_pruned == 7:
            output_channel = channel_dim_prune[-1] 
        else: 
            output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights(model_path, channel_index)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def generate_block_dict(self):
        block_dict = {}
        start_layer = 1 # the 0th layer in features is a simple conv, hence start with 1
        for i, (_, _, n, _) in enumerate(self.cfgs): 
            end_layer = start_layer + n
            print(f'block {i}, start with features layer {start_layer}, end with features layer {end_layer}')
            block_dict[i] = (start_layer, end_layer)
            start_layer = end_layer  
        return block_dict

    def _initialize_weights(self, model_path=None, channel_index=None):
        model_weights = torch.load(model_path)
        if not isinstance(model_weights, dict):
            model_weights = model_weights.state_dict()

        my_weight = self.state_dict()
    
        if self.block_to_be_pruned == 7:
            channel_ind = torch.cuda.LongTensor(np.where(channel_index[0] != 0)[0])
            for k, v in model_weights.items():
                name = k[7:] # remove module in name 
                if 'conv.0' in k:
                    my_weight[name] = v[channel_ind, :, :, :] # the last conv: pruned
                elif 'conv.1' in k and 'num_batches_tracked' not in k:
                    my_weight[name] = v[channel_ind] # bn related: pruned
                elif 'classifier.weight' in k:
                    my_weight[name] = v[:, channel_ind] # classification weight: pruned
                elif 'Sensity_Filter' in k: # Sensity_layer: unused
                    continue 
                else:
                    my_weight[name] = v # feature blocks: unpruned 
        else:
            counter = 0
            for k, v in model_weights.items():
                name = k[7:] # remove module in name
                if 'features' in k:
                    layer_num = int(k.split('.')[2])        
                    if layer_num >= self.block_dict[self.block_to_be_pruned][0] and layer_num < self.block_dict[self.block_to_be_pruned][1]:
                        if 'conv1' in k:
                            channel_ind = torch.cuda.LongTensor(np.where(channel_index[counter] != 0)[0])
                            my_weight[name] = v[channel_ind, :, :, :] # 1st IR block conv: pruned 
                        if 'bn1' in k:
                            if 'num_batches_tracked' not in k: 
                                my_weight[name] = v[channel_ind] # 1st bn layer: pruned 
                        if 'conv2' in k:
                            my_weight[name] = v[channel_ind, :, :, :] # 2nd IR block conv: pruned
                        if 'bn2' in k:
                            if 'num_batches_tracked' not in k: 
                                my_weight[name] = v[channel_ind] # 2nd bn layer: pruned 
                        if 'bn3' in k:
                            if 'num_batches_tracked' not in k:
                                my_weight[name] = v  # 3rd bn layer: unpruned
                        if 'conv3' in k:
                            my_weight[name] = v[:, channel_ind, :, :] # 3rd IR block conv: pruned
                            counter += 1 # increment after last conv layer
                    else:
                        my_weight[name] = v # other IR blocks: unpruned
                else:
                    my_weight[name] = v # last conv, classifier: unpruned

        self.load_state_dict(my_weight)


def mobilenet_compress(block_to_be_pruned, model_path, channel_path, channel_dim):
    model = MobileNetV2Compress(block_to_be_pruned, model_path, channel_path, channel_dim, num_classes=1000, width_mult=1.)
    return model

def init_channel_dim(width_mult, t_block_zero):
    ''' initiate channel dim
    Args:
        width_mult: refer to original MobileNetv2 construction
        t_block_zero: refer to original MobileNetv2 construction
    Returns:
        channel_dim: list of lists, containing the output channel of first IR block, default not pruned
    '''
    input_dim_block_zero = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
    hidden_dim_block_zero = round(input_dim_block_zero * t_block_zero) 

    channel_dim = []
    channel_dim.append([hidden_dim_block_zero])
    return channel_dim

def update_channel_dim(channel_path, channel_dim):
    ''' update channel dim, based on prunining result
    Args:
        channel_path: list of arrays results from pruning process
        channel_dim: channel dim object from previous block or init
    Returns:
        channel_dim: list of lists, updated channel dim 
    '''
    channel_index = torch.load(channel_path)
    block_channel_dim=[]
    for index in channel_index:
        block_channel_dim.append(np.where(index != 0)[0].shape[0])
    channel_dim.append(block_channel_dim)

    return channel_dim


class MobileNetV2Test(nn.Module):
    def __init__(self, channel_dim, num_classes=1000, width_mult=1.):
        super(MobileNetV2Test, self).__init__()

        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]

        # building inverted residual blocks
        block = InvertedResidual_Pruned 
        for i, (t, c, n, s) in enumerate(self.cfgs): 
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
             # first block not pruned, following 6 block pruned
            for j in range(n):
                hidden_channel = channel_dim[i][j]
                if j == 0:
                    layers.append(block(input_channel, output_channel, hidden_channel, s, expand_ratio=t, layer_id=j))
                else:
                    layers.append(block(input_channel, output_channel, hidden_channel, 1, expand_ratio=t, layer_id=j)) 
                input_channel = output_channel

        self.features = nn.Sequential(*layers)
        # building last several layers
        # output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        output_channel = channel_dim[-1][0]
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def mobilenet_test(channel_dim):
    model = MobileNetV2Test(channel_dim, num_classes=1000, width_mult=1.)
    return model

# channel_dim = [[16], [32, 32], [...], [...], [...], [...], [...], [1280]]



############################# EXP
# model_path = '/nas/guosuhan/auto-prune/logs/imagenet/results_4/model/block_1/model.pth'
# channel_path = '/nas/guosuhan/auto-prune/logs/imagenet/results_4/model/block_1/channel_index.pth'
# channel_dim = torch.load('/nas/guosuhan/auto-prune/logs/imagenet/results_4/logs/block_1/channel_dim.pth')
# channel_dim = init_channel_dim(1, 1) 

# my_weight = mobilenet_compress(1, model_path, channel_path, channel_dim)
# print(my_weight)
# my_weight = my_weight.state_dict()

# channel_index = torch.load(channel_path)
# channel_dim = []
# for index in channel_index:
#     channel_dim.append(np.where(index != 0)[0].shape[0])

# model_weights = torch.load(model_path)
# if not isinstance(model_weights, dict):
#     model_weights = model_weights.state_dict()

# torch.cuda.LongTensor(np.where(channel_index[0] != 0)[0])
# channel_ind = torch.cuda.LongTensor(np.where(channel_index[0] != 0)[0])

# for k, v in model_weights.items():
#     name = k[7:] # remove module in name 
#     if 'conv.0' in k:
#         # my_weight[name] = v[channel_ind, :, :, :] # the last conv: pruned
#         print(name)
#         # print(v.shape)
#     elif 'conv.1' in k and 'num_batches_tracked' not in k:
#         # my_weight[name] = v[channel_ind] # bn related: pruned
#         print(name)
#         # print(v.shape)
#     elif 'classifier.weight' in k:
#         # my_weight[name] = v[:, channel_ind] # classification weight: pruned
#         print(name)
#         # print(v.shape)
#     # elif 'classifier.bias' in k:
#         # my_weight[name] = v[channel_ind] # classification bias: pruned
#         # print(name)
#         # print(v.shape)
#     elif 'Sensity_Filter' in k: # Sensity_layer: unused
#         continue 
#     else:
#         my_weight[name] = v # feature blocks: unpruned 
#         # print(name)
#         # print(v.shape)



# channel_index = torch.load('/nas/guosuhan/auto-prune/logs/imagenet/results_1/model/channel_index.pth')
# model = torch.load('/nas/guosuhan/auto-prune/logs/imagenet/results_1/model/model.pth')
# model_weights = model.state_dict()
# for k, v in model_weights.items():
#     name = k[7:]
#     print(f'{name}, weight shape {v.shape}')
#     if 'features' in k:
#         print(name.split('.')[1])
#         layer_num = int(k.split('.')[2])
#         print(layer_num)
#         if layer_num > 1:
#             channel_ind = torch.cuda.LongTensor(np.where(channel_index[layer_num-2] != 0)[0])

# channel_dim = []
# for index in channel_index:
#     channel_dim.append(np.where(index != 0)[0].shape[0])
    # print(np.where(index != 0)[0].shape)


# model_ft = mobilenet_v2('/nas/guosuhan/auto-prune/logs/imagenet/results_1/model/model.pth', '/nas/guosuhan/auto-prune/logs/imagenet/results_1/model/channel_index.pth')

# model =  MobileNetV2(None, None, num_classes=1000, width_mult=1.)
# self_weights = model.state_dict()
# for k, v in self_weights.items():
#     print(k)

# # load channel dim
# channel_path = '/nas/guosuhan/auto-prune/logs/imagenet/results_4/model/block_1/channel_index.pth'
# channel_index = torch.load(channel_path)
# channel_dim = []
# for index in channel_index:
#     channel_dim.append(np.where(index != 0)[0].shape[0])

# # init model
# block_to_be_pruned = 1
# model_weights = torch.load('/nas/guosuhan/auto-prune/logs/imagenet/results_4/model/block_1/model.pth')
# model_weights = model_weights.state_dict()
# my_model = MobileNetV2Compress(block_to_be_pruned, None, channel_path)
# my_weight = my_model.state_dict()


# # block_to_be_pruned < 7 
# counter = 0
# for k, v in model_weights.items():
#     name = k[7:] # remove module in name
#     if 'features' in k:
#         layer_num = int(k.split('.')[2])        
#         if layer_num >= my_model.block_dict[block_to_be_pruned][0] and layer_num < my_model.block_dict[block_to_be_pruned][1]:
#             # channel_ind = torch.cuda.LongTensor(np.where(channel_index[counter] != 0)[0])
#             # print(f'name {name}, channel_ind {channel_ind}')
#             # print(f'name {name}, layer_num {layer_num}')
#             # counter += 1
#             # if 'conv' in k:
#             if 'conv1' in k:
#                 channel_ind = torch.cuda.LongTensor(np.where(channel_index[counter] != 0)[0])
#                 # print(f'name {name}, channel_ind {channel_ind}')
#                 my_weight[name] = v[channel_ind, :, :, :]
#             if 'bn1' in k:
#                 # print(f'name {name}, channel_ind {channel_ind}')
#                 if 'num_batches_tracked' not in k: # prune bn layer
#                     my_weight[name] = v[channel_ind] 
#             if 'conv2' in k:
#                 # print(f'name {name}, channel_ind {channel_ind}') 
#                 my_weight[name] = v[channel_ind, :, :, :]
#             if 'bn2' in k:
#                 # print(f'name {name}, channel_ind {channel_ind}')
#                 if 'num_batches_tracked' not in k: # prune bn layer
#                     my_weight[name] = v[channel_ind] 
#             if 'bn3' in k:
#                 # print(f'name {name}, channel_ind {channel_ind}') 
#                 if 'num_batches_tracked' not in k: # prune bn layer
#                     my_weight[name] = v
#             if 'conv3' in k:
#                 # print(f'name {name}, channel_ind {channel_ind}') 
#                 my_weight[name] = v[:, channel_ind, :, :] 
#                 counter += 1
#     else:
#         my_weight[name] = v


# channel_ind = torch.cuda.LongTensor(np.where(channel_index[0] != 0)[0])
# for k, v in model_weights:
#     if 'conv.0' in k: # prune the last conv
#         my_weight[name] = v[channel_ind, :, :, :]
#     elif 'conv.1' in k and 'num_batches_tracked' not in k:
#         my_weight[name] = v[channel_ind] # prune bn layers
#     elif 'classifier.weight' in k:
#         my_weight[name] = v[channel_ind, :] # prune classification weight
#     elif 'classifier.bias' in k:
#         my_weight[name] = v[channel_ind] # prune classification bias
#     elif 'Sensity_Filter' in k: # Sensity_layer: unused
#         continue 
#     else:
#         my_weight[name] = v # feature blocks: unpruned 




# if self.block_to_be_pruned == 7:
#     channel_ind = torch.cuda.LongTensor(np.where(channel_index[0] != 0)[0])
#     for k, v in model_weights:
#         if 'conv.0' in k: # prune the last conv
#             my_weight[name] = v[channel_ind, :, :, :]
#         elif 'conv.1' in k and 'num_batches_tracked' not in k:
#             my_weight[name] = v[channel_ind] # prune bn layers
#         elif 'classifier.weight' in k:
#             my_weight[name] = v[channel_ind, :] # prune classification weight
#         elif 'classifier.bias' in k:
#             my_weight[name] = v[channel_ind] # prune classification bias
#         elif 'Sensity_Filter' in k: # Sensity_layer: unused
#             continue 
#         else:
#             my_weight[name] = v # feature blocks: unpruned 
# else:
#     counter = 0
#     for k, v in model_weights.items():
#         name = k[7:] # remove module in name
#         if 'features' in k:
#             layer_num = int(k.split('.')[2])        
#             if layer_num >= self.block_dict[self.block_to_be_pruned][0] and layer_num < self.block_dict[self.block_to_be_pruned][1]:
#                 if 'conv1' in k:
#                     channel_ind = torch.cuda.LongTensor(np.where(channel_index[counter] != 0)[0])
#                     my_weight[name] = v[channel_ind, :, :, :] # 1st IR block conv: pruned 
#                 if 'bn1' in k:
#                     if 'num_batches_tracked' not in k: 
#                         my_weight[name] = v[channel_ind] # 1st bn layer: pruned 
#                 if 'conv2' in k:
#                     my_weight[name] = v[channel_ind, :, :, :] # 2nd IR block conv: pruned
#                 if 'bn2' in k:
#                     if 'num_batches_tracked' not in k: 
#                         my_weight[name] = v[channel_ind] # 2nd bn layer: pruned 
#                 if 'bn3' in k:
#                     if 'num_batches_tracked' not in k:
#                         my_weight[name] = v  # 3rd bn layer: unpruned
#                 if 'conv3' in k:
#                     my_weight[name] = v[:, channel_ind, :, :] # 3rd IR block conv: pruned
#                     counter += 1 # increment after last conv layer
#         else:
#             my_weight[name] = v # other IR blocks, last conv, classifier: unpruned