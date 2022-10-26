import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch.nn as nn
import math
import torch
import Sensity_Filter
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
total_pruned_layers = 16 + 1 = 17
num_layer_to_be_pruned: number of layers to prune in that block
beta should have shape [num_layer_to_be_pruned, beta_range]
channel index should be a list of length num_layer_to_be_pruned, each element is the pruning mask 
block_to_be_pruned: start with 1, end with 7(inclusive), 7 indicates the last conv layer (oup=1280)


In features, 
there is 18 layers, id range from 0 to 17
there is 7 IR blocks, id range from 0 to 6

feature id 0 indicates the first conv layer, not pruned
feature id 1 indicates the first IR block, not prune
feature id 2-17 indicate the IR blocks to be pruned
feature id 2-17 form IR blocks 1-6, hence, block_to_be_pruned include index 1-6
'''

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



class MobileNetV2(nn.Module):
    def __init__(self, model_path, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
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

        # set block index for Setting_Sensit
        self.block_dict = self.generate_block_dict()

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                if i == 0:
                    layers.append(block(input_channel, output_channel, s, expand_ratio=t, layer_id=None))
                else:
                    layers.append(block(input_channel, output_channel, 1, expand_ratio=t, layer_id=None))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights(model_path)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def generate_block_dict(self):
        block_dict = {}
        start_layer = 1 # the 0th layer in features is a simple conv, hence start with 1, notice that though the first conv is not to be pruned, it's included in the dict 
        for i, (_, _, n, _) in enumerate(self.cfgs): 
            end_layer = start_layer + n
            print(f'block {i}, start with features layer {start_layer}, end with features layer {end_layer}')
            block_dict[i] = (start_layer, end_layer)
            start_layer = end_layer  
        return block_dict 


    def _initialize_weights(self, model_path=None):
        if model_path:
            model_weight = torch.load(model_path)
            if not isinstance(model_weight, dict):
                model_weight = model_weight.state_dict()

            my_weight = self.state_dict()
            my_keys = list(my_weight)
            new_keys = []
            for item in my_keys:
                if 'Sensity_Filter' not in item:
                    new_keys.append(item)
                else:
                    print('woops, somthing wrong')
            for i, (k, v) in enumerate(model_weight.items()):
                my_weight[new_keys[i]] = v
            self.load_state_dict(my_weight)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

# model = MobileNetV2(model_path='model/pretrained_mobilev2.pth', num_classes=1000, width_mult=1)
# # dummy_input = torch.randn(4, 3, 224, 224)
# # out = model(dummy_input)
# # print(out.shape)
# torch.save(model, 'model/model_1.pth')

# from Comp_ssty import init_sensity_model, Setting_Sensity
# model = init_sensity_model('model/model_1.pth')
# model.block_dict

# val_loader = torch.load('/nas/guosuhan/auto-prune/data/imagenet64/prune/val_loader')
# for inputs, targets in val_loader:
#     print(inputs.shape) # [64, 3, 224, 224]
#     print(targets.shape) #[64]
#     break

# ssty, target = Setting_Sensity(model, inputs, targets, 0.5, 5, 1, 'cpu')
# len(ssty)
# len(target)
# ssty[0].shape
# target[0].shape

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

class InvertedResidual_Sensity(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, layer_id):
        super(InvertedResidual_Sensity, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.expand_ratio = expand_ratio
        self.identity = stride == 1 and inp == oup
        self.ReLU = nn.ReLU6(inplace=True)

        self.beta = 1.0
        self.channel_index = None
        self.layer_id = layer_id
        
        self.Sensity_Filter = Sensity_Filter.Use_Sensity(1, hidden_dim) 
        # ks = 1 because of depthwise conv cannot take more than 1
        self.vec = None

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
        global ssty

        output = x
        if self.expand_ratio == 1:
            x = self.ReLU(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
        else:
            x = self.ReLU(self.bn1(self.conv1(x)))
            if self.training:
                x, self.vec = self.Sensity_Filter(x, ssty[self.layer_id], self.beta)
            else:
                x, self.vec = self.Sensity_Filter(x, None, self.beta, self.channel_index)

            x = self.ReLU(self.bn2(self.conv2(x)))
            x = Sensity_Filter.Use_Mask_Conv.apply(x, self.vec)
            x = self.bn3(self.conv3(x))

        if self.identity:
            return x + output
        else:
            return x 


class MobileNetV2_Sensity(nn.Module):
    def __init__(self, block_to_be_pruned, model_path, channel_dim, num_classes=1000, width_mult=1.):
        super(MobileNetV2_Sensity, self).__init__()
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

        # set channel_dim to build model after first round of pruning
        self.channel_dim = channel_dim

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
                    block = InvertedResidual_Sensity
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
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        # building sensitivity layer for the last convolution and apply to classification
        if self.block_to_be_pruned == 7:
            print('prune last conv kernel')
            self.beta = 1.0
            self.channel_index = None
            
            self.Sensity_Filter = Sensity_Filter.Use_Sensity(1, output_channel) 
            # ks = 1 because of depthwise conv cannot take more than 1
            self.vec = None

        self._initialize_weights(model_path)

    def forward(self, x, in_ssty, channel_index=None):
        global ssty
        ssty = in_ssty
        if not self.training:
            self.set_channel_index(channel_index)

        x = self.features(x)
        x = self.conv(x)
        if self.block_to_be_pruned == 7:
            if self.training:
                x, self.vec = self.Sensity_Filter(x, ssty[0], self.beta)
            else:
                x, self.vec = self.Sensity_Filter(x, None, self.beta, self.channel_index)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        mask_vec = self.get_mask_vector()
        return x, mask_vec

    def generate_block_dict(self):
        block_dict = {}
        start_layer = 1 # the 0th layer in features is a simple conv, hence start with 1
        for i, (_, _, n, _) in enumerate(self.cfgs): 
            end_layer = start_layer + n
            print(f'block {i}, start with features layer {start_layer}, end with features layer {end_layer}')
            block_dict[i] = (start_layer, end_layer)
            start_layer = end_layer  
        return block_dict

    def set_channel_index(self, channel_index):
        # set channel index for last conv block, oup=1280 
        if self.block_to_be_pruned == 7: 
            self.channel_index = channel_index[-1]
        # set channel index for IR blocks 1-6, skip 0
        else: 
            counter = 0
            item_range = self.block_dict[self.block_to_be_pruned]
            for item in self.features._modules:
                if int(item) >= item_range[0] and int(item) < item_range[1]:
                    self.features._modules[item].channel_index = channel_index[counter] # block dict starts from layer 1 not 0
                    counter += 1
        
    def get_mask_vector(self):
        vector_list = []
        # retreive mask vector for last conv block, oup=1280
        if self.block_to_be_pruned == 7: 
            vector_list.append(self.vec) 
        # retreive mask vector for IR blocks 1-6, skip 0 
        else:
            item_range = self.block_dict[self.block_to_be_pruned]
            for item in self.features._modules:  
                if int(item) >= item_range[0] and int(item) < item_range[1]:
                    vector_list.append(self.features._modules[item].vec)
        return vector_list


    def set_beta_factor(self, sf):
        # set beta for last conv block, oup=1280 
        if self.block_to_be_pruned == 7:
            self.beta = sf[0]
        # set beta for IR blocks 1-6, skip 0 
        else:
            counter = 0 
            item_range = self.block_dict[self.block_to_be_pruned]
            for item in self.features._modules:
                if int(item) >= item_range[0] and int(item) < item_range[1]:
                    self.features._modules[item].beta = sf[counter] # beta has shape [n, 1]
                    counter += 1

    def _initialize_weights(self, model_path):
        model_weight = torch.load(model_path)
        if not isinstance(model_weight, dict):
            model_weight = model_weight.state_dict()

        # there is a conv layer for Use_sensity needs to be initialized,
        # other filter will be overwritten by the code after this chunk
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_() 

        my_weight = self.state_dict()

        for k, v in model_weight.items():
            name = k[7:] # strip 'module' in k
            my_weight[name] = v 

        self.load_state_dict(my_weight)


def mobile_v2(model_path):
    return MobileNetV2(model_path, num_classes=1000, width_mult=1)

def mobile_v2_ssnet(block_to_be_pruned, model_path, channel_dim):
    return MobileNetV2_Sensity(block_to_be_pruned, model_path, channel_dim, num_classes=1000, width_mult=1.) 

# block_id = 2
# compression_rate=0.4
# model_dir_path='/nas/guosuhan/auto-prune/logs/imagenet/results_5/model'
# log_dir_path='/nas/guosuhan/auto-prune/logs/imagenet/results_5/logs'
# channel_dim = torch.load(log_dir_path + '/block_' + str(block_id-1) + '/channel_dim.pth')
# channel_dim
# model_ft = mobile_v2_ssnet(block_id, model_dir_path+'/model.pth', channel_dim).to(device)
# print(model_ft)

# model_path = '/nas/guosuhan/auto-prune/logs/imagenet/results_4/model/model.pth'
# channel_dim = torch.load('/nas/guosuhan/auto-prune/logs/imagenet/results_4/logs'+ '/block_' + str(2-1) + '/channel_dim.pth')
# model_ft = MobileNetV2_Sensity(2, model_path, channel_dim)

# my_weight = model_ft.state_dict()

# for k, v in my_weight.items():
#     print(k)


# model = torch.load(model_path)
# model_weight = model.state_dict()

# print(model_ft)
# print(model)

# true_model = mobile_v2('/nas/guosuhan/auto-prune/logs/imagenet/results_4/model/pretrained_mobilev2.pth')
# print(true_model)


# before_compress = torch.load('/nas/guosuhan/auto-prune/logs/imagenet/results_4/model/block_1/model.pth')
# print(before_compress)

# compressed = torch.load('/nas/guosuhan/auto-prune/logs/imagenet/results_4/model/block_1/compressed_model.pth')
# print(compressed)

# for k, v in model_weight.items():
#     name = k[7:]
#     print(name)
#     if 'Sensity_Filter' in name:
#         continue
#     else:
#         my_weight[name] = v

# for k, v in model_weight.items():
#     my_weight[name] = v


# for name, layer in model.named_modules():
#     name = name[7:] # # remove 'module' in name
#     print(name)
#     if isinstance(layer, nn.Conv2d): 
#         nn.init.xavier_normal_(layer.weight)
#         # conv layer in features
#         if 'features' and 'conv1' in name:
#             if int(name.split('.')[1] )!= 1:
#                 print(name)
#                 # layer.mask_weight = nn.Parameter(torch.ones_like(layer.weight))
#                 # layer.forward = types.MethodType(conv_mask_forward, layer)
#         # last conv layer, not in features module
#         if 'conv.0' in name:
#             print(name)
#             # layer.mask_weight = nn.Parameter(torch.ones_like(layer.weight))
#             # layer.forward = types.MethodType(conv_mask_forward, layer)