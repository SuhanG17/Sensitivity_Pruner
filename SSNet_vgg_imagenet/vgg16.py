import torch
from torch import nn
import math

class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16,self).__init__()
        self.feature_1 = nn.Sequential()
        self.classifier = nn.Sequential()

        # add feature layers
        self.feature_1.add_module('conv1_1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu1_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv1_2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu1_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_1.add_module('conv2_1', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu2_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv2_2', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu2_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_1.add_module('conv3_1', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu3_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv3_2', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu3_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv3_3', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu3_3', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool3', nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_1.add_module('conv4_1', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu4_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv4_2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu4_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv4_3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu4_3', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool4', nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_1.add_module('conv5_1', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu5_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv5_2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu5_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv5_3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu5_3', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool5', nn.MaxPool2d(kernel_size=2, stride=2))

        # add classifier
        self.classifier.add_module('fc6', nn.Linear(512*7*7, 4096))
        self.classifier.add_module('relu6', nn.ReLU(inplace=True))
        self.classifier.add_module('dropout6', nn.Dropout())

        self.classifier.add_module('fc7', nn.Linear(4096, 4096))
        self.classifier.add_module('relu7', nn.ReLU(inplace=True))
        self.classifier.add_module('dropout7', nn.Dropout())

        self.classifier.add_module('fc8', nn.Linear(4096, 1000))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         m.bias.data.zero_()
        my_weight = self.state_dict()
        my_keys = list(my_weight.keys())
        count = 0
        for k, v in model_weight.items():
            my_weight[my_keys[count]] = v
            count += 1
        self.load_state_dict(my_weight)

    def forward(self, x):
        x = self.feature_1(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x