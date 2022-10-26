import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from Compresing_SSNet import SsNet_test
from thop import profile
from thop import clever_format


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--gpu_id', default='2', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--compression_rate', default=0.3, type=float, help='the percentage of 1 in compressed model')
args = parser.parse_args()
best_prec1 = 0
print(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
def main():
    global args, best_prec1, device

    model = SsNet_test('results/models/fine_tune/model.pth')

    input = torch.randn(1, 3, 32, 32)
    flops, params = profile(model, inputs=(input,))
    all_flops = 314.031*(10**6)
    rate = 1 - flops/all_flops
    print('Beta : 0.1')
    flops, params = clever_format([flops, params], "%.3f")
    print('FLOPs : ' + str(flops))
    print('FLOPs Reduction : '+ str(int(rate*100)) + '%')
    print('params : ' + str(params))
    # evaluate and train


if __name__ == '__main__':
    main()
