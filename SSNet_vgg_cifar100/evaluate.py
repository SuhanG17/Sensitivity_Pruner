import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5,1,4'
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from Compresing_SSNet import SsNet_test
from thop import profile
from thop import clever_format


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--gpu_id', default='4,0,6', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--compression_rate', default=0.4, type=float, help='the percentage of 1 in compressed model')
args = parser.parse_args()
best_prec1 = 0
print(args)
def main():

    # create model
    model = SsNet_test('results/models/fine_tune/model.pth')

    input = torch.randn(1, 3, 32, 32)
    
    flops, params = profile(model, inputs=(input,))
    all_flops = 333.310*(10**6)
    all_param = 34.015*(10**6)
    rate = int((all_flops-flops)*100/all_flops)
    params_rate = int((all_param-params)*100/all_param)
    print('Compression Rate : '+str(args.compression_rate))
    flops, params = clever_format([flops, params], "%.3f")
    print('FLOPs : ' + str(flops))
    print('FLOPs Drop: ' + str(rate) + '%')
    print('Params : ' + str(params))
    print('Params Drop: ' + str(params_rate)+ '%')
    # evaluate and train


if __name__ == '__main__':
    main()
