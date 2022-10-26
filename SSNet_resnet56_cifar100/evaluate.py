import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from Compresing_SSNet import Resnet_test
from thop import profile
from thop import clever_format


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--gpu_id', default='1', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--compression_rate', default=0.4, type=float, help='the percentage of 1 in compressed model')
args = parser.parse_args()
best_prec1 = 0
print(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
def main():
    global args, best_prec1, device
    # create model
    model = Resnet_test('results/models/model.pth')
    input = torch.randn(1, 3, 32, 32)
    
    flops, params = profile(model, inputs=(input,))
    print('Compression Rate : '+str(args.compression_rate))
    orgin_flops = 1305*(10**6)
    origin_param = 23.705*(10**6)
    rate = int((orgin_flops - flops)*100/orgin_flops)
    param_rate = (origin_param - params)*100/origin_param
    flops, params = clever_format([flops, params], "%.3f")
    print('FLOPs : ' + str(flops))
    print('FLOPs Drop : ' + str(rate) + '%')
    print('Params : ' + str(params))
    print('Params Drop : {:.2f}%'.format(param_rate))


if __name__ == '__main__':
    main()
