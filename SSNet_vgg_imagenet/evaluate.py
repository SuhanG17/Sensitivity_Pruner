import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5,1,4'
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
import SSNet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--gpu_id', default='4,0,6', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--compression_rate', default=0.4, type=float, help='the percentage of 1 in compressed model')
args = parser.parse_args()
best_prec1 = 0
print(args)
def main():

    # create model
    # pretrained
    # model = SsNet_test('/nas/laibl/SSNet/SSNet-CNN/code/SSNet_vgg_imagenet/results/models/model_1.pth')
    # model = SSNet.Vgg16()
    # pruned
    model = SsNet_test('results/models/fine_tune/model.pth')
    # model = SsNet_test('/nas/laibl/SSNet/SSNet-CNN/prediction_result/ImageNet/vgg16/results_0.6/models/model.pth')


    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))

    # pretrained
    # print('Compression Rate : '+str(1.0))
    # flops, params = clever_format([flops, params], "%.3f")
    # print('FLOPs : ' + str(flops))
    # print('Params : ' + str(params))
    
    flops, params = profile(model, inputs=(input,))
    all_flops = 15.511*(10**6)*(10**3)
    all_param = 138.366*(10**6)
    # rate = int((all_flops-flops)*100/all_flops)
    rate = round((all_flops-flops)*100/all_flops, 3)
    # params_rate = int((all_param-params)*100/all_param)
    params_rate = round((all_param-params)*100/all_param, 3) 
    print('Compression Rate : '+str(args.compression_rate))
    flops, params = clever_format([flops, params], "%.3f")
    print('FLOPs : ' + str(flops))
    print('FLOPs Drop: ' + str(rate) + '%')
    print('Params : ' + str(params))
    print('Params Drop: ' + str(params_rate)+ '%')
    # evaluate and train


if __name__ == '__main__':
    main()
