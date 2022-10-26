import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,7'
import torch

from thop import profile
from thop import clever_format
# load model
from SSNet_mobilenetv2 import mobile_v2 # pretraine model
from CompressingSSNet import mobilenet_test # compressed + finetuned model


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--pretrained_model', action='store_true',  
                    help='if specified, check params and flops for pretrained model (not pruned, compression rate = 1.)')
parser.add_argument('--model_dir_path', default='/nas/guosuhan/auto-prune/logs/imagenet/results_4/model',
                    type=str, help='the path of model folder')
parser.add_argument('--channel_dim_path', default='/nas/guosuhan/auto-prune/logs/imagenet/results_4/logs/block_7/channel_dim.pth', type=str,
                    help='path to channel_dim, not used for model compress, but to build model with correct channel num')
parser.add_argument('--checkpoint_path', default='/nas/guosuhan/auto-prune/logs/imagenet/results_4/checkpoint', type=str,
                    help='path to save finetune checkpoint, state_dict')
parser.add_argument('--compression_rate', default=0.4, type=float, help='the percentage of 1 in compressed model')
args = parser.parse_args()
best_prec1 = 0
print(args)
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

def main():

    # create model
    if args.pretrained_model: # pretrained model
        model = mobile_v2(args.model_dir_path+'/pretrained_mobilev2.pth').to(device)
        model = model.to(device) 
    else: 
        channel_dim = torch.load(args.channel_dim_path)
        model = mobilenet_test(channel_dim)
        model_ft = torch.nn.DataParallel(model).to(device)
        weight = torch.load(os.path.join(args.checkpoint_path, 'model_state_dict.pth'))        
        model_ft.load_state_dict(weight)
        model = model_ft.module 

    # dummy input
    input = torch.rand(1, 3, 224, 224).to(device) 
    
    # calculate flops and params 
    if args.pretrained_model: # pretrained model
        flops, params = profile(model, inputs=(input,))
        print('Compression Rate : '+str(1.0))
        flops, params = clever_format([flops, params], "%.3f")
        print('FLOPs : ' + str(flops))
        print('Params : ' + str(params))

    else:
        flops, params = profile(model, inputs=(input,))
        all_flops = 314.194*(10**6)
        all_param = 3.505*(10**6)
        rate = round((all_flops-flops)*100/all_flops, 2)
        params_rate = round((all_param-params)*100/all_param, 2)
        print('Compression Rate : '+str(args.compression_rate))
        flops, params = clever_format([flops, params], "%.3f")
        print('FLOPs : ' + str(flops))
        print('FLOPs Drop: ' + str(rate) + '%')
        print('Params : ' + str(params))
        print('Params Drop: ' + str(params_rate)+ '%')


if __name__ == '__main__':
    main()

