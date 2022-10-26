import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from Compresing_SSNet import Resnet_test
from thop import profile
from thop import clever_format


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=2, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', default=False, type=bool, help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--gpu_id', default='3,1,0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--data_base', default='/nas/imagenet', type=str, help='the path of dataset')
parser.add_argument('--train_path', default='datasets/finetune/train_loader', type=str, help='the path of dataset')
parser.add_argument('--val_path', default='datasets/finetune/val_loader', type=str, help='the path of dataset')
args = parser.parse_args()
best_prec1 = 0
print(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def main():
    global args, best_prec1, device

    model = Resnet_test('results-0.2/models/fine_tune/model.pth')

    input = torch.randn(1, 3, 32, 32)
    
    flops, params = profile(model, inputs=(input,))
    
    flops, params = clever_format([flops, params], "%.3f")
    print(flops)
    print(params)
    # evaluate and train


if __name__ == '__main__':
    main()
