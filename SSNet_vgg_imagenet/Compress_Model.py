
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
from Compresing_SSNet import SsNet_compressed
import argparse
import torch.backends.cudnn as cudnn


parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--layer_id', default=0, type=int, help='the id of compressed layer, starting from 0')
parser.add_argument('--gpu_id', default='0,1', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()
print(args)
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(model_path):
    # 1. create compressed model
    vgg16_new = SsNet_compressed(layer_id=args.layer_id, model_path=model_path)
    # 2. Model setup
    vgg16_new = vgg16_new.cuda()
    vgg16_new = torch.nn.DataParallel(vgg16_new.cuda(), device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    torch.save(vgg16_new, 'results/models/model.pth')

    print('Finished!')


if __name__ == '__main__':
    folder_path = 'results/models/layer_' + str(args.layer_id)+'/'

    main(folder_path)
