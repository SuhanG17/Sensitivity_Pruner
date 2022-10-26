import os
import torch
from Compresing_SSNet import SsNet_compressed
import argparse
import torch.backends.cudnn as cudnn


parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--layer_id', default=12, type=int, help='the id of compressed layer, starting from 0')
parser.add_argument('--gpu_id', default='5,1,2', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def main(model_path):
    # 1. create compressed model
    vgg16_new = SsNet_compressed(layer_id=args.layer_id, model_path=model_path)
    # 2. Model setup
    vgg16_new = vgg16_new.cuda()
    vgg16_new = torch.nn.DataParallel(vgg16_new.cuda(), device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    torch.save(vgg16_new, 'results/models/model.pth')
    # torch.save(vgg16_new, 'results/models/model_2.pth')

    print('Finished!')


if __name__ == '__main__':
    folder_path = 'results/models/layer_' + str(args.layer_id)+'/'

    main(folder_path)
