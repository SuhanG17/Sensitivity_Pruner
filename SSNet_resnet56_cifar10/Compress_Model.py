import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from Compresing_SSNet import Resnet
import argparse
import torch.backends.cudnn as cudnn


parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--group_id', default=3, type=int, help='the id of compressed layer, starting from 0')
args = parser.parse_args()
print(args)


def main():
    # 1. create compressed model
    resnet50_new = Resnet(group_id=args.group_id)
    # Phase 2 : Model setup
    resnet50_new = resnet50_new.cuda()
    resnet50_new = torch.nn.DataParallel(resnet50_new.cuda(), device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    # new_model_param = vgg16_new.state_dict()
    torch.save(resnet50_new, 'results/models/model.pth')
    print('Finished!')


if __name__ == '__main__':
    main()
