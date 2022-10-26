# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import torch
import os
import argparse
import torch.backends.cudnn as cudnn
from CompressingSSNet import mobilenet_compress, init_channel_dim, update_channel_dim


parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--block_id', default=1, type=int, help='block id to be pruned, range from 1 to 7(inclusive)')
parser.add_argument('--width_mult', default=1., type=float, help='the width_mult param in network')
parser.add_argument('--t_block_zero', default=1, type=int, help='the expansion ratio for block zero, which is not pruned')
parser.add_argument('--log_dir_path', default='/nas/guosuhan/auto-prune/logs/imagenet/results_2/logs', help='path to logs folder')
parser.add_argument('--model_dir_path', default='/nas/guosuhan/auto-prune/logs/imagenet/results_2/model', type=str, help='path to model folder')
args = parser.parse_args()
print(args)

device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

def main():
    print('Compress model')
    # phase 1. init/load channel dim
    if args.block_id == 1:
        print('init channel_dim')
        channel_dim = init_channel_dim(args.width_mult, args.t_block_zero)
        print('init Finished!')
    else:
        print('load channel_dim')
        channel_dim = torch.load(args.log_dir_path + '/block_' + str(args.block_id-1) + '/channel_dim.pth')
        print('load Finished!')

    # Phase 2. create compressed model
    model_path = args.model_dir_path + '/block_' + str(args.block_id) + '/model.pth'
    channel_path = args.model_dir_path + '/block_' + str(args.block_id) + '/channel_index.pth' 
    mbnetv2_new = mobilenet_compress(args.block_id, model_path, channel_path, channel_dim)

    # Phase 3 : Model setup
    mbnetv2_new = torch.nn.DataParallel(mbnetv2_new).to(device)
    cudnn.benchmark = True
    # save to block dir
    save_model_path = args.model_dir_path + '/block_' + str(args.block_id) + '/compressed_model.pth' 
    save_dict_path =  args.model_dir_path + '/block_' + str(args.block_id) + '/compressed_model_state_dict.pth'
    torch.save(mbnetv2_new, save_model_path)
    torch.save(mbnetv2_new.state_dict(), save_dict_path)
    # update model for next block pruning
    update_path = args.model_dir_path + '/model.pth'
    torch.save(mbnetv2_new, update_path) 
    print('Compression Finished!')

    # Phase 4: update channel_dim
    print('update channel_dim')
    channel_dim = update_channel_dim(channel_path, channel_dim)
    save_path = args.log_dir_path + '/block_' + str(args.block_id) 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(channel_dim, save_path + '/channel_dim.pth')
    print('update finished and saved')

    print('The most recent channel dim:')
    print(channel_dim)


if __name__ == '__main__':
    main()
