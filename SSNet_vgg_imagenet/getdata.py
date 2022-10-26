import torch
from torchvision import models
from torchvision import datasets, transforms
import argparse
import os
import time

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--data_base', default='/nas/imagenet', type=str, help='the path of dataset')
parser.add_argument('--batch_size', default=24, type=int, help='batch size')
parser.add_argument('--gpu_id', default='0,1', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
def main():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = args.data_base
    print('\n[Phase 1 Prune] : Data Preperation')
    print("| Preparing data...")
    dsets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    train_loader = torch.utils.data.DataLoader(dsets['train'], batch_size=args.batch_size, shuffle=True, num_workers=8,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dsets['val'], batch_size=args.batch_size, shuffle=False, num_workers=8,pin_memory=True)
    print('data_loader_success!')
    
    torch.save(train_loader, 'datasets_64/prune/train_loader')
    torch.save(val_loader, 'datasets_64/prune/val_loader')
    t1 = time.time()
    train_loader = torch.load('datasets_64/prune/train_loader')
    t2 = time.time()
    val_loader = torch.load('datasets_64/prune/val_loader')
    t3 = time.time()
    print('train loading time: ' + str(t2-t1))
    print('val loading time: ' + str(t3-t2))
    # train_path = 'datasets/train'
    # val_path = 'datasets/val'
    # if not os.path.exists(train_path):
    #     os.makedirs(train_path)
    # if not os.path.exists(val_path):
    #     os.makedirs(val_path)
    
    print('\n[Phase 2 Finetune] : Data Preperation')
    print("| Preparing data...")
    dsets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    train_loader = torch.utils.data.DataLoader(dsets['train'], batch_size=64, shuffle=True, num_workers=8,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dsets['val'], batch_size=64, shuffle=False, num_workers=8,pin_memory=True)
    print('data_loader_success!')
    
    torch.save(train_loader, 'datasets_64/finetune/train_loader')
    torch.save(val_loader, 'datasets_64/finetune/val_loader')
    t1 = time.time()
    train_loader = torch.load('datasets_64/finetune/train_loader')
    t2 = time.time()
    val_loader = torch.load('datasets_64/finetune/val_loader')
    t3 = time.time()
    print('train loading time: ' + str(t2-t1))
    print('val loading time: ' + str(t3-t2))
    # for x in enumerate(val_loader):
    #     i, (input, target) = x
    #     name = val_path + '/' + str(i)
    #     torch.save(x, name)
    

if __name__ == "__main__":
    main()
    