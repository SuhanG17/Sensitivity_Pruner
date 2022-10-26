import torch
from torchvision import datasets, transforms
import argparse
import os
import time

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--data_base', default='/nas/imagenet', type=str, help='the path of dataset')
parser.add_argument('--root_dir', default='/nas/data/imagenet', type=str, help='path to store loaders')
parser.add_argument('--batch_size', default=24, type=int, help='batch size')
parser.add_argument('--gpu_id', default='1,2', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
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

    os.makedirs(os.path.join(args.root_dir, 'prune'), exist_ok=True)
    os.makedirs(os.path.join(args.root_dir, 'finetune'), exist_ok=True)
    
    torch.save(train_loader, args.root_dir+'/prune/train_loader')
    torch.save(val_loader, args.root_dir+'/prune/val_loader')
    t1 = time.time()
    train_loader = torch.load(args.root_dir+'/prune/train_loader')
    t2 = time.time()
    val_loader = torch.load(args.root_dir+'/prune/val_loader')
    t3 = time.time()
    print('train loading time: ' + str(t2-t1))
    print('val loading time: ' + str(t3-t2))
    
    print('\n[Phase 2 Finetune] : Data Preperation')
    print("| Preparing data...")
    dsets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    train_loader = torch.utils.data.DataLoader(dsets['train'], batch_size=64, shuffle=True, num_workers=8,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dsets['val'], batch_size=64, shuffle=False, num_workers=8,pin_memory=True)
    print('data_loader_success!')
    
    torch.save(train_loader, args.root_dir+'/finetune/train_loader')
    torch.save(val_loader, args.root_dir+'/finetune/val_loader')
    t1 = time.time()
    train_loader = torch.load(args.root_dir+'/finetune/train_loader')
    t2 = time.time()
    val_loader = torch.load(args.root_dir+'/finetune/val_loader')
    t3 = time.time()
    print('train loading time: ' + str(t2-t1))
    print('val loading time: ' + str(t3-t2))
    

if __name__ == "__main__":
    main()
    