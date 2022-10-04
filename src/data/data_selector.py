import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split

def data_sel(name, root, train=True, val=0.2):
    if name == 'cifar10':
        ds = datasets.CIFAR10(root=root, train=True, transform=transforms.Compose([
                    torchvision.transforms.RandomCrop(size=32, padding=4),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010], 
                        ),
                    ]), download=True)
        
        test_ds = datasets.CIFAR10(root=root, train=False, transform=transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010], 
                        ),
                    ]), download=True)
        
        if train:
            num_val = int(val*len(ds))
            num_train = len(ds) - num_val
            train_ds, val_ds = random_split(ds, [num_train, num_val])
            return train_ds, val_ds

        return test_ds


        
        

