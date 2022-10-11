import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

def data_sel(name, root, train=True, val=0.2):
    if name == 'cifar10':
        ds = datasets.CIFAR10(root=root, train=True, transform=transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010], 
                        ),
                    ]), download=True)
        
        test_ds = datasets.CIFAR10(root=root, train=False, transform=transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010], 
                        ),
                    ]), download=True)
        
        if train:
            num_val = int(val*len(ds))
            train_indices, val_indices = train_test_split(range(len(ds)), test_size=num_val, random_state=42)

            # generate subset based on indices
            train_ds = Subset(ds, train_indices)
            val_ds = Subset(ds, val_indices)
                            
            return train_ds, val_ds

        return test_ds


        
        

