import torch
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

def load_mnist():
    NUM_OF_TRAIN = 49000
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])

    mnist_train = dset.MNIST('./', train=True, download=True, transform=transform)
    loader_train = DataLoader(mnist_train, batch_size=64,
                            sampler=sampler.SubsetRandomSampler(range((NUM_OF_TRAIN))))
    
    mnist_val = dset.MNIST('./', train=True, download=True, transform=transform)
    loader_val = DataLoader(mnist_val, batch_size=64,
                            sampler=sampler.SubsetRandomSampler(range((NUM_OF_TRAIN, 50000))))
    
    mnist_test = dset.MNIST('./', train=False, download=True, transform=transform)
    loader_test = DataLoader(mnist_test, batch_size=64)

    return loader_train, loader_val, loader_test
