import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

from Model.VGG16 import VGG16Like
import Model.load_model as lm
import Dataset.load_data as ld
import train
from train import train as tt, check_accuracy

loader_train, loader_val, loader_test = ld.load_mnist()
model = VGG16Like()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print_every = 200
tt(model, optimizer,loader_train, loader_val, epochs=3)
check_accuracy(loader_test, model)
torch.save(model.state_dict(), './handWrittenModel.pkl')

