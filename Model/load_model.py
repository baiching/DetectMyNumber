import torch
import torch.nn as nn
import torchvision

def load_model():
    model = torchvision.models.resnet50(pretrained=True)
    return model
