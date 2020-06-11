import torch
import torchvision

def load_model():
    model = torchvision.models.resnet50(pretrained=True)
    return model