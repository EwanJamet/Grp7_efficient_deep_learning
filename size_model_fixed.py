from torchvision.datasets import CIFAR10
import numpy as np 
import torchvision.transforms as transforms
import torch 
from torch.utils.data.dataloader import DataLoader
from vgg import *
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import argparse
import os
import pickle
import time
import matplotlib.pyplot as plt
import binaryconnect 


model_file = "/homes/r21vassa/Documents/eff_dl/vgg16/save_model/save_model_state_final.pth"
model = torch.load(model_file, map_location=torch.device('cpu'))

s = 0
for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            s += torch.sum(module.weight == 0)

def count_parameters(model):
    return sum(p[torch.abs(p)>10**-15].numel() for p in model.parameters() if( p.requires_grad))

c = count_parameters(model)

print(c)

print("Sparsity in model: {:.2f}%".format((100. * s)/ c))
