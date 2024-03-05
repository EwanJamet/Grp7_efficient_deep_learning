import numpy as np
import torch
import os
from vgg import *

model_file = "D:/Users/cowan/Desktop/COURS_IMT/Semestre_u/Deep_Learning/Grp7_efficient_deep_learning/Size_model/save_model_state_final.pth"

model = torch.load(model_file, map_location=torch.device('cpu'))

def count_parameters(model):
    return sum(p[torch.abs(p)>0.01].numel() for p in model.parameters() if( p.requires_grad))

print(count_parameters(model))