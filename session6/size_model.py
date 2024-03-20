
import numpy as np 
import torch 
import torch.nn as nn
import os

SAVE_TESTING = "final"
NAME_FILE = "densenet_mini_ref_fact"
OUTPUT_DIRECTORY  = os.path.join(os.path.dirname(os.path.realpath(__file__)), NAME_FILE)
model_file = os.path.join(OUTPUT_DIRECTORY, "save_model/save_model_state_" + SAVE_TESTING + ".pth")

# model_file = "/homes/e21jamet/efficient-deep-learning/Grp7_efficient_deep_learning/session6/densenet_mini_fact_test_v1/save_model/save_model_state_final.pth"
model = torch.load(model_file, map_location=torch.device('cpu'))

sparsity = 0
for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            sparsity += torch.sum(module.weight == 0)

def count_parameters(model):
    return sum(p[torch.abs(p)>10**-15].numel() for p in model.parameters() if( p.requires_grad))

number_parameter = count_parameters(model)

print("Number of parameters : ", number_parameter , " or in log scale :", np.log10(number_parameter))

print("Sparsity in model: {:.2f}%".format((100. * sparsity)/ number_parameter))




weight_model = os.path.getsize(model_file)
print("Weight of the model :",weight_model,"Bytes")
