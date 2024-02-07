from torchvision.datasets import CIFAR10
import numpy as np 
import torchvision.transforms as transforms
import torch 
from torch.utils.data.dataloader import DataLoader
from vgg import *
import torch.nn as nn
import torch.optim as optim

#parameters
len_epoch = 100



## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network. 
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

### The data from CIFAR10 will be downloaded in the following folder
rootdir = './data/cifar10'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

trainloader = DataLoader(c10train,batch_size=32,shuffle=True) #[1500,32,10,10]
testloader = DataLoader(c10test,batch_size=32) 

## number of target samples for the final dataset
num_train_examples = len(c10train)
num_samples_subset = 15000

## We set a seed manually so as to reproduce the results easily
seed  = 2147483647

## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
indices = list(range(num_train_examples))
np.random.RandomState(seed=seed).shuffle(indices)## modifies the list in place

## We define the Subset using the generated indices 
c10train_subset = torch.utils.data.Subset(c10train,indices[:num_samples_subset])
print(f"Initial CIFAR10 dataset has {len(c10train)} samples")
print(f"Subset of CIFAR10 dataset has {len(c10train_subset)} samples")

# Finally we can define anoter dataloader for the training data
trainloader_subset = DataLoader(c10train_subset,batch_size=32,shuffle=True)
### You can now use either trainloader (full CIFAR10) or trainloader_subset (subset of CIFAR10) to train your networks.

data = trainloader_subset 
criterion = nn.CrossEntropyLoss()
net = VGG('VGG11')

# for test_images, test_labels in trainloader_subset:  
#     sample_image = test_images[0]    # Reshape them according to your needs.
#     sample_label = test_labels[0]

# print(sample_image)
# print(sample_label)


for epoch in range(len_epoch):


    running_loss = 0.0

    for i, data in enumerate(trainloader_subset):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # # zero the parameter gradients
        # optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
        # print("output init=",outputs)

        # labels = labels.float()
        # outputs = torch.argmax(outputs,dim=1).float()

        # print("output=",outputs)
        # print("labels=",labels)

        loss = criterion(outputs, labels)
        loss.backward()
        # optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
