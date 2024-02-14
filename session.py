from torchvision.datasets import CIFAR10
import numpy as np 
import torchvision.transforms as transforms
import torch 
from torch.utils.data.dataloader import DataLoader
from vgg import *
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse

#hyperparameter

TEST = False
# TEST = True

len_epoch = 100
len_epoch_test = 100

Learning_rate = 1
Batch_size = 32

num_samples_subset = 50000

NAME_FILE = "VG11_50000_100epoch_scheduler"

SCHEDULER_ON = True

VALIDATION_CYCLE_ON = True
Valid_cycle = 4 #how many epoch we want to go trought witouth using the validation test

## We set a seed manually so as to reproduce the results easily
seed  = 2147483647

global list_train 
global list_valid 
list_train = np.array([])
list_valid = np.array([])

#parameters

device = torch.device("cuda:0")
use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=Learning_rate, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="VGG11", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=Batch_size, type=int, help='batch size')
parser.add_argument('--epoch', default=len_epoch, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
args = parser.parse_args()


## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the modelwork. 
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

trainloader = DataLoader(c10train,batch_size=Batch_size,shuffle=True) 
testloader = DataLoader(c10test,batch_size=Batch_size) 

## number of target samples for the final dataset
num_train_examples = len(c10train)


## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
indices = list(range(num_train_examples))
np.random.RandomState(seed=seed).shuffle(indices)## modifies the list in place

## We define the Subset using the generated indices 
c10train_subset = torch.utils.data.Subset(c10train,indices[:num_samples_subset])
print(f"Initial CIFAR10 dataset has {len(c10train)} samples")
print(f"Subset of CIFAR10 dataset has {len(c10train_subset)} samples")

# Finally we can define anoter dataloader for the training data
trainloader_subset = DataLoader(c10train_subset,batch_size=32,shuffle=True)
### You can now use either trainloader (full CIFAR10) or trainloader_subset (subset of CIFAR10) to train your modelworks.

data = trainloader_subset 
criterion = nn.CrossEntropyLoss()
model = VGG('VGG11')
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)

#scheduler

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=args.lr,step_size_up=20,mode="triangular2")
lrs = []



def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = (lam * x + (1 - lam) * x[index, :])
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



def infer_mixup(inputs,labels,correct,total,TEST):
    
    inputs, labels = inputs.cuda(), labels.cuda()

    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels,args.alpha, use_cuda)
    inputs, targets_a, targets_b = map(Variable, (inputs,targets_a, targets_b))
    outputs = model(inputs)

    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

    
    if TEST :
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        return loss,correct,total
    else :
        return loss



def infer_classic(inputs,labels,correct,total,TEST):
    
    inputs,labels  = inputs.to(device),labels.to(device)
    outputs = model(inputs)
    outputs = outputs.to(device)
    loss = criterion(outputs, labels)

    if TEST:
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()

        return loss,correct,total
    else:
        return loss



def m_train():
    print("Enter in training mode")
    global list_train
    global list_valid
    VALIDATION_TEST_ON = False
    reset_lr = SCHEDULER_ON

    for epoch in range(len_epoch):

        # Training set initialization
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()
        TEST = False
        
        #training set infer
        for batch_index, (inputs,labels) in enumerate(trainloader_subset):
            
            if epoch < len_epoch/2 :
                loss = infer_classic(inputs,labels,correct,total,TEST)
            else :
                if reset_lr:
                    optimizer.param_groups[0]["lr"] = 0.1
                    reset_lr = False
                loss = infer_mixup(inputs,labels,correct,total,TEST)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    

            # statistics
            running_loss += loss.item()
        
        #get and print loss
        print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_index + 1, running_loss / len(trainloader_subset)))
        list_train = np.append(list_train,running_loss/len(trainloader_subset))


        #scheduler
        if SCHEDULER_ON:
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step(running_loss)
            #print lr
            print("lrs = ",lrs[-1])
        
        if VALIDATION_CYCLE_ON and epoch % Valid_cycle == 0:
            VALIDATION_TEST_ON = True


        if VALIDATION_TEST_ON:
            #validation test
            running_acc_valid  = 0
            running_loss_valid  = 0
            correct = 0
            total = 0

            model.eval()
            TEST = True

            #validation infer
            for batch_index, (inputs,labels) in enumerate(testloader):
                if epoch < len_epoch/2 :
                    loss_valid,correct,total = infer_classic(inputs,labels,correct,total,TEST)
                    
                else :
                    loss_valid,correct,total = infer_mixup(inputs,labels,correct,total,TEST)
                
                #accumulation 
                running_loss_valid += loss_valid.item()
 
            
            #save stat 
            list_valid = np.append(list_valid,100.*correct/total) #save accurancy batch
            list_valid = np.append(list_valid,running_loss_valid/len(testloader))
            #print acc
            print('[%d, %5d] acc_valid: %.3f' % (epoch + 1, batch_index + 1, 100.*correct/total)) 

            if VALIDATION_CYCLE_ON:
                VALIDATION_TEST_ON = False
    

def m_test (model):
    print("Entering test mode")
    model.eval()
    TEST = True
    mean_batch_acc = 0
    mean_batch_loss = 0
    for epoch in range(len_epoch_test):
        #initialization test
        correct = 0
        total = 0   
        running_acc  = 0
        running_loss  = 0    
        #infer test    
        for batch_index, (inputs,labels) in enumerate(testloader):

            loss,correct,total = infer_classic(inputs,labels,correct,total,TEST)
   
            running_loss+= loss.item()   
        #calcul mean batch
        
        mean_batch_loss += running_loss/len(testloader)
        mean_batch_acc += 100.*correct/total
        
        print("Still",len_epoch_test-epoch,"epoch(s) to proceed.")

    
    return mean_batch_acc/len_epoch_test,mean_batch_loss/len_epoch_test



def m_save():
    global list_train
    global list_valid
    torch.save(model, '/homes/e21jamet/Documents/efficient-deep-learning/models_cifar100/'+ NAME_FILE +'.pth')
    np.save( '/homes/e21jamet/Documents/efficient-deep-learning/models_cifar100/'+ NAME_FILE +'_train_values.npy', list_train)
    np.save( '/homes/e21jamet/Documents/efficient-deep-learning/models_cifar100/'+ NAME_FILE +'_valid_values.npy', list_valid)


if not TEST:
    m_train()
    m_save()

if TEST :
    model = torch.load('/homes/e21jamet/Documents/efficient-deep-learning/models_cifar100/'+ NAME_FILE +'.pth')
    acc,loss = m_test(model)
    print('acc =',acc.numpy())

