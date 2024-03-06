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


#hyperparameter

NAME_FILE = "vgg16"

SAVE_TESTING = "vgg16"

OUTPUT_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), NAME_FILE)

TEST = False

len_epoch = 40

Learning_rate = 0.0001
Batch_size = 32

num_samples_subset = 50000

DATA_NOT_DOWNLOAD = False

SCHEDULER_ON = True

MIXUP_ON = False
MIXUP_SAMPLE = 15 #nombre d'epoch sur lesquelles faire le mixup

BINARIZATION = False

PRUNING_TEST = False

GLOBAL_PRUNING_TEST = False

FINE_TUNING_PRUNING = True
PRUNING_NBR_EPOCH = 9
PRUNING_PERCENTAGE = 0.1

VALIDATION_CYCLE_ON = True
Valid_cycle = 5 #how many epoch we want to go trought witouth using the validation test, and save the state too

PRINT_TRAINING_ACC_ON = True
Training_acc_cycle = Valid_cycle

## We set a seed manually so as to reproduce the results easily
seed  = 2147483647

global list_train 
global list_valid 
list_train = np.array([])
list_valid = np.array([])

#parameters

device = torch.device("cuda:0")
use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch VGG Training')
parser.add_argument('--lr', default=Learning_rate, type=float, help='learning rate')
parser.add_argument('--model', default="VGG16", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default=NAME_FILE, type=str, help='name of run')
parser.add_argument('--seed', default=seed, type=int, help='random seed')
parser.add_argument('--batch-size', default=Batch_size, type=int, help='batch size')
parser.add_argument('--epoch', default=len_epoch, type=int,
                    help='total epochs to run')
parser.add_argument('--VALIDATION_CYCLE_ON', default=VALIDATION_CYCLE_ON, type=bool)  
parser.add_argument('--Valid_cycle', default=Valid_cycle, type=int)  
parser.add_argument('--SCHEDULER_ON', default=SCHEDULER_ON, type=bool)        
parser.add_argument('--MIXUP_ON', default=MIXUP_ON, type=bool)   
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

c10train = CIFAR10(rootdir,train=True,download=DATA_NOT_DOWNLOAD,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=DATA_NOT_DOWNLOAD,transform=transform_test)

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
model = VGG(args.model)


model = model.to(device)
if BINARIZATION:
    model_bc = binaryconnect.BC(model)
    model_bc.model = model_bc.model.to(device)

if BINARIZATION:
    optimizer = optim.SGD(model_bc.model.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)
else:
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
    if BINARIZATION:
        outputs = model_bc.model(inputs)
    else:
        outputs = model(inputs)

    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

    
    if TEST :
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
    
    return loss,correct,total




def infer_classic(inputs,labels,correct,total,TEST):
    
    if BINARIZATION:
        model_bc.binarization()

    inputs,labels  = inputs.to(device),labels.to(device)
    if BINARIZATION:
        outputs = model_bc.model(inputs)
    else:
        outputs = model(inputs)
    outputs = outputs.to(device)
    loss = criterion(outputs, labels)

    if TEST:
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()

    return loss,correct,total




def m_train(OUTPUT_DIRECTORY):
    print("--- Enter in training mode ---")
    global list_train
    global list_valid
    list_train_acc,list_train_loss,list_valid_loss,list_valid_acc,list_lr = np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
    VALIDATION_TEST_ON = False
    reset_lr = SCHEDULER_ON
    MIXUP_LOSS_ON = False
    old_running_loss = 99999999

    for epoch in range(len_epoch if not FINE_TUNING_PRUNING else PRUNING_NBR_EPOCH):

        # Training set initialization
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()
        TEST = False
        
        if PRINT_TRAINING_ACC_ON and epoch % Training_acc_cycle == 0:
            TEST = True
        
         # to ensure the old loss << new loss             

        #training set infer
        for batch_index, (inputs,labels) in enumerate(trainloader_subset):


            if MIXUP_LOSS_ON and MIXUP_ON:

                if reset_lr:
                    optimizer.param_groups[0]["lr"] = args.lr
                    reset_lr = False
                loss,correct,total = infer_mixup(inputs,labels,correct,total,TEST)
                
            else :
                loss,correct,total = infer_classic(inputs,labels,correct,total,TEST)

            
            optimizer.zero_grad()
            loss.backward()

            if BINARIZATION:
                model_bc.restore()

            optimizer.step()  

            if BINARIZATION:  
                model_bc.clip()

            # statistics
            running_loss += loss.item()
        
        params = ()
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                params += ((module, "weight"),)
                
        if FINE_TUNING_PRUNING:

            '''
            params = ()
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    params += ((module, "weight"),)
            '''
            prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=PRUNING_PERCENTAGE)
                
                # for name, module in model.named_modules():
                #     if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                #         prune.remove(module, "weight")



        #get and print loss
        print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_index + 1, running_loss / len(trainloader_subset)))
        
        #scheduler
        if SCHEDULER_ON:
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step(running_loss)
            #print lr
            print("lrs = ",lrs[-1])

        
        #get and print acc
        if TEST:
            print('[%d, %5d] acc_training: %.3f' % (epoch + 1, batch_index + 1, 100.*correct/total))
            list_train_acc = np.append(list_train_acc,100.*correct/total)
            list_train_loss = np.append(list_train_loss,running_loss/len(trainloader_subset))
            if SCHEDULER_ON:
                list_lr = np.append(list_lr,lrs[-1])
        
        
        if VALIDATION_CYCLE_ON and epoch % Valid_cycle == 0:
            VALIDATION_TEST_ON = True


        if VALIDATION_TEST_ON:
            #validation test
            running_loss_valid  = 0
            correct = 0
            total = 0

            if BINARIZATION:
                model_bc.model.eval()
            else:
                model.eval()
            TEST = True

            #validation infer
            for batch_index, (inputs,labels) in enumerate(testloader):

                loss_valid,correct,total = infer_classic(inputs,labels,correct,total,TEST)
                    
                
                #accumulation 
                running_loss_valid += loss_valid.item()
            #save state 
            if BINARIZATION:
                torch.save(model_bc.model, OUTPUT_DIRECTORY +'/save_model/save_model_state_'+ str(epoch) +'.pth')
            else:
                torch.save(model, OUTPUT_DIRECTORY +'/save_model/save_model_state_'+ str(epoch) +'.pth')


            #save stat 
            list_valid_acc = np.append(list_valid_acc,100.*correct/total) #save accurancy batch
            list_valid_loss = np.append(list_valid_loss,running_loss_valid/len(testloader))
            #print acc
            print('[%d, %5d] loss_valid: %.3f' % (epoch + 1, batch_index + 1, running_loss_valid / len(testloader)))
            print('[%d, %5d] acc_valid: %.3f' % (epoch + 1, batch_index + 1, 100.*correct/total)) 

            if VALIDATION_CYCLE_ON:
                VALIDATION_TEST_ON = False
        
        # condition to start mixing
        if MIXUP_ON and not(MIXUP_LOSS_ON):
            if  epoch > len_epoch - MIXUP_SAMPLE :
                MIXUP_LOSS_ON = True
                print("--- Starting MIXUP ---")
    


    # building final list to return
    list_train = np.concatenate((np.array([list_train_loss]),np.array([list_train_acc])),axis=0)
    if SCHEDULER_ON:
        list_train = np.concatenate((list_train,np.array([list_lr])),axis=0)
    list_valid = np.concatenate((np.array([list_valid_loss]),np.array([list_valid_acc])),axis=0)




def print_result(OUTPUT_DIRECTORY): 
    print("--- Enter print result ---")
    train_value = np.load(OUTPUT_DIRECTORY + '/train_values.npy')
    test_value = np.load(OUTPUT_DIRECTORY + '/valid_values.npy')
    train_acc,train_loss = train_value[1],train_value[0]
    test_acc,test_loss = test_value[1],test_value[0]
    objects = []

    with open(OUTPUT_DIRECTORY + '/parameters.pkl', 'rb') as f:
        parameters = pickle.load(f)
    
    nbr_epoch = np.arange(0,parameters['epoch'],parameters['Valid_cycle'])

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)

    ax1.plot(nbr_epoch,train_acc,label = 'train')
    ax1.plot(nbr_epoch,test_acc,label = 'test')
    ax1.set_xlim(0, parameters['epoch'])
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(nbr_epoch,train_loss,label = 'train')
    ax2.plot(nbr_epoch,test_loss,label = 'test')
    ax2.set_xlim(0, parameters['epoch'])
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.grid(True)
    ax2.legend()

    plt.savefig(OUTPUT_DIRECTORY + "/result_model.png")

    if parameters['SCHEDULER_ON']:
        fig, ax = plt.subplots()
        ax.plot(nbr_epoch,np.log10(train_value[2]),label = 'train')
        ax.set_xlabel('epoch')
        ax.set_ylabel('learning rate log scale')
        ax.grid(True)
        ax.legend()
        plt.savefig(OUTPUT_DIRECTORY + "/result_learning_rate.png")

    print("--- Graph Result save ---")


def m_save(time_execution,OUTPUT_DIRECTORY):
    global list_train
    global list_valid

    # saving as a dict the parameter
    parameters = vars(args)
    parameters['time_training_min'] = time_execution
    with open(OUTPUT_DIRECTORY+'/parameters.pkl', 'wb') as fp:
        pickle.dump(parameters, fp)

    if BINARIZATION:
        torch.save(model_bc.model, OUTPUT_DIRECTORY +'/save_model/save_model_state_final.pth')
    else:
        torch.save(model, OUTPUT_DIRECTORY +'/save_model/save_model_state_final.pth')
    np.save( OUTPUT_DIRECTORY + '/train_values.npy', list_train)
    np.save( OUTPUT_DIRECTORY + '/valid_values.npy', list_valid)


def m_test (model, i=0):
    print("Entering test mode")

    if PRUNING_TEST : 
        print("---entering pruning---")
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=i)

    if GLOBAL_PRUNING_TEST :
        params = ()
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                params += ((module, "weight"),)
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=i)

    model.eval()
    TEST = True
    
    #initialization test
    correct = 0
    total = 0   
    running_loss  = 0    
    # infer test    
    for batch_index, (inputs,labels) in enumerate(testloader):
        
        # inputs=inputs.half()
        loss,correct,total = infer_classic(inputs,labels,correct,total,TEST)

        running_loss+= loss.item() 
        
    return 100.*correct/total,running_loss/len(testloader)


############################### MAIN ##########################################################

if TEST:
    model = torch.load(OUTPUT_DIRECTORY + '/save_model/save_model_state_'+SAVE_TESTING+'.pth')
    acc,loss = m_test(model)
    print('acc =',acc.numpy())

'''

if TEST :
    list_acc=[]
    for i in range(20):
        model = torch.load(OUTPUT_DIRECTORY + '/save_model/save_model_state_'+SAVE_TESTING+'.pth')
        acc,loss = m_test(model, i/20)
        list_acc.append(acc)
        print('acc =',acc.numpy())
    plt.plot([x*5 for x in range(20)], list_acc)
    plt.ylabel("% precision")
    plt.xlabel("pruning %")
    plt.grid(True)
    plt.savefig(OUTPUT_DIRECTORY + "/pruning_acc.png")
'''


if FINE_TUNING_PRUNING:
    model = torch.load(OUTPUT_DIRECTORY + '/save_model/save_model_state_'+SAVE_TESTING+'.pth')
    m_train(OUTPUT_DIRECTORY)


if not TEST and not FINE_TUNING_PRUNING:

    # check if the path already exist and create a new directory
    version = 0
    OUTPUT_DIRECTORY_TEST = OUTPUT_DIRECTORY
    while(os.path.exists(OUTPUT_DIRECTORY_TEST)):
        if(version > 0):
            OUTPUT_DIRECTORY_TEST = OUTPUT_DIRECTORY 
        version += 1
        OUTPUT_DIRECTORY_TEST = OUTPUT_DIRECTORY_TEST +"_v" + str(version)
        
    OUTPUT_DIRECTORY = OUTPUT_DIRECTORY_TEST

    os.mkdir(OUTPUT_DIRECTORY)
    os.mkdir(OUTPUT_DIRECTORY+ "/save_model")

    #Start infer
    start_time = time.time()
    m_train(OUTPUT_DIRECTORY)
    time_execution = (time.time() - start_time)/60
    print("--- Train in %s minutes ---" % time_execution)
    m_save(time_execution,OUTPUT_DIRECTORY)
    print("--- Save completed ---")
    
    print_result(OUTPUT_DIRECTORY)
