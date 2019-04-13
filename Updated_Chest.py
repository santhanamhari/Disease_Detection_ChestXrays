'''Pre-trained Resnet-50'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import sys

import os
import argparse
import shutil
import time
import gc

# pre-trained model
from model import resnet50
#from models import *
#from utils import progress_bar


parser = argparse.ArgumentParser(description='Resnet 50 Pre-trained Training')
parser.add_argument('disease_num',type=int,  help ='type of disease used')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc_train = 0  # best training accuracy
best_acc_valid = 0 # best validation accuracy
best_acc_test = 0 # best testing accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Confusion Matrix Values                                                                                                  
best_tp = 0 
best_tn = 0
best_fp = 0
best_fn = 0

num = int(sys.argv[3])
disease = ('Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax') 

# display what test it is in the slurm file
print("Resnet Sparse")
print("Healthy vs. " + disease[num])

# directories
traindir = '/scratch/gpfs/hs12/Full_Data/binary_class/Healthy-' + disease[num] + '/train'
validdir = '/scratch/gpfs/hs12/Full_Data/binary_class/Healthy-' + disease[num] + '/validation'
testdir = '/scratch/gpfs/hs12/Full_Data/binary_class/Healthy-' + disease[num] + '/test'

# Data
print('==> Preparing data..')
train_dataset = torchvision.datasets.ImageFolder(traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),  #Resolution is 1024x1024, gray
#        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
 ]))

valid_dataset = torchvision.datasets.ImageFolder(validdir,
    transforms.Compose([
    transforms.Resize(256), #Resolution is 1024x1024, gray                                        
    transforms.CenterCrop(224),
#    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
]))

test_dataset = torchvision.datasets.ImageFolder(testdir,
    transforms.Compose([
    transforms.Resize(256), #Resolution is 1024x1024, gray                                                                              
    transforms.CenterCrop(224),
#    transforms.RandomHorizontalFlip(),                                                                                                           
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
]))

#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=4)

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# classes = ('Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax')
classes = (disease[num], 'No Finding')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
#net = ShuffleNetV2(1)
net = resnet50(None)


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global best_acc_train
    net.train()
    train_loss = 0
    correct_train = 0
    total_train = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total_train += targets.size(0)
        correct_train += predicted.eq(targets).sum().item()

    acc_train = 100.*correct_train/total_train
    if acc_train > best_acc_train:
      # print('Training Accuracy: ', acc_train)
        state = {
            'net': net.state_dict(),
            'acc': acc_train,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc_train = acc_train

        print('Training Accuracy: ', best_acc_train)

# Validation - used as testing set 
def validation(epoch):
    global best_acc_valid
    net.eval()
    valid_loss = 0
    correct_valid = 0
    total_valid = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total_valid += targets.size(0)
            correct_valid += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc_valid = 100.*correct_valid/total_valid
    if acc_valid > best_acc_valid:
      # print('Validation Accuracy: ', acc_valid)
        state = {
            'net': net.state_dict(),
            'acc': acc_valid,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc_valid = acc_valid
    
        print('Validation Accuracy: ', best_acc_valid)


def test(epoch):
    global best_acc_test
    global best_tp
    global best_tn
    global best_fp
    global best_fn
    
    net.eval()
    test_loss = 0
    correct_test = 0
    total_test = 0
    tp = 0
    tn = 0
    fp = 0 
    fn = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total_test += targets.size(0)
            correct_test += predicted.eq(targets).sum().item()
        
            # Confusion matrix calcs
            for i in range(len(outputs)):
                if (classes[predicted[i]] == classes[targets[i]]):
                    if (classes[predicted[i]] == classes[0]):
                        tp = tp + 1 #both are Disease
                    else:
                        tn = tn + 1 #both are Healthy
                else: 
                    if (classes[predicted[i]] == classes[0]):
                        fp = fp + 1 #predicted Disease, actually Healthy
                    else: 
                        fn = fn + 1 #predicted Healthy, actually Disease
            best_tp = tp
            best_tn = tn
            best_fp = fp
            best_fn = fn

    # Save checkpoint.                                                                                                               
    acc_test = 100.*correct_test/total_test
    if acc_test > best_acc_test:
      # print('Test Accuracy: ', acc_test)                                                                                                                                                         
        state = {
            'net': net.state_dict(),
            'acc': acc_test,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc_test = acc_test

        print('Test Accuracy: ', best_acc_test)


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    validation(epoch)
    test(epoch)

print('Final Training Accuracy: ', best_acc_train)
print('Final Validation Accuracy: ', best_acc_valid)
print('Final Testing Accuracy: ', best_acc_test) 

print ('TP: ', best_tp)
print ('TN: ', best_tn)
print ('FP: ', best_fp)
print ('FN: ', best_fn)
