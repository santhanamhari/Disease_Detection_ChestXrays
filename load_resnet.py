#!/usr/bin/python3

import argparse
import os
import shutil
import time
import gc

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import sys

from model import resnet50

CKPT = 'checkpoint.pth.tar'

model = resnet50(None)

# Use .cuda() only on machine with GPU available!
# model = torch.nn.DataParallel(model).cuda()
model = torch.nn.DataParallel(model)

print("=> loading checkpoint '{}'".format(CKPT))
checkpoint = torch.load(CKPT, map_location='cpu')
start_epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['state_dict'])
print(checkpoint['state_dict']  )
print("=> loaded checkpoint '{}' (epoch {})"
.format(CKPT, checkpoint['epoch']))


with torch.no_grad():
    x = torch.randn(1,3,224,224)
    y = model(x)
