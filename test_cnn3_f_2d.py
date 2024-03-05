import torch
import os
import sys
import time
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
import ctypes
_cudart = ctypes.CDLL('libcudart.so')

inst_length = 50
context_length = 111
B = 4096# N
device = 'cuda:4'

ck1 = 2
ck2 = ck1
ck3 = ck2

ch0 = inst_length
ch1 = 128
ch2 = 256
ch3 = 256

cs1 = 2
cs2 = cs1
cs3 = cs2

cp1 = 1
cp2 = 0
cp3 = 0

fc1_dim = 1024

o_dim = 3

#CNN3_F(3,2,128,2,1,2,256,2,0,2,256,2,0,1024)
#Based on simnet model in https://github.com/lingda-li/simnet/tree/master/ml
class CNN3_F(nn.Module):
    def __init__(self, out, ck1, ch1, cs1, cp1, ck2, ch2, cs2, cp2, ck3, ch3, cs3, cp3, f1):
        super(CNN3_F, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inst_length, out_channels=ch1, kernel_size=(1, ck1), stride=(1, cs1), padding=(0, cp1))
        print('CNN model conv1', sum(p.numel() for p in self.conv1.parameters())/1e3, 'K parameters')
        self.conv2 = nn.Conv2d(in_channels=ch1, out_channels=ch2, kernel_size=(1, ck2), stride=(1, cs2), padding=(0, cp2))
        print('CNN model conv2', sum(p.numel() for p in self.conv2.parameters())/1e3, 'K parameters')
        self.conv3 = nn.Conv2d(in_channels=ch2, out_channels=ch3, kernel_size=(1, ck3), stride=(1, cs3), padding=(0, cp3))
        print('CNN model conv3', sum(p.numel() for p in self.conv3.parameters())/1e3, 'K parameters')
        self.f1_input = math.floor((context_length + 2 * cp1 - ck1) / cs1 + 1)
        print(self.f1_input)
        self.f1_input = math.floor((self.f1_input + 2 * cp2 - ck2) / cs2 + 1)
        print(self.f1_input)
        self.f1_input = math.floor((self.f1_input + 2 * cp3 - ck3) / cs3 + 1)
        self.f1_input *= ch3
        self.f1_input = int(self.f1_input)
        print(self.f1_input)
        self.fc1 = nn.Linear(self.f1_input, f1)
        print('CNN model fc1', sum(p.numel() for p in self.fc1.parameters())/1e3, 'K parameters')
        self.fc2 = nn.Linear(f1, out)
        print('CNN model fc2', sum(p.numel() for p in self.fc2.parameters())/1e3, 'K parameters')

    def forward(self, x):

        #x = x.view(-1, inst_length, context_length)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = x.reshape(-1, self.f1_input)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)

        #check without activation
        #x = x.view(-1, inst_length, context_length)
        #x = self.conv1(x)
        #x = self.conv2(x)
        #x = self.conv3(x)
        #x = x.view(-1, self.f1_input)
        #x = self.fc1(x)
        #x = self.fc2(x)
        return x


cnn = CNN3_F(o_dim,ck1,ch1,cs1,cp1,ck2,ch2,cs2,cp2,ck3,ch3,cs3,cp3,fc1_dim)
cnn = cnn.half().to(device)
print('CNN model', sum(p.numel() for p in cnn.parameters())/1e3, 'K parameters')

data3 = torch.rand(B, 50, 1, 111, dtype=torch.float16)
data3 = data3.to(device)
print('Data shape', data3.shape)

if (len(sys.argv) == 3):
    data3 = data3.to(memory_format=torch.channels_last)
    cnn = cnn.to(memory_format=torch.channels_last)

t0 = time.perf_counter()
_cudart.cudaProfilerStart()
for i in range(int(sys.argv[1])):
    y = cnn(data3)
_cudart.cudaProfilerStop()
torch.cuda.synchronize()
t1 = time.perf_counter()
print(y.shape)
print(f'Elapsed time: {t1 - t0:.2f} seconds')
#print(f'TFLOP/s = {TFLOP/(t1-t0)}')

