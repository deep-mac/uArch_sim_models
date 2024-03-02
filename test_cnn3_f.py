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


#CNN3_F(3,2,128,2,1,2,256,2,0,2,256,2,0,1024)
#Based on simnet model in https://github.com/lingda-li/simnet/tree/master/ml
class CNN3_F(nn.Module):
    def __init__(self, out, ck1, ch1, cs1, cp1, ck2, ch2, cs2, cp2, ck3, ch3, cs3, cp3, f1):
        super(CNN3_F, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=inst_length, out_channels=ch1, kernel_size=ck1, stride=cs1, padding=cp1)
        #self.conv1 = nn.Conv1d(in_channels=inst_length, out_channels=ch1, kernel_size=ck1)
        print('CNN model conv1', sum(p.numel() for p in self.conv1.parameters())/1e3, 'K parameters')
        self.conv2 = nn.Conv1d(in_channels=ch1, out_channels=ch2, kernel_size=ck2, stride=cs2, padding=cp2)
        print('CNN model conv2', sum(p.numel() for p in self.conv2.parameters())/1e3, 'K parameters')
        self.conv3 = nn.Conv1d(in_channels=ch2, out_channels=ch3, kernel_size=ck3, stride=cs3, padding=cp3)
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

        x = x.view(-1, inst_length, context_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.f1_input)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        #check without activation
        #x = x.view(-1, inst_length, context_length)
        #x = self.conv1(x)
        #x = self.conv2(x)
        #x = self.conv3(x)
        #x = x.view(-1, self.f1_input)
        #x = self.fc1(x)
        #x = self.fc2(x)
        return x


data3 = torch.rand(B, 50, 111, dtype=torch.float16)
print(data3.shape)
data3 = data3.to(device)
cnn = CNN3_F(3,2,128,2,1,2,256,2,0,2,256,2,0,1024)
cnn = cnn.half().to(device)
print('CNN model', sum(p.numel() for p in cnn.parameters())/1e3, 'K parameters')

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

