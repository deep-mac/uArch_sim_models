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

in_dim = 50
h_dim = 128
o_dim = 32
seq_len = 128
n_layer = 2
B = 4096# N
device = 'cuda:0'

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, out_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True)
        print('LSTM layer ', sum(p.numel() for p in self.lstm.parameters())/1e3, 'K parameters')

        self.out= nn.Linear(hidden_dim, out_dim)
        print('Linear layer ', sum(p.numel() for p in self.out.parameters())/1e3, 'K parameters')

    def forward(self, x):
        lstm_out, hidden = self.lstm(x)
        out = self.out(hidden[0][-1])
        return out

data3 = torch.rand(B, seq_len, in_dim, dtype=torch.float16)
print(data3.shape)
data3 = data3.to(device)
lstm = LSTM(in_dim, h_dim, o_dim)
lstm = lstm.half().to(device)
print('LSTM model', sum(p.numel() for p in lstm.parameters())/1e3, 'K parameters')

t0 = time.perf_counter()
_cudart.cudaProfilerStart()
for i in range(int(sys.argv[1])):
    y = lstm(data3)
_cudart.cudaProfilerStop()
torch.cuda.synchronize()
t1 = time.perf_counter()
print(y.shape)
print(f'Elapsed time: {t1 - t0:.2f} seconds')
#print(f'TFLOP/s = {TFLOP/(t1-t0)}')

