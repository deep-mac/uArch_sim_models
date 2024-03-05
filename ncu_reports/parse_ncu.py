import sys
import numpy as np
import csv

#from matplotlib import pyplot as plt


fields = ['kernel', 'grid', 'block', 'cycles', 'sm_util', 'tc_util', 'dram_util']

csvfile = open(sys.argv[1] +".csv", 'w')
csvwriter = csv.writer(csvfile)
csvwriter.writerow(fields)

def bucket(counts, val, cycles):
    bucket = int(val/10)
    counts[bucket] += cycles
    return counts

f1 = open(sys.argv[1])
lines = f1.readlines()
kernel_name = {'gemm':[], 'nchw':[], 'elem':[]}
cycles = {'gemm':[], 'nchw':[], 'elem':[]}
sm_util = {'gemm':[], 'nchw':[], 'elem':[]}
tc_util = {'gemm':[], 'nchw':[], 'elem':[]}
mem_util = {'gemm':[], 'nchw':[], 'elem':[]}
block = {'gemm':[], 'nchw':[], 'elem':[]}
grid = {'gemm':[], 'nchw':[], 'elem':[]}
#cycles = []
#sm_util = []
#tc_util = []
#mem_util = []
#block = []
#grid = []
for i, line in enumerate(lines):
    #print(line)
    if (line.lstrip().startswith("void") or line.lstrip().startswith("ampere") or line.lstrip().startswith("sm80")): #or line.lstrip().startswith("fmha")):
        if 'gemm' in line.lstrip() or 'cudnn_infer' in line.lstrip():
            type_ = 'gemm' 
        elif 'nchw' in line.lstrip() or 'nhwc' in line.lstrip():
            type_ = 'nchw'
        elif 'elementwise' in line.lstrip():
            type_ = 'elem'
        if type_ == 'gemm':
            if "cutlass" in line:
                kernel_name[type_].append(line.lstrip().split()[1])
            else:
                kernel_name[type_].append(line.lstrip().split()[0])
        else:
            kernel_name[type_].append(line.lstrip().split()[1])
       # print(type_)
    elif (line.lstrip().startswith("gpc__cycle")):
        cycles[type_].append(int(line.lstrip().replace(",", "").split()[-1]))
    elif (line.lstrip().startswith("sm__through")):
        sm_util[type_].append(float(line.lstrip().split()[-1]))
    elif (line.lstrip().startswith("sm__pipe_tensor")):
        tc_util[type_].append(float(line.lstrip().split()[-1]))
    elif (line.lstrip().startswith("gpu__dram")):
        mem_util[type_].append(float(line.lstrip().split()[-1] ))
    elif (line.lstrip().startswith("launch__block_size")):
        block[type_].append(int(line.lstrip().replace(",", "").split()[-1]))
    elif (line.lstrip().startswith("launch__grid_size")):
        grid[type_].append(int(line.lstrip().replace(",", "").split()[-1]))
#print(kernel_name)
#print(cycles)
#print(sm_util)
for type_ in ['gemm', 'nchw', 'elem']:
    for i, kernel_ in enumerate(kernel_name[type_]):
        row = [kernel_, grid[type_][i], block[type_][i], cycles[type_][i], sm_util[type_][i], tc_util[type_][i], mem_util[type_][i]]
        csvwriter.writerow(row)

total_cycles = np.sum(cycles['gemm']) + np.sum(cycles['nchw']) + np.sum(cycles['elem'])
print("Total cycles: ", total_cycles)
#print("GEMM stats\tCycles:", np.sum(cycles['gemm']), np.sum(cycles['gemm'])/total_cycles*100,"%",  "\tAvg. SM Util:", np.mean(sm_util['gemm']), "\tAvg TC Util:", np.mean(tc_util['gemm']), "\t Avg. Mem Util:", np.mean(mem_util['gemm']))
#print("nchw stats\tCycles:", np.sum(cycles['nchw']), np.sum(cycles['nchw'])/total_cycles*100,"%",  "\tAvg. SM Util:", np.mean(sm_util['nchw']), "\tAvg TC Util:", np.mean(tc_util['nchw']), "\t Avg. Mem Util:", np.mean(mem_util['nchw']))
#print("elem stats\tCycles:", np.sum(cycles['elem']), np.sum(cycles['elem'])/total_cycles*100,"%",  "\tAvg. SM Util:", np.mean(sm_util['elem']), "\tAvg TC Util:", np.mean(tc_util['elem']), "\t Avg. Mem Util:", np.mean(mem_util['elem']))
print(f'GEMM stats\tCycles: {np.sum(cycles["gemm"])}, Perc: {np.sum(cycles["gemm"])/total_cycles*100:.2f}%,\tAvg. SM Util:, {np.mean(sm_util["gemm"]):.2f},\tAvg TC Util: {np.mean(tc_util["gemm"]):.2f},\t Avg. Mem Util: {np.mean(mem_util["gemm"]):.2f}')
print(f'nchw stats\tCycles: {np.sum(cycles["nchw"])}, Perc: {np.sum(cycles["nchw"])/total_cycles*100:.2f}%,\tAvg. SM Util:, {np.mean(sm_util["nchw"]):.2f},\tAvg TC Util: {np.mean(tc_util["nchw"]):.2f},\t Avg. Mem Util: {np.mean(mem_util["nchw"]):.2f}')
print(f'elem stats\tCycles: {np.sum(cycles["elem"])}, Perc: {np.sum(cycles["elem"])/total_cycles*100:.2f}%,\tAvg. SM Util:, {np.mean(sm_util["elem"]):.2f},\tAvg TC Util: {np.mean(tc_util["elem"]):.2f},\t Avg. Mem Util: {np.mean(mem_util["elem"]):.2f}')
