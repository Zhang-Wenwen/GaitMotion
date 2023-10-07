import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pickle, os
import utils, re
import matplotlib.pyplot as plt



def Step_seg_zero_step(seq_buffer, data, seg_index, self, labels, file):
    # this will separate acc and gyro data into step by step segment.
    seq_buffer = seq_buffer
    diff_up = np.where(np.diff(data[:, -2], axis=0) == 1)
    diff_down = np.where(np.diff(data[:, -2], axis=0) == -1)
    diff_down = diff_down[0][1:]  # the first down is from the ground truth,
    seg = seg_index 
    diff_up = diff_up[0]  

    # convert from degrees to radian
    data[:, 3:6] = np.radians(data[:, 3:6])
    
    for i in np.arange(1,len(seg)-1,1):   # IF WE DROP THE FIRST AND LAST STEP
    #for i in np.arange(0,len(seg),1):   # IF WE DON'T DROP THE FIRST AND LAST STEP
        acc = data[diff_up[seg[i]] - int(seq_buffer / 2):diff_down[seg[i]] + int(seq_buffer / 2), 0:3]
        gyro = data[diff_up[seg[i]] - int(seq_buffer / 2):diff_down[seg[i]] + int(seq_buffer / 2), 3:6]
        if len(acc) < 50:
            print("pulse step in " + file)
            continue
        if len(acc) > self.seq_length:
            cutting_size = len(acc) - self.seq_length
            cutting_size_l = int(cutting_size / 2)  # index of start date and end date
            cutting_size_r = cutting_size - cutting_size_l
            acc = acc[cutting_size_l:len(acc) - cutting_size_r, :]
            gyro = gyro[cutting_size_l:len(gyro) - cutting_size_r, :]
        else:
            padding_size_r = self.seq_length - len(acc)
            padding_size_l = int(padding_size_r / 2)
            # number of data points of padding size for left and right
            padding_size_r = padding_size_r - padding_size_l
            acc = np.pad(acc, [(padding_size_l,padding_size_r), (0,0)],'reflect')
            gyro = np.pad(gyro, [(padding_size_l,padding_size_r), (0,0)], 'reflect')

        self.labels.append(labels[i])
        self.data.append(np.concatenate((acc,gyro), axis=1))

class GaitDataset(Dataset):
    def __init__(self, filenames, batch_size, seq_length,seq_buffer, transform, rate=1000, testing=0, mtype="Normal",testing_with_discard=0):
        # `filenames` is a list of strings that contains all file names.
        # `batch_size` determines the number of files that we want to read in a chunk.
        #  self.data -> acc, gyro and labels. ground truth and ZUPT results.
        #  self.labels -> ground truth of stride length from GaitRite system
        self.filenames = filenames
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.transform = transform
        self.seq_buffer = seq_buffer
        self.data = []
        self.labels = []
        self.subject_dict = pd.DataFrame({}) #empty dict to save extra info 
        self.rate = rate

        init_step=0
        for file in self.filenames:
            with open(file, 'rb') as f:  
                x, y, seg_index = pd.read_pickle(f)
            for i in np.arange(6):
                x[:, i] = utils.butter_lowpass_filter(x[:, i], 20, self.rate, order=3)

            x = x[200:, :] 

            # the start index of matlab is 1, while python starts from 0. Adjust matlab to python here.
            seg_index = np.arange(np.where(np.diff(x[:, -2], axis=0) == 1)[0].size - 1)

            labels = y['Stride_length'][1:].values / 100

            # segment the data into steps
            Step_seg_zero_step(self.seq_buffer,x, seg_index, self, labels, file)

            # record the subject ID and the walking type for each step
            if testing:
                if testing_with_discard:   # if discard the fisrt and last step.
                    self.subject_dict = utils.add_extra_info(file,init_step,seg_index[-1]+init_step-2,self.subject_dict)
                    init_step += len(seg_index)-2
                else:
                    self.subject_dict = utils.add_extra_info(file,init_step,seg_index[-1]+init_step,self.subject_dict)
                    init_step += len(seg_index)
        print("finish loading, start training")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):  # index means index of the chunk.
        label = self.labels[index].astype(float).reshape(-1, 1)
        data = self.data[index].astype(float)
        return data, label
