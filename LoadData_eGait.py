import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pickle, os
import utils, re
import matplotlib.pyplot as plt
from scipy import interpolate

def get_Gaitrite(filename):
    df = pd.read_csv("eGait/GoldStandard_GaitRite/"+filename+'txt',
                        delimiter=",", skiprows=8)
    y = df['StrideLength'].values

    border_file="eGait/GoldStandard_StrideBorders/"+filename+"txt"
    data = pd.read_csv(border_file, skiprows=9, header=None)
    start_array = data[0].values
    end_array = data[1].values

    return y, start_array, end_array

def upsample(signal):
    signal_upsampled = np.zeros((800,6))
    for i in np.arange(signal.shape[1]):
        t = np.linspace(0, 1, len(signal))
        func = interpolate.interp1d(t, signal[:,i], kind='cubic')
        t_upsampled = np.linspace(0, 1, 800)
        signal_upsampled[:,i] = func(t_upsampled)

    # convert acc unit, gyro unit
    signal_upsampled[:,0:3]=signal_upsampled[:,0:3] * 9.80665
    signal_upsampled[:, 3:6] = np.radians(signal_upsampled[:, 3:6])

    return signal_upsampled

def eGait_step_seg(x, y, start_border, end_border, self):
    for i in np.arange(len(start_border)-1):
        start_border_shift=max(start_border[i]-20,0)
        step_x = x[start_border_shift:end_border[i]+20,:]
        step_x = upsample(step_x[:100,:])  # upsample the signal
        self.labels.append(y[i])
        self.data.append(step_x)

class eGaitDataset(Dataset):
    def __init__(self, filenames, seq_length):
        # `filenames` is a list of strings that contains all file names.
        # `batch_size` determines the number of files that we want to read in a chunk.
        #  self.data -> acc, gyro and labels. ground truth and ZUPT results.
        #  self.labels -> ground truth of stride length from GaitRite system
        self.filenames = filenames
        self.seq_length = seq_length
        self.data = []
        self.labels = []
        self.subject_dict = pd.DataFrame({}) #empty dict to save extra info 

        for file in self.filenames:
            x = pd.read_csv(file, index_col=0).values
            y, start_border, end_border= get_Gaitrite(os.path.basename(file)[:-3])

            # if y.shape[0]<5:
            #     continue
            
            # adjust orientation of IMUs
            x [:, [0, 2]] = -1 * x[:, [2, 0]]
            x [:, [3, 5]] = -1 * x[:, [5, 3]]

            for i in np.arange(6):
                x[:, i] = utils.butter_lowpass_filter(x[:, i], 20, 102.4, order=3)

            # unit convertion: cm to m 
            y = y / 100

            # segment the data into steps
            eGait_step_seg(x, y, start_border, end_border, self)
        print("finish loading data")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):  # index means index of the chunk.
        label = self.labels[index]
        data = self.data[index].astype(float)
        return data, label
