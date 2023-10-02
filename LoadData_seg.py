import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pickle, os
import utils, re
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # Sort the batch by sequence length
    batch_sorted = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)

    # Separate sequences and labels
    sequences = [item[0] for item in batch_sorted]
    labels = [item[1] for item in batch_sorted]

    # Get lengths
    lengths = [seq.shape[0] for seq in sequences]

    # Pad sequences
    padded_sequences = pad_sequence(torch.tensor(sequences[0]), batch_first=True)
    padded_labels = pad_sequence(torch.tensor(labels[0]), batch_first=True)
    
    return padded_sequences, lengths, padded_labels


def find_start_stop_flag(arr):
    zero_indices = np.where(arr == 0)[0]

    # If there are no zeros, return None for both indices
    if len(zero_indices) == 0:
        return None, None

    # Return the first and last index from zero_indices
    return zero_indices[0], zero_indices[-1]

def crop_subarrays_with_overlap(values, zero_indices, subarray_length=2000, overlap=0.3, savepng='output.png'):
    subarrays = []
    labelarray=[]
    step = int(subarray_length * (1 - overlap))
    i = 0

    while i < len(values) - step:
        # Adjust start to the closest zero if it's not already zero
        if values[i,-2] != 0:
            i = min(zero_indices[zero_indices > i])

        # Crop the subarray
        subarray = values[i:i+subarray_length]

        # Ensure the subarray ends with a zero, if not adjust the end
        if subarray[-1,-2] != 0:
            zero_end = max(zero_indices[zero_indices < i+subarray_length])
            subarray = values[i:zero_end+1]

        # Pad the subarray if its length is less than 2000
        if len(subarray) < subarray_length:
            pad_length = subarray_length - len(subarray)
            subarray = np.pad(subarray, [(pad_length // 2, pad_length - pad_length // 2),(0,0)], 'edge')

        i += step
        if sum(subarray[:,-2]) == 0:
            continue

        # plt.plot(subarray[:,0:3])
        # plt.plot(subarray[:,-2]*10)
        # plt.savefig('test_crop/'+os.path.basename(savepng)[:-4]+str(i)+'.png')
        # plt.close()

        subarrays.append(subarray[:,0:6])
        labelarray.append(subarray[:,-2])

    return subarrays, labelarray


class segGaitDataset(Dataset):
    def __init__(self, filenames, seq_length, rate=1000):
        # `filenames` is a list of strings that contains all file names.
        # `batch_size` determines the number of files that we want to read in a chunk.
        #  self.data -> acc, gyro and labels. ground truth and ZUPT results.
        #  self.labels -> ground truth of stride length from GaitRite system
        self.filenames = filenames
        self.seq_length = seq_length
        self.data = []
        self.labels = []
        self.subject_dict = pd.DataFrame({}) #empty dict to save extra info 
        self.rate = rate

        for file in self.filenames:
            # x[:,-2] is the ground truth for segmentation tasks
            # y is the ground truth for stride length prediction 
            with open(file, 'rb') as f:  
                x, y, seg_index = pd.read_pickle(f)
            for i in np.arange(6):
                x[:, i] = utils.butter_lowpass_filter(x[:, i], 20, self.rate, order=3)


            x = x[200:, :] 

            first, last = find_start_stop_flag(x[:,-2])

            # crop extra steps
            x = x[first:last,:]

            zero_indices = np.where(x[:,-2] == 0)[0]

            subarrays, labelarray = crop_subarrays_with_overlap(x[:-1], zero_indices, subarray_length=seq_length, overlap=0.3, savepng=file)

            self.labels.extend(labelarray)
            self.data.extend(subarrays)

        print("finish loading, start training")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):  # index means index of the chunk.
        label = self.labels[index].reshape(-1,1)
        data = self.data[index].reshape(-1,6)
        return data, label
