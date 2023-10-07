import numpy as np
from scipy import signal
import glob, random
import re, os
import pandas as pd

def prepare_files(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    files = []
    val_files = []
    test_files = []
    # train on the normal file data, randomly pick up three files as test data
    for m_type in args.type:
        files.extend(glob.glob("./"+m_type+"//*.pkl"))  

    # random sample 10% of all the files as validation files
    val_files = random.sample(files, int(0.1*len(files)))

    # pick out all the data of one or several person for test
    # test Shuffle/Normal/Stroke subjects separately
    test_files=glob.glob("./Shuffle"+"//P"+args.subID+"_*.pkl")
    test_files.extend(glob.glob("./Stroke"+"//P"+args.subID+"_*.pkl"))
    test_files.extend(glob.glob("./Normal"+"//P"+args.subID+"_*.pkl"))    

    files = list(set(files) - set(test_files))
    files = list(set(files) - set(val_files))

    return files, test_files, val_files


def eGait_files(args):
    # random sample 2 files for testing, 2 files for eval
    # use the rest files for testing
    np.random.seed(args.seed)
    random.seed(args.seed)
    files = []
    val_files = []
    test_files =[]

    dir_adres= "./eGait/ConvertedCSV/"
    test_files.extend(glob.glob(dir_adres+"*.csv"))  
    files = random.sample(test_files, 2)
    test_files = list(set(test_files) - set(files))
    val_files = random.sample(test_files, 2)
    test_files = list(set(test_files) - set(val_files))

    return files, test_files, val_files

def butter_lowpass_filter(data, w, fs, order=5):
    # w = cutoff / (fs / 2)
    b, a = signal.butter(order, w, btype='low', analog=False, fs=fs)
    y = signal.lfilter(b, a, data)
    return y


def zupt(data):
    # This function will output step segmentation results by ZUPT algorithm.
    acc_s = data[:, 0:3].T
    gyro_s = data[:, 3:6].T
    datasize = len(data)
    # Compute accelerometer magnitude
    acc_mag = np.around(np.sqrt(acc_s[0] ** 2 + acc_s[1] ** 2 + acc_s[2] ** 2), decimals=4)
    # Compute gyroscope magnitude
    gyro_mag = np.around(np.sqrt((gyro_s[0]) ** 2 + (gyro_s[1]) ** 2 + (gyro_s[2]) ** 2), decimals=4)

    acc_stationary_threshold_H = 11
    acc_stationary_threshold_L = 9
    gyro_stationary_threshold = 50

    stationary_acc_H = (acc_mag < acc_stationary_threshold_H)
    stationary_acc_L = (acc_mag > acc_stationary_threshold_L)
    stationary_acc = np.logical_and(stationary_acc_H, stationary_acc_L)  # C1
    stationary_gyro = (gyro_mag < gyro_stationary_threshold)  # C2

    stationary = np.logical_and(stationary_acc, stationary_gyro)

    # this window is necessary to clean stationary array from false stance detection
    W = 100
    for k in range(datasize - W + 1):
        if (stationary[k] == True) and (stationary[k + W - 1] == True):
            stationary[k:k + W] = np.ones((W))

    for k in range(datasize - W + 1):
        if (stationary[k] == False) and (stationary[k + W - 1] == False):
            stationary[k:k + W] = np.zeros((W))

    return (stationary.reshape((len(stationary), 1))).astype(int)

def erosion_1d(signal, window_size=3):
    padding = window_size // 2
    padded_signal = np.pad(signal, (padding, padding), 'edge')
    output = np.zeros_like(signal)
    
    for i in range(len(signal)):
        output[i] = np.min(padded_signal[i:i+window_size])
        
    return output

def dilation_1d(signal, window_size=3):
    padding = window_size // 2
    padded_signal = np.pad(signal, (padding, padding), 'edge')
    output = np.zeros_like(signal)
    
    for i in range(len(signal)):
        output[i] = np.max(padded_signal[i:i+window_size])
        
    return output

def add_extra_info(file,init_step,num_files,subject_dict):
    # save the subject information in the dataframe. The stored information is: (1). ID of subject. (2). Walk type (Normal/Parkinson/Stroke). (3). start step of each trial.
    # (4). stop step of each trial. (5). Which trial this is. (6). Left side/Right side of the file.
    [walking_type, sub_name] =  os.path.split(file)
    [sub_id, trial] = re.findall(r'\d+\.\d+|\d+', sub_name)
    
    # subject_dict=subject_dict.append({"subjectID": sub_id, "walkingtype": walking_type[2:], "start_steps":init_step, "end_steps": num_files, 
    #                                   "trial":trial,"side":sub_name[-5]}, ignore_index=True)
    subject_dict=pd.concat([subject_dict, pd.DataFrame([{"subjectID": sub_id, "walkingtype": walking_type[2:], "start_steps":init_step, 
                                                         "end_steps": num_files, "trial":trial,"side":sub_name[-5]}])], ignore_index=True)
    return subject_dict