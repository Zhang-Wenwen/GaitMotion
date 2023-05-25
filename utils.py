import numpy as np
from scipy import signal


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
    # plt.plot(gyro_mag),plt.plot(np.ones(datasize)*50),plt.show()

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

