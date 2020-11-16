from scipy import signal
import numpy as np

def bandpass_filter(data, fs=100, low=0.3, high=35): #no reference for freq    
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq
    
    try:
        assert len(data.shape) == 2
        assert data.shape[1] == 3000
    except:
        print("Error: please check data shape, it should be 2D array of a raw signal (nsamples, 3000).")
        raise ValueError

    order = 2
    b, a = signal.butter(order, [low, high], btype='band')
    tmp = signal.filtfilt(b, a, np.concatenate(data))
    
    return tmp.reshape((data.shape[0], data.shape[1]))