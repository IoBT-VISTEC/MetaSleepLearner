import os
import numpy as np
from sleep_stage import print_n_samples_each_class
from utils import get_balance_class_oversample
import re

pjoin = os.path.join
MASS_tmp_channel = None # to load only once

import numpy as np
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


def normal_npz_loader(data_dir, channel, subject_idx):
    """
    Load data from 1 subject (subject_idx)
    return shape = (n_nights, samples, 3000, 1) # normally n_nights = 1
    """
    data_path = pjoin(data_dir, channel)
    # print("Load data from: {} subj: {}".format(data_path, subject_idx))
    
    temp = np.load(data_path)
    x = temp['X']
    y = temp['sleep_stages']
    pid = temp['pid']
    
    x = np.array([x[pid==subject_idx]])
    y = np.array([y[pid==subject_idx]])
    
    return x, y

def get_subject_lists(ds_name, data_dir, channel):
    if 'sleepedfx_st' in ds_name:
        return [1, 2]+list(range(4, 23))+[24]
    elif 'sleepedfx' in ds_name:
        return list(range(0, 20))
    elif 'mass' in ds_name:
        data_path = pjoin(data_dir, channel)
        temp = np.load(data_path)
        return range(0, len(temp['x_list.npy']))
    elif 'ucd' in ds_name:
        data_path = pjoin(data_dir, channel)
        files = os.listdir(data_path)
        return sorted([int(s[-7:-4]) for s in files])
    else:
        data_path = pjoin(data_dir, channel)
        temp = np.load(data_path)
        try:
            pid = temp['pid']
        except:
            pid = temp['pid.npy']

        # sorted descending in order to select the same subject as val tasks (for CAP)
        all_pid = sorted(list(set(pid)), reverse=True) 
    
        return all_pid

def loader(data_dir, channel, subject_idx):
    if 'cap' in data_dir.lower():
        return cap_loader(data_dir, channel, subject_idx)
    elif 'isruc' in data_dir.lower():
        return isruc_loader(data_dir, channel, subject_idx)
    elif 'mass' in data_dir.lower():
        return mass_loader(data_dir, channel, subject_idx)
    elif 'ucd' in data_dir.lower():
        return ucd_loader(data_dir, channel, subject_idx)
    else:
        return sleepedf_loader(data_dir, channel, subject_idx)

def cap_loader(data_dir, channel, subject_idx):
    return normal_npz_loader(data_dir, channel, subject_idx)

def isruc_loader(data_dir, channel, subject_idx):
    return normal_npz_loader(data_dir, channel, subject_idx)

def sleepedf_loader(data_dir, channel, subject_idx):
    """
    Load data from 1 subject (subject_idx)
    return shape = (n_nights, samples, 3000, 1) -> n_nights = 1 or 2
    """
    allfiles = os.listdir(pjoin(data_dir, channel))
    subject_files = []
    print('data_dir', data_dir)
    if 'st' in data_dir.lower():
        for idx, f in enumerate(allfiles):
            if subject_idx < 10:
                pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]J0\.npz$".format(subject_idx))
            else:
                pattern = re.compile("[a-zA-Z0-9]*{}[1-9]J0\.npz$".format(subject_idx))
            if pattern.match(f):
                subject_files.append(os.path.join(data_dir, channel, f))
    else:
        for idx, f in enumerate(allfiles):
            if subject_idx < 10:
                pattern = re.compile("[a-zA-Z0-9]*0{}[1-9][EFG]0\.npz$".format(subject_idx)) 
            else:
                pattern = re.compile("[a-zA-Z0-9]*{}[1-9][EFG]0\.npz$".format(subject_idx))
            if pattern.match(f):
                subject_files.append(os.path.join(data_dir, channel, f))

    if len(subject_files) == 0 or len(subject_files) > 2:
        # each subject can have only up to 2 nights
        print(subject_files)
        raise Exception("Invalid file pattern")

    def load_npz_file(npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

    def load_npz_list_files(npz_files):
        """Load data and labels from list of npz files."""
        data = []
        labels = []
        fs = None
        for npz_f in npz_files:
            print( "Loading {} ...".format(npz_f))
            tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            tmp_data = np.squeeze(tmp_data)
            # tmp_data = tmp_data[:, :, np.newaxis, np.newaxis] -> add 1 to last dimension

            # Casting
            tmp_data = tmp_data.astype(np.float32)
            tmp_labels = tmp_labels.astype(np.int32)

            data.append(tmp_data)
            labels.append(tmp_labels)

        return data, labels

    subject_files = sorted(subject_files)
    print( "Load data from: {}".format(subject_files))
    data, labels = load_npz_list_files(subject_files)

    return data, labels

def ucd_loader(data_dir, channel, subject_idx):
    """
    Load data from 1 subject (subject_idx)
    return shape = (n_nights, samples, 3000, 1) -> n_nights = 1 or 2
    """
    allfiles = os.listdir(pjoin(data_dir, channel))
    subject_files = []
    print('data_dir', data_dir)
    
    subject_files = "ucddb" + "{:03d}".format(subject_idx) + ".npz"
    print("subject_files:", subject_files)
    fpath = os.path.join(data_dir, channel, subject_files)

    if not os.path.exists(fpath):
        raise Exception("Invalid file name:", fpath)

    def load_npz_file(npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

    print( "Load data from: {}".format(fpath))
    data, labels, sampling_rate = load_npz_file(fpath)

    return [data], [labels]



def mass_loader(data_dir, channel, subject_idx):
    """
    Load data from 1 subject (subject_idx)
    return shape = (n_nights, samples, 3000, 1) # normally n_nights = 1
    """
    global MASS_data, mass_x, mass_y, MASS_tmp_channel
    if MASS_tmp_channel == None or channel != MASS_tmp_channel:
        print('loading MASS data..')
        data_path = pjoin(data_dir, channel)
        MASS_tmp_channel = channel
        # print("Load data from: {} subj: {}".format(data_path, subject_idx))
        MASS_data = np.load(data_path)
        
        mass_x = MASS_data['x_list.npy']
        mass_y = MASS_data['y_list.npy']
        
    return np.array([np.squeeze(mass_x[subject_idx])]), np.array([mass_y[subject_idx]])
