# -*- coding: utf-8 -*-
import datetime
import os
from Logger import Logger
pjoin = os.path.join

############################
# model hyperparameters ###
###########################
maml = {'meta_lr': 0.001,                      # meta-weight lr
        'update_lr': 0.001,                    # inside task lr
        'niters':50000,                        # number of meta-iterations
        'K': 10,                               # number of samples per each iteration & each task (K Shot)
        'num_updates': 10,                     # number of update steps in each task
        'ntasks': 9,                           # number of meta-tasks
        'datasets': {
            'train': {
                'mass': { 
                    'ss1_eeg_c3_a2_eog_emg.npz': 'all' , # list of id or 'all'
                    'ss3_eeg_c3_a2_eog_emg.npz': 'all' , # list of id or 'all'
                    'ss5_eeg_c3_a2_eog_emg.npz': 'all'
                } 
            },
            'validate': {
                'mass': { 'ss4_eeg_c3_a2_eog_emg.npz': [0, 1, 3, 5, 8, 9, 12, 15, 16] } 
            }
        }
       }
pretrain = {'lr': 0.01,                             # pretrain lr
            'niters': 50000,                        # number of meta-iterations
            'K': 10,                                # number of samples per each iteration & each task (K Shot)
            'datasets': {
                'train': {
                    'mass': { 
                        'ss1_eeg_c3_a2_eog_emg.npz': 'all' , # list of id or 'all'
                        'ss3_eeg_c3_a2_eog_emg.npz': 'all' , # list of id or 'all'
                        'ss5_eeg_c3_a2_eog_emg.npz': 'all' 
                    } 
                },
                'validate': {
                    'mass': { 'ss4_eeg_c3_a2_eog_emg.npz': [0, 1, 3, 5, 8, 9, 12, 15, 16] } 
                }
            },
            'multiply_batch_size': 9,                 # default is 1 (batch_size = K * multiply_batch_size)
            'one_batch_per_iter': True,               # False: run all batches
            'fix_batch_size': None                    # `None` = use multiply_batch_size / specify `batch_size`
           }
finetune = {'channel': 'eeg_fpz_cz_eog_emg'}          # type the same as in datasets dict belows


############################
# #### run time settings ####
# ###########################
do_bandpass = True
min_samples = 0                # filter out if nsamples in any class < min_samples
pretrain_dataset = ['mass']
finetune_dataset = 'sleepedfx'
nepochs_per_sample = 1         
modals = ['eeg', 'eog', 'emg']               # list of modals of inputs
cnn_type = '2D'                # to specify 1D/2D CNN for 1 modal

############################
######### dataset #########
###########################

# specify your own path & filename for each dataset
datasets = { 'sleepedfx': {'path':'../../data/sleep_edfx_sc', 
                           'channels': ['eeg_fpz_cz_eog_emg'],
                           'bandpass': (None, None) 
                          },
            'sleepedfx_st': {'path':'../../data/sleep_edfx_st', 
                           'channels': ['eeg_fpz_cz_eog_emg'],
                           'bandpass': (None, None) 
                            },
            'mass': {'path':'../../data/MASS', 
                           'channels': [ 
                                         'ss1_eeg_c3_a2_eog_emg.npz',
                                         'ss2_eeg_eog_emg.npz', # use new version (C4-A1 included)
                                         'ss3_eeg_c3_a2_eog_emg.npz',
                                         'ss4_eeg_c3_a2_eog_emg.npz',
                                         'ss5_eeg_c3_a2_eog_emg.npz',
                                       ],
                           'bandpass': (None, None)
                    },
            'cap': {'path':'../../data/CAP',
                    'channels': [ 'CAP_normal_eog_emg.npz',
                                  'CAP_patients_eog_emg.npz',
                                ],
                    'bandpass': (None, None)
                   },
            'isruc': {'path':'../../data/ISRUC', 
                      'channels': ['ISRUC_subgroupI_eog_emg.npz',
                                   'ISRUC_subgroupIII_C3-A2_eog_emg.npz' 
                                  ],
                    'bandpass': (None, None)
                     },
            'ucd': {'path':'../../data/UCD', 
                    'channels': [ 'c3a2_eog_emg' ],
                    'bandpass': (None, None)
                },
           }

#################################
##### other path configure ######
#################################
while True:
    current_time = pjoin('./logs/', datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        
    if not os.path.exists(current_time):
        # prevent overwriting the existing one
        print('Creating new dir:', current_time)
        os.makedirs(current_time)
    
        log = pjoin(current_time, 'log.txt')
        csv_result_path = pjoin(current_time, 'results.csv')
        meta_weight_path = pjoin(current_time, 'metatrain_weight')      # path to save meta-train weights
        normal_weight_path = pjoin(current_time, 'normaltrain_weight')  # path to save normal pre-train weights
        finetune_weight_path = pjoin(current_time, 'finetune_weight')    # path to save fine-tune weights
        
        # specify path here for fine-tuning
        pretrain_path = pjoin('./logs/', 
                              '2020-10-10_09:24:11', # datetime of weight
                              'metatrain_weight'     # type (directory name)
                             ) 
        logger = Logger(log)

        break
    else:
        print('⚠️ Found path exists, trying again..')
