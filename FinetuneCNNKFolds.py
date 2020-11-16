# -*- coding: utf-8 -*-
from FinetuneCNN import Finetune
import configure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pprint
import os
import tensorflow as tf
from tensorflow import layers, keras
import tempfile
import itertools
import timeit
from sklearn.metrics import confusion_matrix, f1_score
import math
import random
import data_loader
import utils
import preprocessor
import datetime
import bot
import time

pjoin = os.path.join
logger = configure.logger

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ["CUDA_VISIBLE_DEVICES"]="7"


# +
###### Requires subjects list setup #######
if 'ss2' in configure.finetune['channel']:
    subjects = list(range(0, 19)) # 19 subjects (C4 included)
elif 'ss4' in configure.finetune['channel']:
    subjects = [0, 1, 3, 5, 8, 9, 12, 15, 16] # meta-validation
elif configure.finetune_dataset == 'sleepedfx':
    subjects = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 70,
                71, 72, 73, 74, 75, 76, 77, 80, 81, 82]
elif configure.finetune_dataset == 'sleepedfx_st': 
    subjects = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24]
elif 'ISRUC_subgroupIII' in configure.finetune['channel']:
    subjects = list(range(1, 11))
elif 'ISRUC_subgroupI' in configure.finetune['channel']:
    subjects = list(range(1, 101))

elif configure.finetune_dataset ==  'cap':
    if not 'patients' in configure.finetune['channel']:
        subjects = ['n1', 'n2', 'n3', 'n5', 'n6', 'n7', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14', 'n15']
    else:        
        subjects = ['brux2', 'ins1', 'ins2', 'ins3', 'ins4', 'ins5', 'ins6', 'ins7', 'ins8', 'ins9', 
                    'narco1', 'narco2', 'narco3', 'narco4', 'narco5', 'nfle1', 'nfle10', 'nfle11', 
                    'nfle12', 'nfle13', 'nfle14', 'nfle15', 'nfle16', 'nfle17', 'nfle18', 'nfle19', 
                    'nfle2', 'nfle20', 'nfle21', 'nfle22', 'nfle23', 'nfle24', 'nfle25', 'nfle26', 
                    'nfle27', 'nfle28', 'nfle29', 'nfle3', 'nfle30', 'nfle31', 'nfle32', 'nfle33', 
                    'nfle34', 'nfle35', 'nfle36', 'nfle37', 'nfle38', 'nfle39', 'nfle4', 'nfle40', 
                    'nfle5', 'nfle6', 'nfle7', 'nfle8', 'nfle9', 'plm1', 'plm10', 'plm2', 'plm3', 
                    'plm4', 'plm5', 'plm6', 'plm7', 'plm8', 'plm9', 'rbd1', 'rbd10', 'rbd11', 'rbd12', 
                    'rbd13', 'rbd14', 'rbd15', 'rbd16', 'rbd17', 'rbd18', 'rbd19', 'rbd2', 'rbd20', 
                    'rbd21', 'rbd22', 'rbd3', 'rbd4', 'rbd5', 'rbd6', 'rbd7', 'rbd8', 'rbd9', 'sdb1', 
                    'sdb2', 'sdb3', 'sdb4'] # all subjects
            
elif configure.finetune_dataset ==  'ucd':
    subjects = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

logger.log("Finetune on subjects (individually):", subjects)
logger.finetune_log()


# +
all_results_path = None
results = []
test_results = []

nfolds = 5
K_set = [5]
nepochs_set = [5, 10, 15, 20, 50, 100, 150, 200, 300, 400, 500]
lr_set = [0.1, 0.01, 0.001, 0.0001, 0.00001]

# +
for subject_id in subjects:
    
    go_next_subj = False
    
    logger.log("------------------- subject_id:", subject_id, '---------------------')
    
    val_loss = None
    best_finetune_weight_path = None
    best_k, best_lr, best_nepochs = None, None, None
    
    for K in K_set:

        for lr in lr_set:
            
            sum_loss = 0
            for random_time in range(0, nfolds):
                
                results = []
                random_seed = (random_time + 1) * 3
                logger.log('random time:', random_time, 'seed:', random_seed)


                if all_results_path == None:
                    all_results_path = configure.current_time
                    current_time = configure.current_time
                else:
                    while True:
                        current_time = pjoin('./logs/', datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
                        if not os.path.exists(current_time):
                            logger.log('Creating new dir:', current_time)
                            try:
                                os.makedirs(current_time)
                                break
                            except FileExistsError as e:
                                logger.log('Error:', e)
                        else:
                            logger.log(current_time, 'Found path exists, trying again..')
                        
                        time.sleep(3)
                            

                finetune = Finetune(K, 
                                    subject_id = subject_id,
                                    model_pretrain_path = configure.pretrain_path,
                                    finetune_weight_path = pjoin(current_time, 'finetune_weight'),
                                    seed = random_seed,
                                    nepochs = nepochs_set,
                                    lr = lr)

                if len(finetune.y_train) == 0 and len(finetune.y_val) == 0 and len(finetune.y_test) == 0:
                    # subjects who don't have enough samples (K*3)
                    results.append([subject_id, '-', '-', '-', '-', 
                                    configure.pretrain_path, 
                                    configure.finetune["channel"], '-', 
                                    '-', '-', '-', '-', '-', '-', finetune.finetune_weight_path])
                    go_next_subj = True
                    break

                else:
                    finetune.init_model_ops()
                    utils.create_dir(finetune.finetune_weight_path)
                    finetune_result = finetune.train()

                    for nepochs in nepochs_set:
                        loss = finetune_result['epoch_'+str(nepochs)]['val_loss']
                        f1 = finetune_result['epoch_'+str(nepochs)]['f1']

                        one_res = [subject_id, random_time, K, lr, nepochs, 
                                   configure.pretrain_path, 
                                   configure.finetune["channel"], loss]
                        one_res.extend(f1)
                        one_res.append(np.mean(f1))
                        one_res.append(finetune.finetune_weight_path)

                        logger.log(one_res)
                        results.append(one_res)

                        del one_res

                df = pd.DataFrame(results, columns=['subj_id', 'random_round', 'K', 'lr', 'nepochs', 
                                                     'pretrain_path', 'channel',
                                                     'loss', 'W', 'N1', 'N2', 'N3', 'REM', 'MF1', 
                                                     'finetune_weight_path'])
                print(df)

                # each fold save weight in separate directory, but the first one keep all fold's result
                all_result_csv = pjoin(all_results_path, 'all_folds.csv')
                if os.path.exists(all_result_csv):
                    df.to_csv(all_result_csv, mode='a', header=False, index=False)
                else:
                    df.to_csv(all_result_csv, index=False)
                    
                
                del finetune, results, df
                
            if go_next_subj:
                break
                
        if go_next_subj:
                break

    bot.sendMsg('[DONE] FinetuneCNNKFolds fine-tune path:', all_results_path, '=> Subject', subject_id, 'done.')
                
bot.sendMsg('[DONE] FinetuneCNNKFolds fine-tune path:', all_results_path, ' 🥳🥳🥳')
