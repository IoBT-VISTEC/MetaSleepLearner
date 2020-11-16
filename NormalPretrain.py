# -*- coding: utf-8 -*-
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
import preprocessor
import utils
import configure
from Model import DeepFeatureNet, loss_func, MultiModalNet

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# +
os.environ["CUDA_VISIBLE_DEVICES"]="4"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# -


pjoin = os.path.join
logger = configure.logger
logger.pretrain_log()

class DataManagement:
    
    def __init__(self):
        self.X_train, self.y_train, self.X_val, self.y_val = self.load_all_data()
        
    def load_data(self, data_type='train'):
        conf = configure.pretrain
        logger.log('preparing..', data_type)
        results_x, results_y = [], []
        K = conf['K']
        task_index = -1
        for dataset in conf['datasets'][data_type]:
            for channel in conf['datasets'][data_type][dataset]:
                
                task_index += 1
                results_x.append([])
                results_y.append([])
                logger.log(dataset, ':', channel)
                
                subj_list = conf['datasets'][data_type][dataset][channel]
                if type(subj_list) == str and subj_list == 'all':
                    subjects = data_loader.get_subject_lists(dataset, configure.datasets[dataset]['path'], channel)
                else:
                    subjects = conf['datasets'][data_type][dataset][channel]
                logger.log('subjects:', len(subjects), subjects)
                
                nperc = np.zeros(shape=(5,))
                for subj in subjects:
                    xx, yy = data_loader.loader(configure.datasets[dataset]['path'], channel, subj)
                    
                    for x, y in zip(xx, yy):
                        class_samples = utils.get_sample_per_class(y)
                        if len(class_samples) == 5 and all([cl > K*2 for cl in class_samples]):
                            # use only if nsamples/class more than 'K' * 2
                            bp = configure.datasets[dataset]['bandpass']
                            if bp[0] != None and bp[1] != None:
                                logger.log('bandpass:', dataset, 'at', bp)
                                x = preprocessor.bandpass_filter(x, low=bp[0], high=bp[1])
                            
                            if len(configure.modals) == 1:
                                if x.shape[-1] == 3:
                                    x = x[:,:,0] # EEG ONLY (From 3 modals file)

                                x = np.expand_dims(x, axis=-1)
                               
                                if '2D' in configure.cnn_type:
                                    # just for testing on 2D-CNN with 1 modal
                                    x = np.expand_dims(x, axis=-1)

                                    logger.log('‼️‼️ Make sure you use 2D-CNN with 1 modal‼️‼️')
                                    logger.log('x', x.shape)
                            else:
                                if len(x.shape) == 2:
                                    x = np.expand_dims(x, axis=-2)
                                    
                            
                            if data_type == 'train':
                                logger.log('$ train subj: {} {} {} -> not oversample'.format(subj, 
                                                                                         x.shape, 
                                                                                         y.shape, 
                                                                                         class_samples))
                                x, y = utils.get_balance_class_oversample(x, y, logger)
                                
                            else:
                                logger.log('$ val subj: {} {}'.format(subj, class_samples))   
                             
                            nperc += class_samples
                            results_x[task_index].append(x)
                            results_y[task_index].append(y)
                            
                        else:
                            logger.log('$ removed subj: {} {} (< K*2)'.format(subj, class_samples))  
                   
                logger.log('task:', task_index, nperc, np.sum(nperc))

                
        results_x, results_y = np.array(results_x), np.array(results_y)
        logger.log('n_meta'+data_type+'_tasks =', len(results_x)) 
        for tid, d in enumerate(results_x):
            logger.log('task:', tid, data_type+':', len(d), 'records, x[0]:', d[0].shape)

            
        return results_x, results_y
        
    def load_all_data(self):
        '''
        load train and val data
        return in shape x = (nsubjects, nsamples, 3000, 1, nmodals(optional))
        '''      
        logger.log('PRE-TRAINING DATA..')
        x_train, y_train = self.load_data('train')
        x_val, y_val = self.load_data('validate')
        
        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)
        x_val = np.concatenate(x_val)
        y_val = np.concatenate(y_val)
        print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
        
        return x_train, y_train, x_val, y_val
                
    def get_data(self):
        return self.X_train, self.y_train, self.X_val, self.y_val 


class NormalPretrain(object):
    def __init__(self):
        self.train_loss = []
        self.valid_loss = []
        self.train_acc = []
        self.valid_acc = []
        self.train_f1 = []
        self.valid_f1 = []
        self.interval_save_model = 5
        self.nmodals = len(configure.modals)
        
        self.nsamples_per_class = configure.pretrain["K"]
        if self.nmodals == 1:
            logger.log('‼️configure.cnn_type:', configure.cnn_type)
        
        if configure.pretrain['fix_batch_size'] == None:
            multiply_batch_size = configure.pretrain["multiply_batch_size"]
            self.batch_size = self.nsamples_per_class * 5 * configure.pretrain["multiply_batch_size"]
            self.val_samples_per_task = configure.pretrain["K"] * 2 * 5

            logger.log('* multiply_batch_size = ', multiply_batch_size)
            logger.log('* batch_size = K * 5 * multiply_batch_size = ', self.batch_size)
            logger.log('* val_size = K * 5 * 2 = ', self.val_samples_per_task)
        else:
            #### manual fix batch_size & val size
            self.val_samples_per_task = None # use all validation samples
            self.batch_size = configure.pretrain['fix_batch_size']
            logger.log('* batch_size =', self.batch_size)
            logger.log('* val_size = all samples')
            ######
        
        self.data_manager = DataManagement()
        self.x_train, self.y_train, self.x_val, self.y_val = self.data_manager.get_data()
        
        if self.nmodals > 1:
            logger.log('Using.. MultiModalNet')
            self.model = MultiModalNet()
        elif configure.nepochs_per_sample == 1:
            logger.log('Using.. DeepFeatureNet')
            self.model = DeepFeatureNet()
        else:
            raise Exception('Model incorrect')
        
    def init_model_ops(self):
        self.weights = self.model.construct_weights()

        self.is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
        self.lr = tf.placeholder(tf.float32, shape=(), name='lr')
        if self.nmodals == 1 and '1D' in configure.cnn_type:
            self.inputs = tf.placeholder(tf.float32, 
                                         shape=[None, 3000*configure.nepochs_per_sample, 1], 
                                         name='inputs')
        else:
            self.inputs = tf.placeholder(tf.float32, 
                                         shape=[None, 3000*configure.nepochs_per_sample, 1, self.nmodals], 
                                         name='inputs')
            
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')
        self.labels_one_hot = tf.one_hot(self.labels, 5, axis=-1)
        
        self.outputs, self.losses, self.accuracies = [], [], []
           
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            self.outputs = self.model.construct_model(self.inputs, self.weights, self.is_train)
            self.reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
            self.total_loss = tf.reduce_mean(tf.add(loss_func(self.outputs, self.labels_one_hot), 
                                                    self.reg_loss, name='total_loss'))
            self.total_accuracy = tf.reduce_mean(tf.contrib.metrics.accuracy(tf.argmax(self.outputs, 1), 
                                                                             tf.argmax(self.labels_one_hot, 1)))

        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            self.optimizer = optimizer = tf.train.AdamOptimizer(configure.pretrain['lr'])
            self.grads_and_vars = self.optimizer.compute_gradients(self.total_loss, tf.trainable_variables())

            self.apply_grads_op = self.optimizer.apply_gradients(self.grads_and_vars, name='apply_grads_op')
            self.saver = tf.train.Saver(tf.trainable_variables())
            
            
    def is_any_weight_changed(self, prev_weights, current_weights, names):
        for index, name in enumerate(names):
            print('checking weight:', name)
            w = current_weights[index]
            while True:
                if type(w) is list or type(w) is np.ndarray:
                    w = w[0]
                else:
                    break
            logger.log(name, 'compare:', w, prev_weights[index])
            if w != prev_weights[index]:
                return True
        return False
    
    
    def run_epoch(self, x, y, sess, training):
        sum_losses = 0
        n_batches = 0
        total_samples = 0
        correct_prediction = 0
        y_true_all = []
        y_pred_all = []
        
        utils.create_dir(configure.normal_weight_path)
        
        if training:
            p = range(0, len(x))

            start_time = timeit.default_timer()
            nbatches = len(x)
            logger.log('training for {} mini-batches'.format(nbatches))
            utils.printProgressBar(0, nbatches, prefix = 'Progress:', suffix = 'Complete', length = 50)
            for i in range(nbatches):
                x_select, y_select = np.array(x[p[i]]), np.array(y[p[i]])
                if self.nmodals == 1 and '2D' in configure.cnn_type:
                    x_select = np.expand_dims(x_select, axis=-1)
                
                if len(y_select) > 0:
                    _, loss_value, y_true, y_logits = sess.run([self.apply_grads_op, self.total_loss, 
                                                                self.labels, self.outputs], 
                                                               feed_dict={self.inputs: x_select,
                                                                          self.labels: y_select,
                                                                          self.is_train: True})
                    sum_losses += loss_value
                    n_batches += 1
                    y_pred = np.argmax(y_logits, axis=-1)
                    correct_prediction += (y_true == y_pred).sum()
                    y_true_all.append(y_true)
                    y_pred_all.append(y_pred)
                    total_samples += len(y_true)
                
                    utils.printProgressBar(i + 1, nbatches, prefix = 'Progress:', suffix = 'Complete', length = 50)
                
                if configure.pretrain['one_batch_per_iter']:
                    break
                    
            sum_losses /= n_batches
            acc = correct_prediction/total_samples
            y_true_all = np.hstack(y_true_all)
            y_pred_all = np.hstack(y_pred_all)
            f1 = f1_score(y_true_all, y_pred_all, average='macro')
            duration = timeit.default_timer() - start_time
            self.train_loss.append(sum_losses)
            self.train_acc.append(acc)
            self.train_f1.append(f1)
            
        else:
            start_time = timeit.default_timer()
            for i in range(int(np.ceil(len(x)/self.batch_size))):
                x_select = np.array(x[self.batch_size*i:self.batch_size*(i+1)])
                if self.nmodals == 1 and '2D' in configure.cnn_type:
                    x_select = np.expand_dims(x_select, axis=-1)
                
                loss_value, y_true, y_logits = sess.run([self.total_loss, self.labels, self.outputs], 
                                                       feed_dict={self.inputs: x_select,
                                                                  self.labels: y[self.batch_size*i:self.batch_size*(i+1)],
                                                                  self.is_train: False})
                sum_losses += loss_value
                n_batches += 1
                y_pred = np.argmax(y_logits, axis=-1)
                correct_prediction += (y_true == y_pred).sum()
                y_true_all.append(y_true)
                y_pred_all.append(y_pred)
                total_samples += len(y_true)

            sum_losses /= n_batches
            acc = correct_prediction/total_samples
            y_true_all = np.hstack(y_true_all)
            y_pred_all = np.hstack(y_pred_all)
            f1 = f1_score(y_true_all, y_pred_all, average='macro')
            duration = timeit.default_timer() - start_time
            self.valid_loss.append(sum_losses)
            self.valid_acc.append(acc)
            self.valid_f1.append(f1)
            
            # log per class results
            per_class_acc_txt, per_class_acc_arr = utils.get_per_class_acc(y_true_all, y_pred_all)
            logger.log(per_class_acc_txt)
            per_class_f1_txt, per_class_f1_arr = utils.get_per_class_f1(y_true_all, y_pred_all)
            logger.log(per_class_f1_txt)
        
        return sum_losses, acc, f1, duration, y_true_all, y_pred_all
    
    def get_random_val(self):
        print('Randomly pick validation samples..')
        x_valid, y_valid = [], []
        for task in range(len(self.x_val)):
            logger.log('picking..', self.val_samples_per_task, 'samples from task', task)

            # randomly picked samples from all tasks
            xa, ya, xb, yb = utils.pick_samples(self.x_val[task], 
                                                self.y_val[task], 
                                                self.val_samples_per_task,
                                                logger=logger,
                                                fix_val_sample=False,
                                                request_b=False
                                               )

            x_valid.extend(xa)
            y_valid.extend(ya)

            print('xa: ', np.array(xa).shape, ' ya:', len(ya))

        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)

        print('x_valid', x_valid.shape, 'y_valid', y_valid.shape)

        if self.nmodals == 1:
            x_valid = np.vstack(x_valid).reshape([-1, 3000*configure.nepochs_per_sample, 1])
        else:
            x_valid = np.vstack(x_valid).reshape([-1, 3000*configure.nepochs_per_sample, 1, self.nmodals])
        y_valid = np.hstack(y_valid)
            
        return x_valid, y_valid

    def train(self):
        
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            
            train_acc, val_acc = [], []
            train_loss, val_loss = [], []
            train_f1, val_f1 = [], []
            best_loss = None
            y_valid_best = []
            output_best = []
            ytrue_val, ypred_val = [], []
            best_iter = -1
        
            lr = configure.pretrain['lr']

            weight_name_list = ['conv11_w', 'conv12_w', 'fc1_w', 'fc1_b']
            weight_list = [self.weights[wi] for wi in weight_name_list]
            prev_weights = [None for i in range(0, len(weight_name_list))]
            
            print('Combine all samples..')
            if self.nmodals == 1 and '2D' in configure.cnn_type:
                self.x_train = np.vstack(self.x_train).reshape([-1, 3000*configure.nepochs_per_sample, 1])
            else:
                self.x_train = np.vstack(self.x_train).reshape([-1, 3000*configure.nepochs_per_sample, 1, self.nmodals])
            
            self.y_train = np.hstack(self.y_train)
    
            logger.log('After combined samples:')
            logger.log('train samples =', self.x_train.shape, self.y_train.shape)
            
            nepochs = configure.pretrain['niters']
            if self.val_samples_per_task == None:
                # use all validation samples
                if self.nmodals == 1:
                    x_valid = np.vstack(self.x_val).reshape([-1, 3000*configure.nepochs_per_sample, 1])
                else:
                    x_valid = np.vstack(self.x_val).reshape([-1, 3000*configure.nepochs_per_sample, 1, self.nmodals])

                y_valid = np.hstack(self.y_val)
                
                    
            for iter_no in range(0, nepochs):
                logger.log('########## ITER:', iter_no, '##########')
                
                ###### setup data loader -- can arrage before coming in this loop too ######
                x_all, y_all = utils.arrange_all_minibatches(self.x_train, self.y_train, self.batch_size, logger)
                ###############################
                
                sum_losses, acc, f1, duration, ytrue, ypred = self.run_epoch(x_all, y_all, sess, training=True)
                logger.log('Training Loss: {0:.4f}, Accuracy: {1:.4f}, F1: {2:.4f} Duration: {3:.2f}'.format(sum_losses, 
                                                                                                        acc, 
                                                                                                        f1, 
                                                                                                        duration))
                train_acc.append(acc)
                train_loss.append(sum_losses)
                train_f1.append(f1)
                del x_all, y_all

                if self.val_samples_per_task != None:
                    # randomly pick samples every iteration
                    x_valid, y_valid = self.get_random_val()
 
                logger.log('val samples =', x_valid.shape, x_valid.shape)
                if len(x_valid) > 0:    
                    # Run validation for 1 epoch
                    sum_losses_val, acc_val, f1_val, duration_val, ytrue_val, ypred_val = self.run_epoch(x_valid, 
                                                                                                       y_valid, 
                                                                                                       sess, 
                                                                                                       training=False)
                    logger.log('Validation Loss: {0:.4f}, Acc: {1:.4f}, F1: {2:.4f} Duration: {3:.2f}'.format(sum_losses_val,
                                                                                                              acc_val, 
                                                                                                              f1_val, 
                                                                                                              duration_val))
                    val_acc.append(acc_val)
                    val_loss.append(sum_losses_val)
                    val_f1.append(f1_val)

                    if best_loss == None or sum_losses_val < best_loss:
                        logger.log('Best iter:', iter_no, 'saved!')
                        best_iter = iter_no
                        best_loss = sum_losses_val
                        self.saver.save(sess, pjoin(configure.normal_weight_path, "model_best.ckpt"))
                        print('saved to', pjoin(configure.normal_weight_path, "model_best.ckpt"))
                        y_valid_best = ytrue_val
                        output_best = ypred_val
                        

                if iter_no % self.interval_save_model == 0 or iter_no == nepochs-1:
                    np.savez(pjoin(configure.normal_weight_path, 'results'), 
                                             train_acc=train_acc, 
                                             train_loss=train_loss,
                                             val_acc=val_acc, 
                                             val_loss=val_loss,
                                             yval_true=ytrue_val, yval_pred=ypred_val,
                                             yval_true_best=y_valid_best, yval_pred_best=output_best,
                                             best_loss=best_loss, best_iter = best_iter
                            )
                    
                    self.saver.save(sess, pjoin(configure.normal_weight_path, "model.ckpt"))
                    print('saved to', pjoin(configure.normal_weight_path, "model.ckpt"))
                    
                del ytrue, ypred, ytrue_val, ypred_val



pretrain = NormalPretrain()

pretrain.init_model_ops()

pretrain.train()


