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
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
import math
import random
import configure
import data_loader
import utils
import preprocessor
from Model import DeepFeatureNet, loss_func, MultiModalNet

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

pjoin = os.path.join
logger = configure.logger

class DataManagement:
    
    def __init__(self, K, subject_id='', seed=None, ds_name=None, channel=None):
        if ds_name != None:
            self.ds_name = ds_name
        else:
            self.ds_name = configure.finetune_dataset
            logger.log('Alert!! use finetune_dataset from configure')
            
        if channel != None:
            self.channel = channel
        else:
            self.channel = configure.finetune['channel']
            logger.log('Alert!! use finetune.channel from configure')
            
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.load_data(K, 
                                                                                                      subject_id, 
                                                                                                      seed)
        
    def load_data(self, K, subject_id, seed):
        '''
        load train and val data
        return in shape x = (ncohorts, nsubjects, nsamples, 3000, 1)
        '''      
        logger.log("K:", K, "subject_id:", subject_id)
        x_train, y_train = [], []
        x_val, y_val = [], []
        x_test, y_test = [], []
        print('Preparing pre-train data..')
        
        n_metaval_tasks = 0
        n_metatrain_tasks = 0
        task_index = -1
        
        self.data_list = []
        
        ds_name = self.ds_name
        logger.log('$$$', ds_name)
        ch = self.channel
        self.data_list.append(ds_name+"_"+ch)
        logger.log('$$', ch)
        subjects = data_loader.get_subject_lists(ds_name, configure.datasets[ds_name]['path'], ch)
        print('subjects:', len(subjects), subjects)

        for subj in subjects:
            if subj == subject_id:
                xx, yy = data_loader.loader(configure.datasets[ds_name]['path'], ch, subj)

                for x, y in zip(xx, yy):
                    x_t, y_t, x_v, y_v, x_te, y_te = [], [], [], [], [], []
                
                    # use only if nsamples/class more than 'min_samples'
                    bp = configure.datasets[ds_name]['bandpass']
                    if bp[0] != None and bp[1] != None:
                        logger.log('bandpass:', ds_name, 'at', bp)
                        x = preprocessor.bandpass_filter(x, low=bp[0], high=bp[1])

                    if len(configure.modals) == 1:
                        print('x', x.shape)
                        
                        if len(x.shape) == 3:
                            x = x[:,:,0] # EEG ONLY (From 3 modals file)
                            x = np.expand_dims(x, axis=-1)
                        else:
                            x = x[:,:,:,0] #UCD
                            
                        logger.log('select 1 modal:', x.shape)
                        
                        
                        if '2D' in configure.cnn_type:
                            # just for testing on 2D-CNN with 1 modal
                            x = np.expand_dims(x, axis=-1)

                            logger.log('‼️‼️ Make sure you use 2D-CNN with 1 modal‼️‼️')
                    else:
                        if len(x.shape) == 3:
                            x = np.expand_dims(x, axis=-2)
                            
                    samples_per_class = utils.get_sample_per_5class(y)
                    logger.log('all:', len(y), 'samples:', samples_per_class)
                    
                    if any([s < K * 3 for s in samples_per_class]):
                        logger.log("‼️ This subject has number of samples < K * 3) ‼️")
                        
                        # use new method to select K samples
                        for c in range(0, 5):
                            num_train = 0
                            num_test = 0
                            num_val = 0

                            x_c = x[y==c]
                            y_c = y[y==c]

                            if seed != -1:
                                x_c, y_c = utils.shuffle_data(x_c, y_c, logger=logger, 
                                                              fix_val_sample=True, seed_no=seed)
                            else:
                                # use first K samples as training samples -> no shuffle
                                logger.log('use first K samples to train')

                            if samples_per_class[c] >= K * 3:
                                # select as normal
                                logger.log('class', c, 'samples are enough:', samples_per_class[c])

                                # 1. Select K samples/class to train
                                x_t.extend(x_c[:K])
                                y_t.extend(y_c[:K])

                                # 2. Select K samples/class to val
                                x_v.extend(x_c[K:K*2])
                                y_v.extend(y_c[K:K*2])

                                # 3. The rest are for testing
                                x_te.extend(x_c[K*2:])
                                y_te.extend(y_c[K*2:])

                            else:
                                ####################################
                                ## SELECT TO TRAIN -> VAL -> TEST ##
                                ####################################

                                # 1. Select K samples/class to train
                                num_train = min(K, samples_per_class[c])
                                logger.log('select', num_train, 'samples to train.')
                                x_t.extend(x_c[:num_train])
                                y_t.extend(y_c[:num_train])
                                samples_per_class[c] -= num_train
                                print('After train set, samples_per_class', c, ' remains:', samples_per_class[c])

                                if samples_per_class[c] > 0:
                                    # 2. Select K samples/class to validate
                                    num_val = min(samples_per_class[c], K)
                                    logger.log('select', num_val, 'samples to validate.')
                                    x_v.extend(x_c[num_train:num_train+num_val])
                                    y_v.extend(y_c[num_train:num_train+num_val])
                                    samples_per_class[c] -= num_val
                                    print('After val set, samples_per_class', c, ' remains:', samples_per_class[c])

                                    if samples_per_class[c] > 0:
                                        x_te.extend(x_c[num_train+num_val: ])
                                        y_te.extend(y_c[num_train+num_val: ])
                                        logger.log('select remaining samples to test:', len(x_c[num_train+num_val: ]))

                                print(num_train, num_val, len(x_c[num_train+num_val: ]))
                                assert num_train + num_val + len(x_c[num_train+num_val: ]) == len(x_c)

                        assert len(x_te) == len(y_te)
                        assert len(x_v) == len(y_v)
                        assert len(x_t) == len(y_t)
                        assert len(x_te) + len(x_v) + len(x_t) == len(x)
                        
                        y_t, y_v, y_te = np.array(y_t), np.array(y_v), np.array(y_te)
                        logger.log('train:', len(y_t), 'samples:', [len(y_t[y_t==clid]) for clid in range(0, 5)])
                        logger.log('val:', len(y_v), 'samples:', [len(y_v[y_v==clid]) for clid in range(0, 5)])
                        logger.log('test:', len(y_te), 'samples:', [len(y_te[y_te==clid]) for clid in range(0, 5)])

                    else:
                        # pick K samples/class to fine-tune, the rest are for validation
                        for c in np.unique(y):
                            x_c = x[y==c]
                            y_c = y[y==c]

                            x_c, y_c = utils.shuffle_data(x_c, y_c, logger=logger, fix_val_sample=True, seed_no=seed)
                            x_t.extend(x_c[:K])
                            y_t.extend(y_c[:K])
                            x_v.extend(x_c[K:K*2])
                            y_v.extend(y_c[K:K*2])
                            x_te.extend(x_c[K*2:])
                            y_te.extend(y_c[K*2:])

                        assert len(x_t) == len(y_t) == K * 5
                        assert len(x_v) == len(y_v) == K * 5
                        assert len(x_t) + len(x_v) + len(x_te) == len(x)
                        assert len(y_t) + len(y_v) + len(y_te) == len(y)
                        
                        logger.log('train:', len(y_t), 'samples:', utils.get_sample_per_class(np.array(y_t)))
                        logger.log('val:', len(y_v), 'samples:', utils.get_sample_per_class(np.array(y_v)))
                        logger.log('test:', len(y_te), 'samples:', utils.get_sample_per_class(np.array(y_te)))

                    x_train.append(x_t)
                    y_train.append(y_t)
                    x_val.append(x_v)
                    y_val.append(y_v)
                    x_test.append(x_te)
                    y_test.append(y_te)
                    
                    break # use first night only
                    
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_val = np.array(x_val)
        y_val = np.array(y_val)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        logger.log('x_train', x_train.shape, 'y_train', y_train.shape)
        logger.log('x_val', x_val.shape, 'y_val', y_val.shape)
        logger.log('x_test', x_test.shape, 'y_test', y_test.shape)
        
        assert len(x_train) == len(y_train) == len(x_val) == len(y_val) == len(x_test) == len(y_test)
        
        return x_train, y_train, x_val, y_val, x_test, y_test
                
    def get_data(self):
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test


class Finetune(object):
    def __init__(self, k, subject_id, model_pretrain_path, 
                 finetune_weight_path = None, seed = None, 
                 nepochs = None, lr = None,
                 ds_name = None, channel = None
                ):
        self.interval_save_model = 5
        
        self.data_manager = DataManagement(k, subject_id, seed, ds_name, channel)
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = self.data_manager.get_data()
        self.batch_size = k * 5
        
        self.nmodals = len(configure.modals)
        self.model_pretrain_path = model_pretrain_path
        
        if nepochs is None:
            self.nepochs = configure.finetune['nepochs']
            logger.log('!!!!!!!! Caution Use nepochs from configure:', nepochs, '!!!!!!!!!')
        else:
            self.nepochs = nepochs
            
        if type(self.nepochs) == list:
            self.nepochs_list = self.nepochs
            self.nepochs = np.max(self.nepochs)
        else:
            raise Exception("nepochs should be a list.")
        
        if lr is None:
            self.lr = configure.finetune['lr']
            logger.log('!!!!!!!! Caution Use lr from configure:', lr, '!!!!!!!!!')
        else:
            self.lr = lr
            
        if self.nmodals > 1:
            logger.log('Using.. MultiModalNet')
            self.pretrain_net = 'featurenet'
            self.model = MultiModalNet()
        elif configure.nepochs_per_sample==1:
            logger.log('Using.. DeepFeatureNet')
            self.model = DeepFeatureNet()
            self.pretrain_net = 'deepfeaturenet'
        else:
            raise Exception('Model incorrect')
            
        if finetune_weight_path:
            self.finetune_weight_path = finetune_weight_path
            logger.log('finetune_weight_path:', finetune_weight_path)
        else:
            self.finetune_weight_path = configure.finetune_weight_path
            logger.log('finetune_weight_path (from configure):', finetune_weight_path)
    
    def train_op(self, weights, loss, lr):
        grad = tf.gradients(loss, list(weights.values()))

        assigns_op = []
        for w, g in zip(weights, grad):
            print(w, weights[w])
            assigns_op.append(tf.assign_sub(weights[w], tf.scalar_mul(lr, g)))
            
        return assigns_op
    
    
    def init_model_ops(self):
        
        self.weights = self.model.construct_weights()

        self.is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
        if self.nmodals == 1 and '1D' in configure.cnn_type:
            self.inputs = inputs = tf.placeholder(tf.float32, 
                                                  shape=[None, 3000*configure.nepochs_per_sample, 1], 
                                                  name='inputs')
        else:
            self.inputs = inputs = tf.placeholder(tf.float32, 
                                                  shape=[None, 3000*configure.nepochs_per_sample, 1, self.nmodals], 
                                                  name='inputs')
            
        self.labels = labels = tf.placeholder(tf.int32, shape=[None], name='labels')
        self.labels_one_hot = tf.one_hot(self.labels, 5, axis=-1)
        
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            self.outputs = self.model.construct_model(self.inputs, self.weights, self.is_train)
            self.reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
            self.loss = loss = loss_func(self.outputs, self.labels_one_hot)
            self.total_loss = tf.reduce_mean(tf.add(loss, 
                                                    self.reg_loss, name='total_loss'))
            self.total_accuracy = tf.reduce_mean(tf.contrib.metrics.accuracy(tf.argmax(self.outputs, 1), 
                                                                             tf.argmax(self.labels_one_hot, 1)))

        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            self.apply_grads_op = self.train_op(self.weights, self.total_loss, self.lr)
            self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)
            self.dfn_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                                                       scope=self.pretrain_net),
                                            max_to_keep=None)
    
    
    def run_epoch(self, x, y, sess, training=False):
        sum_losses = 0
        n_batches = 0
        total_samples = 0
        correct_prediction = 0
        y_true_all = []
        y_pred_all = []
        
        if training:
            
            p = range(0, len(x))

            start_time = timeit.default_timer()
            nbatches = len(x)
            logger.log('training for {} mini-batches'.format(nbatches))
            utils.printProgressBar(0, nbatches, prefix = 'Progress:', suffix = 'Complete', length = 50)
            for i in range(nbatches):
                x_select, y_select = np.array(x[p[i]]), y[p[i]]
                
                _, loss_value, y_true, y_logits = sess.run([self.apply_grads_op, self.total_loss, 
                                                            self.labels, self.outputs], 
                                                           feed_dict={self.inputs: x_select,
                                                                      self.labels: y_select,
                                                                      self.is_train: True
                                                                     })

                sum_losses += loss_value
                n_batches += 1
                y_pred = np.argmax(y_logits, axis=-1)
                correct_prediction += (y_true == y_pred).sum()
                y_true_all.append(y_true)
                y_pred_all.append(y_pred)
                total_samples += len(y_true)

                utils.printProgressBar(i + 1, nbatches, prefix = 'Progress:', suffix = 'Complete', length = 50)

            sum_losses /= n_batches
            acc = correct_prediction/total_samples
            y_true_all = np.hstack(y_true_all)
            y_pred_all = np.hstack(y_pred_all)
            f1 = f1_score(y_true_all, y_pred_all, average='macro')
            duration = timeit.default_timer() - start_time
            
        else:       
            sum_losses, y_true, y_logits = sess.run([self.total_loss,
                                                     self.labels, self.outputs], 
                                                    feed_dict={self.inputs: x,
                                                               self.labels: y,
                                                               self.is_train: False})

            y_pred = np.argmax(y_logits, axis=-1)
            correct_prediction += (y_true == y_pred).sum()
            total_samples += len(y_true)
                
            acc = correct_prediction/total_samples
            f1 = f1_score(y_true, y_pred, average='macro')
        
        return sum_losses, acc, f1, y_true, y_pred
    
            
    def train(self):
        model_pretrain_path = self.model_pretrain_path
        nepochs = self.nepochs
        
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            
            if model_pretrain_path:
                # restore pre-train weights
                ww_tmp_before = sess.run(self.weights['conv11_w'])
                
                if 'random' in model_pretrain_path:
                    self.dfn_saver.restore(sess, pjoin(model_pretrain_path, "model.ckpt"))
                    logger.log('restoring weights from', pjoin(model_pretrain_path, "model.ckpt"))
                else:
                    self.dfn_saver.restore(sess, pjoin(model_pretrain_path, "model_best.ckpt"))
                    logger.log('restoring weights from', pjoin(model_pretrain_path, "model_best.ckpt"))

                ww_tmp_after = sess.run(self.weights['conv11_w'])
                if self.nmodals == 1 and '1D' in configure.cnn_type:
                    print('weights has been restored already? :', ww_tmp_before[0][0][0] != ww_tmp_after[0][0][0])
                    assert ww_tmp_before[0][0][0] != ww_tmp_after[0][0][0]
                else:
                    print('weights has been restored already? :', ww_tmp_before[0][0][0][0] != ww_tmp_after[0][0][0][0])
                    assert ww_tmp_before[0][0][0][0] != ww_tmp_after[0][0][0][0]
                 
            else:
                logger.log('randomly initialized model.')
            

            finetune_weight_path = self.finetune_weight_path
            results = {}
            
            for iter_no in self.nepochs_list:
                results['epoch_'+str(iter_no)] = {'val_loss': 0, 'f1': []}
            print(results)
            
            for record_id in range(0, len(self.x_train)):
                if record_id == 1:
                    raise Exception("This code have not supported 2 nights data yet.")
                
                print()
                train_acc, val_acc = [], []
                train_loss, val_loss = [], []
                train_f1, val_f1 = [], []
                
                logger.log('==================== Record:', record_id, '====================')
                x_train = self.x_train[record_id]
                y_train = self.y_train[record_id]
                x_val = self.x_val[record_id]
                y_val = self.y_val[record_id]
            
                for iter_no in range(0, nepochs):
                    logger.log('********* ITER:', iter_no, '**********')

                    # Run training for 1 epoch
                    sum_losses, acc, f1, ytrue, ypred = self.run_epoch([x_train], [y_train], 
                                                                       sess, training=True)
                    logger.log('Training Loss: {0:.4f}, Accuracy: {1:.4f}, F1: {2:.4f}'.format(sum_losses, 
                                                                                               acc, 
                                                                                               f1))
                    train_acc.append(acc)
                    train_loss.append(sum_losses)
                    train_f1.append(f1)
                    
                    sum_losses_val, acc_val, f1_val, ytrue_val, ypred_val = self.run_epoch(x_val, 
                                                                                           y_val, 
                                                                                           sess, 
                                                                                           training=False)
                    logger.log('Validation Loss: {0:.4f}, Acc: {1:.4f}, F1: {2:.4f}'.format(sum_losses_val,
                                                                                            acc_val, 
                                                                                            f1_val))
                    val_acc.append(acc_val)
                    val_loss.append(sum_losses_val)
                    val_f1.append(f1_val)

                    if iter_no % self.interval_save_model == 0 or iter_no == nepochs-1:
                        np.savez(pjoin(finetune_weight_path, 'results'), 
                                                 train_acc=train_acc, 
                                                 train_loss=train_loss,
                                                 val_acc=val_acc, 
                                                 val_loss=val_loss,
                                                 val_f1=val_f1,
                                                 yval_true=ytrue_val, yval_pred=ypred_val)

                        weight_path = pjoin(finetune_weight_path, "model.ckpt")
                        self.saver.save(sess, weight_path)
                        logger.log('saved weight to', weight_path)

                    if iter_no+1 in self.nepochs_list:
                        weight_path = pjoin(finetune_weight_path, "model_iter"+str(iter_no+1)+".ckpt")
                        self.saver.save(sess, weight_path)
                        logger.log('saved loss, f1 & saved weight to', weight_path)
                        
                        results['epoch_'+str(iter_no+1)]['val_loss'] = sum_losses_val
                        results['epoch_'+str(iter_no+1)]['f1'] = utils.get_per_class_f1(ytrue_val, ypred_val)[1]
                        print('results:', results)
                        
                    if len(self.x_train) == 0:
                        # for no training subjects -> just validate 1 time only
                        break
                    
        return results
    
                    
    def predict(self, finetune_weight_path, best_epoch=None):
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            
            if best_epoch != None:
                model_path = pjoin(finetune_weight_path, "model_iter"+str(best_epoch)+".ckpt")
            else:
                model_path = pjoin(finetune_weight_path, "model.ckpt")
                
            self.saver.restore(sess, model_path)
            logger.log('restore for test from:', model_path)
            
            loss, acc, f1, kappa = [], [], [], []
            for record_id in range(0, len(self.x_test)):
                sum_losses_val, acc_val, f1_val, ytrue_val, ypred_val = self.run_epoch(self.x_test[record_id], 
                                                                                       self.y_test[record_id], 
                                                                                       sess, 
                                                                                       training=False)
                logger.log('Test Loss: {0:.4f}, Acc: {1:.4f}, F1: {2:.4f}'.format(sum_losses_val,
                                                                                  acc_val,
                                                                                  f1_val))
                
                loss.append(sum_losses_val)
                acc.append(acc_val)
                f1.append(utils.get_per_class_f1(ytrue_val, ypred_val)[1])
                kappa.append(cohen_kappa_score(ytrue_val, ypred_val))
                
        return loss, acc, f1, kappa
