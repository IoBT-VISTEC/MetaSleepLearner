# -*- coding: utf-8 -*-
import os
os.getcwd()

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

os.environ["CUDA_VISIBLE_DEVICES"]="6"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


pjoin = os.path.join
logger = configure.logger
logger.metatrain_log()

class DataManagement:
    
    def __init__(self):
        self.X_metatrain, self.y_metatrain, self.X_metaval, self.y_metaval = self.load_all_data()
        
    def load_data(self, data_type='train'):
        conf = configure.maml
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
                
                for subj in subjects:
                    xx, yy = data_loader.loader(configure.datasets[dataset]['path'], channel, subj)
                    
                    for x, y in zip(xx, yy):
                        class_samples = utils.get_sample_per_class(y)
                        
                        # To filter out the same set of subjects as submission version (K=10)
                        logger.log('‼️ K =', K, 'but filter out if subjects contain < 30 samples')
                            
                        if len(class_samples) == 5 and all([cl >= 30 and cl >= K*2 for cl in class_samples]):
                            # use only if nsamples/class more than 'K' * 2
                            bp = configure.datasets[dataset]['bandpass']
                            if bp[0] != None and bp[1] != None:
                                logger.log('bandpass:', dataset, 'at', bp)
                                x = preprocessor.bandpass_filter(x, low=bp[0], high=bp[1])
                            
                            if len(configure.modals) == 1:
                                if x.shape[-1] == 3:
                                    x = x[:,:,0] # EEG ONLY (From 3 modals file)
                                
                                x = np.expand_dims(x, axis=-1)
                            else:
                                x = np.expand_dims(x, axis=-2)
                                
                            results_x[task_index].append(x)
                            results_y[task_index].append(y)
                                
                            logger.log('$ added subj: {} {}'.format(subj, class_samples))     
                        else:
                            logger.log('$ removed subj: {} {} (< K*2)'.format(subj, class_samples))  
                            
        results_x, results_y = np.array(results_x), np.array(results_y)
        logger.log('n_meta'+data_type+'_tasks =', len(results_x)) 
        for tid, d in enumerate(results_x):
            logger.log('task:', tid, data_type+':', len(d), 'records, x[0]:', d[0].shape)
            
        return results_x, results_y
        
    def load_all_data(self):
        '''
        load train and val data
        return in shape x = (ntasks, nsubjects, nsamples, 3000, 1, nmodals(optional))
        '''      
        logger.log('PRE-TRAINING DATA..')
        x_train, y_train = self.load_data('train')
        x_val, y_val = self.load_data('validate')
        
        return x_train, y_train, x_val, y_val
                
    def get_data(self):
        return self.X_metatrain, self.y_metatrain, self.X_metaval, self.y_metaval 


class MAML(object):
    def __init__(self):
        self.train_loss = []
        self.valid_loss = []
        self.train_acc = []
        self.valid_acc = []
        self.train_f1 = []
        self.valid_f1 = []
        self.interval_save_model = 5
        self.nmodals = len(configure.modals)
        logger.log('nmodals:', self.nmodals)
        
        self.ntasks = configure.maml['ntasks']
        self.nsamples_per_class = configure.maml["K"]
        
        self.data_manager = DataManagement()
        self.x_train, self.y_train, self.x_val, self.y_val = self.data_manager.get_data()
        
        if self.nmodals == 1:
            logger.log('‼️configure.cnn_type:', configure.cnn_type)
        
        if self.nmodals > 1:
            logger.log('Using.. MultiModalNet')
            self.model = MultiModalNet()
        elif configure.nepochs_per_sample == 1:
            logger.log('Using.. DeepFeatureNet')
            self.model = DeepFeatureNet()
        else:
            raise Exception('Model incorrect')
        
    def init_model_ops(self):
        self.num_updates = num_updates = configure.maml['num_updates']
        ntasks = self.ntasks
        
        self.weights = self.model.construct_weights()

        self.is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
        self.meta_lr = tf.placeholder(tf.float32, shape=(), name='meta_lr')
        if self.nmodals == 1 and '1D' in configure.cnn_type:
            self.inputa = tf.placeholder(tf.float32, 
                                         shape=[ntasks, None, 3000*configure.nepochs_per_sample, 1], 
                                         name='inputsa')
            self.inputb = tf.placeholder(tf.float32, 
                                     shape=[ntasks, None, 3000*configure.nepochs_per_sample, 1], 
                                     name='inputsb')
        else:
            self.inputa = tf.placeholder(tf.float32, 
                                         shape=[ntasks, None, 3000*configure.nepochs_per_sample, 1, self.nmodals], 
                                         name='inputsa')
            self.inputb = tf.placeholder(tf.float32, 
                                     shape=[ntasks, None, 3000*configure.nepochs_per_sample, 1, self.nmodals], 
                                     name='inputsb')
            
            
        self.labela = tf.placeholder(tf.int32, shape=[ntasks, None], name='labelsa')
        self.labelb = tf.placeholder(tf.int32, shape=[ntasks, None], name='labelsb')
        
        self.labela_one_hot = tf.one_hot(self.labela, 5, axis=-1)
        self.labelb_one_hot = tf.one_hot(self.labelb, 5, axis=-1)
        
        ##### meta-learn #####
        self.outputas, self.outputbs, self.lossesa, self.lossesb, self.accuraciesa, self.accuraciesb = [], [], [], [], [], []
            
        def cond(outputa, outputb, lossa, lossb, acca, accb, i, iters):
            return tf.less(i, iters)

        def task_metalearn(outputa, outputb, lossa, lossb, acca, accb, i, iters):
            inputa = tf.gather(self.inputa, i)
            labela = tf.gather(self.labela_one_hot, i)
            inputb = tf.gather(self.inputb, i)
            labelb = tf.gather(self.labelb_one_hot, i)
            print(inputa.shape, inputb.shape)
            
            weights = self.weights

            task_outputbs, task_lossesb, task_accuraciesb = [], [], []
            task_outputa = self.model.construct_model(inputa, weights, self.is_train)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            task_lossa = tf.reduce_mean(loss_func(task_outputa, labela))
            grad = tf.gradients(task_lossa, list(weights.values()))

            fast_weights = dict()
            for w, g in zip(weights, grad):
                fast_weights[w] = tf.subtract(weights[w], tf.scalar_mul(configure.maml['update_lr'], g))

            output = self.model.construct_model(inputb, fast_weights, self.is_train)
            task_outputbs.append(output)
            task_lossesb.append(tf.reduce_mean(loss_func(output, labelb)))

            for j in range(num_updates - 1):
                print(j, 'adapting..')

                output = self.model.construct_model(inputa, fast_weights, self.is_train)
                loss = tf.reduce_mean(loss_func(output, labela))

                grad = tf.gradients(loss, list(fast_weights.values()))
                for w, g in zip(fast_weights, grad):
                    fast_weights[w] = tf.subtract(fast_weights[w], tf.scalar_mul(configure.maml['update_lr'], g)) 

                output = self.model.construct_model(inputb, fast_weights, self.is_train)
                task_outputbs.append(output)
                task_lossesb.append(tf.reduce_mean(loss_func(output, labelb)))
                
            task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(task_outputa, 1), tf.argmax(labela, 1))
            for j in range(num_updates):
                task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(task_outputbs[j], 1), 
                                                                    tf.argmax(labelb, 1)))

            outputa = outputa.write(i, task_outputa)
            outputb = outputb.write(i, task_outputbs)
            lossa = lossa.write(i, task_lossa)
            lossb = lossb.write(i, task_lossesb)
            acca = acca.write(i, task_accuracya)
            accb = accb.write(i, task_accuraciesb)
            
            return [outputa, outputb, lossa, lossb, acca, accb, 
                    tf.add(i, 1), iters]
        ####### end of task_metalearn #####
        
        iters = tf.constant(ntasks)
        task_outputa = tf.TensorArray(dtype=tf.float32,size=1,dynamic_size=True,clear_after_read=False)
        task_outputb = tf.TensorArray(dtype=tf.float32,size=1,dynamic_size=True,clear_after_read=False)
        task_lossa = tf.TensorArray(dtype=tf.float32,size=1,dynamic_size=True,clear_after_read=False)
        task_lossb = tf.TensorArray(dtype=tf.float32,size=1,dynamic_size=True,clear_after_read=False)
        task_acca = tf.TensorArray(dtype=tf.float32,size=1,dynamic_size=True,clear_after_read=False)
        task_accb = tf.TensorArray(dtype=tf.float32,size=1,dynamic_size=True,clear_after_read=False)
        result = tf.while_loop(cond, task_metalearn, [task_outputa, task_outputb, 
                                                       task_lossa, task_lossb,
                                                       task_acca, task_accb, 
                                                      0, iters],
                               swap_memory=True)
        

        # transpose from (ntasks, num_updates, shape) -> (num_updates, ntasks, shape)
        self.outputas, self.outputbs, self.lossesa, self.lossesb, self.accuraciesa, self.accuraciesb, _i, _iter = result
        
        self.outputas = self.outputas.stack()
        self.outputbs = tf.transpose(self.outputbs.stack(), [1, 0, 2, 3]) 
        self.lossesa = self.lossesa.stack()
        self.lossesb = tf.transpose(self.lossesb.stack(), [1, 0])
        self.accuraciesa = self.accuraciesa.stack()
        self.accuraciesb = tf.transpose(self.accuraciesb.stack(), [1, 0])

        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            self.total_loss1 = total_loss1 = tf.reduce_sum(self.lossesa) / tf.cast(ntasks, tf.float32)
            self.losses2 = losses2 = [tf.reduce_sum(self.lossesb[j]) / tf.cast(ntasks, tf.float32)
                                                  for j in range(num_updates)]
            self.reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
            self.total_losses2 = total_losses2 = tf.add(losses2, self.reg_loss, name='total_losses2')
            
            self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(self.accuraciesa) / tf.cast(ntasks, tf.float32)
            self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(self.accuraciesb[j]) / tf.cast(ntasks, tf.float32)
                                                          for j in range(num_updates)]

        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            self.optimizer = optimizer = tf.train.AdamOptimizer(self.meta_lr)
            self.grads_and_vars = self.optimizer.compute_gradients(total_losses2[num_updates-1], tf.trainable_variables())
            self.apply_grads_op = self.optimizer.apply_gradients(self.grads_and_vars, name='apply_grads_op')
            self.saver = tf.train.Saver(tf.trainable_variables())
                    
    
    def prepare_val_data(self, x, y):
        
        task_inputa, task_inputb, task_labela, task_labelb = [], [], [], []
        for task_index in range(0, len(x)):
            for record_index in range(0, len(x[task_index])):
                
                x_tmp = x[task_index][record_index]
                y_tmp = y[task_index][record_index]
                
                inputa_x, inputa_y, inputb_x, inputb_y = utils.pick_samples(x_tmp, 
                                                                            y_tmp,
                                                                            self.nsamples_per_class * 5,
                                                                            logger=logger,
                                                                            fix_val_sample=False,
                                                                            request_b=True)
                task_inputa.append(inputa_x)
                task_labela.append(inputa_y)
                task_inputb.append(inputb_x)
                task_labelb.append(inputb_y)
        
        if len(task_inputa) != self.ntasks:
            raise Exception('Please specify val list in the same number of tasks')
            
        return task_inputa, task_inputb, task_labela, task_labelb
                
        
    def prepare_data(self, x, y):
        # sample dataset
        task_list = list(range(0, len(x)))
        tasks = []
        task_inputa, task_inputb, task_labela, task_labelb = [], [], [], []

        for task_dup_round in range(0, int(self.ntasks/len(x))):
            random.shuffle(task_list)
            tasks.extend(task_list)
        logger.log('tasks =', tasks)

        # sample records from that task
        records = []
        # so it doesn't pick the same subject in the same meta-iteration
        used_subjects = [[] for i in range(0, len(task_list))] 
        for task_id in tasks:
            while True:
                record_id = random.choice(range(0, len(x[task_id])))
                if not record_id in used_subjects[task_id]:
                    break
            records.append(record_id)
            used_subjects[task_id].append(record_id)

            x_tmp = x[task_id][record_id]
            y_tmp = y[task_id][record_id]
           
            inputa_x, inputa_y, inputb_x, inputb_y = utils.pick_samples(x_tmp, 
                                                                        y_tmp,
                                                                        self.nsamples_per_class * 5,
                                                                        logger=logger,
                                                                        fix_val_sample=False,
                                                                        request_b=True)
            task_inputa.append(inputa_x)
            task_labela.append(inputa_y)
            task_inputb.append(inputb_x)
            task_labelb.append(inputb_y)
            
        logger.log('records:', records)
            
        return task_inputa, task_inputb, task_labela, task_labelb
            
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
        
    def train(self):
        
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            
            train_acc1, train_acc2 = [], []
            train_loss1, train_loss2 = [], []
            val_acc1, val_loss1 = [], []
            val_acc2, val_loss2 = [], []
            best_loss = None
            
            meta_lr = configure.maml['meta_lr']
            update_lr = configure.maml['update_lr']

            weight_name_list = ['conv11_w', 'conv12_w', 'fc1_w', 'fc1_b']
            weight_list = [self.weights[wi] for wi in weight_name_list]
            prev_weights = [None for i in range(0, len(weight_name_list))]
            grads_and_vars = []
            
            for iter_no in range(0, configure.maml['niters']):
                logger.log('########## ITER:', iter_no, '##########')
                
                ## prepare data for 1 iter
                inputsa, inputsb, labelsa, labelsb = self.prepare_data(self.x_train, self.y_train)
                
                if self.nmodals == 1 and '2D' in configure.cnn_type:
                    inputsa = np.expand_dims(inputsa, axis=-1)
                    inputsb = np.expand_dims(inputsb, axis=-1)
                    
                outputa, loss1, total_acc1, outputb, loss2, total_acc2,  _ = sess.run([self.outputas, 
                                                                                          self.total_loss1,  
                                                                                          self.total_accuracy1, 
                                                                                          self.outputbs,
                                                                                          self.total_losses2,
                                                                                          self.total_accuracies2,
                                                                                          self.apply_grads_op], 
                                                                               feed_dict={self.inputa: inputsa, 
                                                                                          self.labela: labelsa,
                                                                                          self.inputb: inputsb, 
                                                                                          self.labelb: labelsb, 
                                                                                          self.is_train: True,
                                                                                          self.meta_lr: meta_lr
                                                                                         })
                
                print('## TRAIN ##')
                print('meta_lr', meta_lr, 'update_lr', update_lr)
                print('loss1', loss1)
                print('total_acc1', total_acc1)
                print('loss2', loss2)
                print('total_acc2', total_acc2)
                print()
                train_acc1.append(total_acc1)
                train_loss1.append(loss1)
                train_acc2.append(total_acc2)
                train_loss2.append(loss2)
                
                results = sess.run(weight_list)
                print('weights:', weight_name_list)
                if all([p == None for p in prev_weights]) or self.is_any_weight_changed(prev_weights, 
                                                                                        results, weight_name_list):
                    for w_index, weight in enumerate(results):
                        w = results[w_index]
                        while True:
                            if type(w) is list or type(w) is np.ndarray:
                                w = w[0]
                            else:
                                break
                        print('copy:', weight_name_list[w_index], w)
                        prev_weights[w_index] = w
                else:
                    raise Exception('No weight changed.')
                    
                    
                print('## VALIDATE ##')    
                inputsa_val, inputsb_val, labelsa_val, labelsb_val = self.prepare_val_data(self.x_val, self.y_val)
                if self.nmodals == 1 and '2D' in configure.cnn_type:
                    inputsa_val = np.expand_dims(inputsa_val, axis=-1)
                    inputsb_val = np.expand_dims(inputsb_val, axis=-1)
                    
                outputa_val, loss_val, acc_val, outputb_val, loss2_val, acc2_val = sess.run([self.outputas, 
                                                                                          self.total_loss1,  
                                                                                          self.total_accuracy1, 
                                                                                          self.outputbs,
                                                                                          self.total_losses2,
                                                                                          self.total_accuracies2], 
                                                                               feed_dict={self.inputa: inputsa_val, 
                                                                                          self.labela: labelsa_val,
                                                                                          self.inputb: inputsb_val, 
                                                                                          self.labelb: labelsb_val, 
                                                                                          self.is_train: False
                                                                                         })
                print('loss1', loss_val)
                print('total_acc1', acc_val)
                print('loss2', loss2_val)
                print('total_acc2', acc2_val)
                print()
                val_acc1.append(acc_val)
                val_loss1.append(loss_val)
                val_acc2.append(acc2_val)
                val_loss2.append(loss2_val)
                
                if best_loss == None or loss2_val[-1] < best_loss:
                    logger.log('Best iter:', iter_no, 'saved!')
                    best_iter = iter_no
                    best_loss = loss2_val[-1]
                    self.saver.save(sess, pjoin(configure.meta_weight_path, "model_best.ckpt"))
                    logger.log('saved at', pjoin(configure.meta_weight_path, "model_best.ckpt"))
                    y_valid_best = labelsb_val
                    output_best = outputb_val
                    
                    
                if iter_no % self.interval_save_model == 0 or iter_no == configure.maml['niters']-1:
                    self.saver.save(sess, pjoin(configure.meta_weight_path, "model.ckpt"))
                    logger.log('saved at', pjoin(configure.meta_weight_path, "model.ckpt"))
                    
                    np.savez(pjoin(configure.meta_weight_path, 'results'), \
                                             train_acc1=train_acc1, train_acc2=train_acc2, \
                                             train_loss1=train_loss1, train_loss2=train_loss2, \
                                             train_outputa=outputa, train_outputb=outputb, \
                                             train_labela=labelsa, train_labelb=labelsb, \
                                             val_acc1=val_acc1, val_loss1=val_loss1, \
                                             val_acc2=val_acc2, val_loss2=val_loss2, \
                                             yval_true=labelsb_val, yval_pred=outputb_val, \
                                             yval_true_best=y_valid_best, yval_pred_best=output_best, \
                                             best_iter=best_iter, best_loss=best_loss
                            )
                   
                    del inputsa, inputsb, labelsa, labelsb, outputa, outputb
                    del inputsa_val, inputsb_val, labelsa_val, labelsb_val
                    del outputa_val, loss_val, acc_val, outputb_val, loss2_val, acc2_val


maml = MAML()

maml.init_model_ops()

maml.train()

