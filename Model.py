#!/usr/bin/env python
# coding: utf-8

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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

pjoin = os.path.join
logger = configure.logger

def conv_bn_relu(name, input_var, weight, stride=1, padding='SAME', bias=None, training=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        x = tf.nn.conv1d(input_var, weight, stride=stride, padding=padding, name='conv1d')
        if bias is not None:
            x = tf.nn.bias_add(x, bias, name='bias_add')

        x = tf.contrib.layers.layer_norm(x)
        x = tf.nn.relu(x, name='relu')

    return x

def conv_2d_bn_relu(name, input_var, weight, stride=[1,1,1,1], padding='SAME', bias=None, training=False):
    # input shape: [batch, in_height, in_width, in_channels]
    # filter shape: [filter_height, filter_width, in_channels, out_channels]
    
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        x = tf.nn.conv2d(input_var, weight, strides=stride, 
                         padding=padding, name='conv2d')
        if bias is not None:
            x = tf.nn.bias_add(x, bias, name='bias_add')

        x = tf.contrib.layers.layer_norm(x)
        x = tf.nn.relu(x, name='relu')

    return x

def fc(name, input_var, weight, bias=None, batchnorm=False, activation=tf.nn.relu, training=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        x = tf.matmul(input_var, weight, name='fc')
        if bias is not None:
            x = tf.nn.bias_add(x, bias, name='bias_add')
        if batchnorm:
            x = tf.contrib.layers.layer_norm(x)
        if activation is not None:
            x = activation(x, name='activation')
    return x

def get_variable(name, shape, dtype=tf.float32, wd=0.0, initializer=tf.glorot_uniform_initializer):
    x = tf.get_variable(name, shape, dtype=dtype, initializer=initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(x), wd, name='weight_loss')
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay)
    return x

def loss_func(pred, label):
    return tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label)


class DeepFeatureNet:
    def construct_weights(self, include_dense=True):
        print('constructing dfn weights')
        weights = dict()
        with tf.variable_scope('deepfeaturenet', reuse=tf.AUTO_REUSE) as scope1:
            if '1D' in configure.cnn_type:
                # 1D-CNN
                weights['conv11_w'] = get_variable('conv11_w', shape=[50, 1, 64], wd=1e-3)
                weights['conv12_w'] = get_variable('conv12_w', shape=[1, 64, 128])
                weights['conv13_w'] = get_variable('conv13_w', shape=[1, 128, 128])

                weights['conv21_w'] = get_variable('conv21_w', shape=[400, 1, 64], wd=1e-3)
                weights['conv22_w'] = get_variable('conv22_w', shape=[1, 64, 128])
                weights['conv23_w'] = get_variable('conv23_w', shape=[1, 128, 128])
            else:
                # 2D-CNN
                weights['conv11_w'] = get_variable('conv11_w', shape=[50, 1, 1, 64], wd=1e-3)
                weights['conv12_w'] = get_variable('conv12_w', shape=[64, 1, 64, 128])
                weights['conv13_w'] = get_variable('conv13_w', shape=[128, 1, 128, 128])

                weights['conv21_w'] = get_variable('conv21_w', shape=[400, 1, 1, 64], wd=1e-3)
                weights['conv22_w'] = get_variable('conv22_w', shape=[64, 1, 64, 128])
                weights['conv23_w'] = get_variable('conv23_w', shape=[128, 1, 128, 128])
            
            if include_dense:
                weights['fc1_w'] = get_variable('fc1_w', shape=[3072, 5])
                weights['fc1_b'] = get_variable('fc1_b', shape=[5])
            
        return weights
    
    def construct_model(self, input_var, weights, is_train, include_dense=True, use_softmax = False):
        # this is actually forwarding method
        print('constructing dfn model')
        
        with tf.variable_scope('deepfeaturenet', reuse=tf.AUTO_REUSE) as scope1:
            with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
                if '1D' in configure.cnn_type:
                    ### 1D-CNN
                    ######## CNNs with small filter size at the first layer #########
                    x1 = conv_bn_relu('conv11', input_var, weights['conv11_w'], stride=8, training=is_train)
                    x1 = conv_bn_relu('conv12', x1, weights['conv12_w'], stride=8, training=is_train)
                    x1 = conv_bn_relu('conv13', x1, weights['conv13_w'], stride=4, training=is_train)
                    x1 = tf.layers.flatten(x1, name='flatten1')

                    ######## CNNs with large filter size at the first layer #########
                    x2 = conv_bn_relu('conv21', input_var, weights['conv21_w'], stride=8, training=is_train)
                    x2 = conv_bn_relu('conv22', x2, weights['conv22_w'], stride=8, training=is_train)
                    x2 = conv_bn_relu('conv23', x2, weights['conv23_w'], stride=4, training=is_train)
                    x2 = tf.layers.flatten(x2, name='flatten2')

                else:
                    ### 2D-CNN
                    x1 = conv_2d_bn_relu('conv11', input_var, weights['conv11_w'], stride=[1,8,8,1], training=is_train)
                    x1 = conv_2d_bn_relu('conv12', x1, weights['conv12_w'], stride=[1,8,8,1], training=is_train)
                    x1 = conv_2d_bn_relu('conv13', x1, weights['conv13_w'], stride=[1,4,4,1], training=is_train)
                    x1 = tf.layers.flatten(x1, name='flatten1')

                    x2 = conv_2d_bn_relu('conv21', input_var, weights['conv21_w'], stride=[1,8,8,1], training=is_train)
                    x2 = conv_2d_bn_relu('conv22', x2, weights['conv22_w'], stride=[1,8,8,1], training=is_train)
                    x2 = conv_2d_bn_relu('conv23', x2, weights['conv23_w'], stride=[1,4,4,1], training=is_train)
                    x2 = tf.layers.flatten(x2, name='flatten2')
                
                
                ######## Aggregate and link two CNNs #########
                y = tf.concat([x1, x2], -1, name='concat')
                if include_dense:
                    if use_softmax:
                        print('create softmax layer')
                        y = fc('fc', y, weights['fc1_w'], bias=weights['fc1_b'], activation=tf.nn.softmax)
                    else:
                        y = fc('fc', y, weights['fc1_w'], bias=weights['fc1_b'], activation=None)
                    
        return y


def flatten(layer, batch_size, seq_len):
    '''
    Used to transform/reshape 4d conv output to 2d matrix
    
    Input(s): Layer - text_cnn layer
              batch_size - how many samples do we feed at once
              seq_len - number of time steps
              
    Output(s): reshaped_layer - the layer with new shape
               number_of_elements - this param is used as a in_size for next layer
    '''
    dims = layer.get_shape()
    print('dims', dims)
    number_of_elements = dims[1:].num_elements()
    
    reshaped_layer = tf.reshape(layer, [batch_size, int(seq_len/2), number_of_elements])
    return reshaped_layer, number_of_elements


class MultiModalNet:
    def construct_weights(self, include_dense=True):
        print('constructing dfn weights')
        weights = dict()
        with tf.variable_scope('featurenet', reuse=tf.AUTO_REUSE) as scope1:

            weights['conv11_w'] = get_variable('conv11_w', shape=[50, 1, 3, 64], wd=1e-3)
            weights['conv12_w'] = get_variable('conv12_w', shape=[64, 1, 64, 128])
            weights['conv13_w'] = get_variable('conv13_w', shape=[128, 1, 128, 128])
            
            weights['conv21_w'] = get_variable('conv21_w', shape=[400, 1, 3, 64], wd=1e-3)
            weights['conv22_w'] = get_variable('conv22_w', shape=[64, 1, 64, 128])
            weights['conv23_w'] = get_variable('conv23_w', shape=[128, 1, 128, 128])
            
            weights['fc1_w'] = get_variable('fc1_w', shape=[3072, 5])
            weights['fc1_b'] = get_variable('fc1_b', shape=[5])
            
        return weights
    
    def construct_model(self, input_var, weights, is_train, include_dense=True, use_softmax = False):
        # perform feed-forward
        print('constructing dfn model')
        
        with tf.variable_scope('featurenet', reuse=tf.AUTO_REUSE) as scope1:
            with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
                # input_var shape: (batch, height, width, channel)
                x1 = conv_2d_bn_relu('conv11', input_var, weights['conv11_w'], stride=[1,8,8,1], training=is_train)
                x1 = conv_2d_bn_relu('conv12', x1, weights['conv12_w'], stride=[1,8,8,1], training=is_train)
                x1 = conv_2d_bn_relu('conv13', x1, weights['conv13_w'], stride=[1,4,4,1], training=is_train)
                y1 = tf.layers.flatten(x1, name='flatten1')
                
                x2 = conv_2d_bn_relu('conv21', input_var, weights['conv21_w'], stride=[1,8,8,1], training=is_train)
                x2 = conv_2d_bn_relu('conv22', x2, weights['conv22_w'], stride=[1,8,8,1], training=is_train)
                x2 = conv_2d_bn_relu('conv23', x2, weights['conv23_w'], stride=[1,4,4,1], training=is_train)
                y2 = tf.layers.flatten(x2, name='flatten2')
                
                y = tf.concat([y1, y2], -1, name='concat')
                if include_dense:
                    if use_softmax:
                        y = fc('fc1', y, weights['fc1_w'], bias=weights['fc1_b'], activation=tf.nn.softmax)
                    else:
                        y = fc('fc1', y, weights['fc1_w'], bias=weights['fc1_b'], activation=None)
                    
        return y

