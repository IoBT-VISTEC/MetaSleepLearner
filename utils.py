# -*- coding: utf-8 -*-
import itertools
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
import os

# Print iterations progress (https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console)
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def norm(data, nmodals, logger=None):
    # data.shape = (nsamples, 3000, 1, <nmodals>)
    nsamples = data.shape[0]
    npoints = data.shape[1]
    if logger:
        logger.log('normalizing data shape:', data.shape, nmodals)
    
    def scale(dat):
        d = np.concatenate(dat, axis=0).reshape(-1)
        assert d.shape[0] == dat.shape[0]*dat.shape[1]
        assert len(d.shape) == 1 #d.shape = (nsamples*3000,)
        
        d = d.reshape(-1, 1) # to scale in sample (not feature)
        scaler = StandardScaler()
        d = scaler.fit_transform(d) 
        print('mean:', scaler.mean_, d.shape)
        assert len(scaler.mean_) == 1 # one subject should have only one mean
        
        return d.reshape(nsamples, npoints, 1)
    
    if nmodals > 1:
        # multimodal
        results = []
        for i in range(nmodals):
            if len(data.shape) == 4:
                dat = data[:, :, :, i]
            elif len(data.shape) == 5:
                dat = data[:, :, :, :, i]
            else:
                dat = data[:, :, i]
            dat = scale(dat)
            results.append(dat)
            
        results = np.array(results)
        assert len(results) == nmodals # results.shape = (3, nsamples, 3000, 1)
        assert len(results.shape) == 4
        
        results = np.transpose(results, (1, 2, 3, 0))
        assert results.shape[-1] == nmodals
        return results
        
    else:
        # one modal
        return scale(data)


def get_per_class_acc(yt, yp):
    classes = np.unique(yt)
    all_correct = 0
    all_samp = 0
    txt = ''
    acc_arr = []
    for c in classes:
        correct = len([l for l, label in zip(yp, yt) if l==label and label==c])
        all_samp_this_class = len(yt[yt==c])
        txt += 'class {}: {} / {} = '.format(c, correct, all_samp_this_class)
        acc = correct / all_samp_this_class * 100
        txt += '%.2f \n' % (acc, )
        all_correct += correct
        all_samp += all_samp_this_class
        acc_arr.append(acc)

    txt += 'All correct: ' + str(all_correct) + '/' + str(all_samp) + ' = %.2f' % (all_correct / all_samp,)
    return txt, acc_arr

def get_per_class_f1(y_true, y_pred):
    precision = np.zeros(5)
    recall = np.zeros(5)
    f1 = np.zeros(5)
    eps = 1e-15

    for c in range(5):
        tp = ((y_true == c) & (y_pred == c)).sum()
        fp = ((y_true != c) & (y_pred == c)).sum()
        fn = ((y_true == c) & (y_pred != c)).sum()
        tn = ((y_true != c) & (y_pred != c)).sum()
        precision[c] = (tp)/(tp+fp+eps)
        recall[c] = (tp)/(tp+fn+eps)
        f1[c] = (2*precision[c]*recall[c])/(precision[c]+recall[c]+eps)

    return 'F1: {}'.format(f1), f1

def get_sample_per_class(y):
    classes = np.unique(y)
    return [len(y[y==c]) for c in classes]

def get_sample_per_5class(y):
    return [len(y[y==c]) for c in range(0, 5)]


def shuffle_data(x, y, logger=None, fix_val_sample=False, seed_no=0):
    # shape(x) = (nsamples, 3000, 1)
    if fix_val_sample:
        np.random.seed(seed_no)
    else:
        if logger:
            logger.log('ALERT!! No fixing random seed.')
        # should be fixed whie fine-tuning
        # raise Exception('fix_val_sample SHOULD BE TRUE!!')
    
    p = np.random.permutation(len(x))
    
    if fix_val_sample:
        if logger:
            logger.log('fix permute @ seed =', seed_no, len(p), p[0:10])
        
    return x[p], y[p]

def shuffle_data_with_subj_id(x, y, y_id, logger, fix_val_sample=False, seed_no=0):
    # shape(x) = (nsamples, 3000, 1)
    if fix_val_sample:
        np.random.seed(seed_no)
    p = np.random.permutation(len(x))
    
    if fix_val_sample:
        logger.log('fix permute:', len(p), p[-10:])
        
    return x[p], y[p], y_id[p]

def arrange_minibatches(x, y, batch_size, batch_id, class_balance=True):
    # return data in shape (nsamples, 3000, 1) - use for one minibatch (batch batch_id)
    resx, resy = [], []
    
    labels = np.unique(y)
    samples_per_class = math.ceil(batch_size/len(labels))
    
    if class_balance:
        # each batch contains class-balance samples
        for c in labels:
            x_this_class = x[y==c]
            y_this_class = y[y==c]
            resx.extend(x_this_class[batch_id*samples_per_class: (batch_id+1)*samples_per_class])
            resy.extend(y_this_class[batch_id*samples_per_class: (batch_id+1)*samples_per_class])

    else:
        # just randomly pick to each batch
        resx.append(x[batch_id*samples_per_class: (batch_id+1)*samples_per_class])
        resy.append(y[batch_id*samples_per_class: (batch_id+1)*samples_per_class])
        
    resx, resy = np.array(resx[0:batch_size]), np.array(resy[0:batch_size])
    assert len(resx) <= batch_size
    assert len(resx) == len(resy)
    return resx, resy

def arrange_all_minibatches(x, y, batch_size, logger = None):
    # return data in shape (nbatches, nsamples, 3000, 1) - use for one epoch (nbatches batches) 
    # data should be "already" oversample if needed*
    nbatches = math.floor(len(x) / batch_size)
    
    labels = np.unique(y)
    samples_per_class = math.ceil(batch_size/len(labels))
    logger.log('nbatches:', nbatches, 'samples_per_class:', samples_per_class)
    
    # shuffle data in each class
    x_class = []
    y_class = []
    for c in labels:
        print('shuffling.. class:', c)
        x_this_class = x[y==c]
        y_this_class = y[y==c]
        x_this_class, y_this_class = shuffle_data(x_this_class, y_this_class, logger)
        x_class.append(x_this_class)
        y_class.append(y_this_class)
        del x_this_class, y_this_class
    
    x_class, y_class = np.array(x_class), np.array(y_class)
    results_x, results_y = [], []
    for batch_id in range(nbatches):
        #print('preparing.. batch:', batch_id)
        
        # interleave samples from each class into one mini-batch
        sample_x, sample_y = [], []
        for c in labels:
            x_this_class = x_class[c]
            y_this_class = y_class[c]
            
            sample_x.extend(x_this_class[batch_id*samples_per_class: (batch_id+1)*samples_per_class])
            sample_y.extend(y_this_class[batch_id*samples_per_class: (batch_id+1)*samples_per_class])
            
            del x_this_class, y_this_class

        results_x.append(sample_x[0:batch_size])
        results_y.append(sample_y[0:batch_size])
        
    results_x, results_y = np.array(results_x), np.array(results_y)
    logger.log(results_x.shape, results_y.shape)
    logger.log(len(results_x[0]), len(results_x[1]), batch_size)
    assert len(results_x) == nbatches == len(results_y)
    assert len(results_x[0]) == len(results_x[1]) == batch_size
    assert len(results_x[0]) == len(results_y[0]) == batch_size
    
    return results_x, results_y


def arrange_all_minibatches_with_subj_id(x, y, y_id, batch_size, logger = None):
    # return data in shape (nbatches, nsamples, 3000, 1) - use for one epoch (nbatches batches) 
    # data should be "already" oversample if needed*
    nbatches = math.floor(len(x) / batch_size)
    
    subjects = list(set(y_id))
    nsubj_per_batch = int(math.ceil(nbatches / len(subjects)))
    np.random.shuffle(subjects)
    
    labels = np.unique(y)
    samples_per_class = math.ceil(batch_size/len(labels))
    logger.log('nbatches:', nbatches, 'samples_per_class:', samples_per_class)
    
    results_x, results_y, results_y_id = [], [], []
    
    # shuffle data in each class
    x_class = []
    y_class = []
    y_id_class = []
    for c in labels:
        print('shuffling.. class:', c)
        x_this_class = x[y==c]
        y_this_class = y[y==c]
        y_id_this_class = y_id[y==c]
        x_this_class, y_this_class, y_id_this_class = shuffle_data_with_subj_id(x_this_class, 
                                                               y_this_class, 
                                                               y_id_this_class, 
                                                               logger)
        x_class.append(x_this_class)
        y_class.append(y_this_class)
        y_id_class.append(y_id_this_class)

        del x_this_class, y_this_class, y_id_this_class

    x_class, y_class, y_id_class = np.array(x_class), np.array(y_class), np.array(y_id_class)
    for batch_id in range(nbatches):
        #print('preparing.. batch:', batch_id)

        # interleave samples from each class into one mini-batch
        sample_x, sample_y, sample_y_id = [], [], []
        for c in labels:
            x_this_class = x_class[c]
            y_this_class = y_class[c]
            y_id_this_class = y_id_class[c]

            sample_x.extend(x_this_class[batch_id*samples_per_class: (batch_id+1)*samples_per_class])
            sample_y.extend(y_this_class[batch_id*samples_per_class: (batch_id+1)*samples_per_class])
            sample_y_id.extend(y_id_this_class[batch_id*samples_per_class: (batch_id+1)*samples_per_class])

        results_x.append(sample_x[0:batch_size])
        results_y.append(sample_y[0:batch_size])
        results_y_id.append(sample_y_id[0:batch_size])

    results_x, results_y, results_y_id = np.array(results_x), np.array(results_y), np.array(results_y_id)
    assert len(results_x) == nbatches == len(results_y) == len(results_y_id)
    assert len(results_x[0]) == len(results_x[1]) == batch_size == len(results_y_id[1])
    assert len(results_x[0]) == len(results_y[0]) == batch_size == len(results_y_id[0])
    
    return results_x, results_y, results_y_id


def pick_samples(x, y, batch_size, class_balanced = True, 
                 logger = None, fix_val_sample=False, request_b=True):
    """
    Randomly pick 'batch_size' samples of set A and B (if request_b = True, otherwise B = [])
        receive data in shape = (nsamples, 3000, 1)
        return data in shape = (batch_size, 3000, 1) of A & (batch_size, 3000, 1) of B
    """
    x = np.array(x)
    y = np.array(y)
    x, y = shuffle_data(x, y, logger, fix_val_sample)

    if not class_balanced:
        return x[0:batch_size], y[0:batch_size], x[batch_size:batch_size*2], y[batch_size:batch_size*2]
    else:
        labels = np.unique(y)
        samples_per_class = math.ceil(batch_size/len(labels))
        
        x_new_a, y_new_a, x_new_b, y_new_b = [], [], [], []
        for l in sorted(labels, reverse=True):
            y_this_class = y[y==l]
            x_this_class = x[y==l]
            x_new_a.extend(x_this_class[0:samples_per_class])
            y_new_a.extend(y_this_class[0:samples_per_class])
            
            if request_b:
                x_new_b.extend(x_this_class[samples_per_class:samples_per_class*2])
                y_new_b.extend(y_this_class[samples_per_class:samples_per_class*2])

                if len(x_this_class[0:samples_per_class]) < samples_per_class:
                    logger.log('ERROR: Too large batch_size, samples are not enough.')
                    raise Exception

                diff_a_b = len(x_this_class[0:samples_per_class]) - len(x_this_class[samples_per_class:samples_per_class*2])
                if diff_a_b != 0:
                    # samples per class are too less
                    # duplicate them
                    x_new_b.extend(x_this_class[0:diff_a_b])
                    y_new_b.extend(y_this_class[0:diff_a_b])
                    logger.log('duplicating', diff_a_b,'samples of class:', l, 'to have same number of a and b.')
             
        if request_b:
            assert len(x_new_a) == len(x_new_b)

        return np.array(x_new_a[0:batch_size]), np.array(y_new_a[0:batch_size]),\
                np.array(x_new_b[0:batch_size]), np.array(y_new_b[0:batch_size])


def pick_samples_with_subj_id(x, y, y_ids, batch_size, class_balanced = True, 
                              logger = None, fix_val_sample=False):
    """
    Randomly pick 'batch_size' samples with subj_id labels
        receive data in shape = (nsamples, 3000, 1)
        return data in shape = (batch_size, 3000, 1)
    """

    x = np.array(x)
    y = np.array(y)
            
    if not class_balanced:
        raise Exception('class imbalance not supported')
    else:
        labels = np.unique(y)
        samples_per_class = math.ceil(batch_size/len(labels))
        
        subjects = np.unique(y_ids)
        samples_per_class_per_subj = int(samples_per_class/len(subjects))
        logger.log('pick samples from subjects:', subjects)
        logger.log('samples_per_class_per_subj:', samples_per_class_per_subj)
        
        x_new, y_new, y_id_new = [], [], []
        for subj_id in subjects:
            x_this_subj = x[y_ids==subj_id]
            y_this_subj = y[y_ids==subj_id]
            x_this_subj, y_this_subj = shuffle_data(x_this_subj, y_this_subj, logger, 
                                                    fix_val_sample, seed_no=subj_id)
            
            for l in sorted(labels, reverse=True):
                y_this_class = y_this_subj[y_this_subj==l]
                x_this_class = x_this_subj[y_this_subj==l]
                x_new.extend(x_this_class[0:samples_per_class_per_subj])
                y_new.extend(y_this_class[0:samples_per_class_per_subj])
                y_id_new.extend([subj_id]*len(x_this_class[0:samples_per_class_per_subj]))
                
        x_new, y_new, y_id_new = np.array(x_new[:batch_size]), np.array(y_new[:batch_size]), np.array(y_id_new[:batch_size])
        assert x_new.shape[0] == y_new.shape[0] == y_id_new.shape[0]
        assert y_new.shape == y_id_new.shape

        return x_new, y_new, y_id_new


def reshape_data_into_seq_minibatches(x_raw, y_raw, batch_size, seq_length):
    """
    Reshape data into sequences for LSTM
        receive data in shape = (n_nights, nsamples, 3000)
        return data in shape = (n_nights, nbatches*batch_size, seq_length, 3000, 1)
    """
    batch_size = batch_size
    seq_length = seq_length

    x_list = []
    y_list = []
    for idx in range(len(x_raw)):
        temp_x = []
        temp_y = []
        for x, y in iterate_batch_seq_minibatches(x_raw[idx], y_raw[idx], batch_size=batch_size, seq_length=seq_length):
            x = x.reshape(batch_size, seq_length, 3000, 1)
            y = y.reshape(batch_size, seq_length, 1)

            temp_x.append(x)
            temp_y.append(y)

        temp_x = np.concatenate(temp_x, axis=0)
        temp_y = np.concatenate(temp_y, axis=0)

        x_list.append(temp_x)
        y_list.append(temp_y)

    return np.array(x_list), np.array(y_list)


def get_balance_class_downsample(x, y):
    """
    Balance the number of samples of all classes by (downsampling):
        1. Find the class that has a smallest number of samples
        2. Randomly select samples in each class equal to that smallest number
    """

    class_labels = np.unique(y)
    n_min_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_min_classes == -1:
            n_min_classes = n_samples
        elif n_min_classes > n_samples:
            n_min_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        idx = np.random.permutation(idx)[:n_min_classes]
        balance_x.append(x[idx])
        balance_y.append(y[idx])
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)

    return balance_x, balance_y


def get_balance_class_oversample(x, y, logger):
    """
    Balance the number of samples of all classes by (oversampling):
        1. Find the class that has the largest number of samples
        2. Randomly select samples in each class equal to that largest number
    """
    logger.log('-- oversample')
    class_labels = np.unique(y)
    n_max_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_max_classes < n_samples:
            n_max_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        n_samples = len(idx)
        n_repeats = int(n_max_classes / n_samples)
        tmp_x = np.repeat(x[idx], n_repeats, axis=0)
        tmp_y = np.repeat(y[idx], n_repeats, axis=0)
        n_remains = n_max_classes - len(tmp_x)
        if n_remains > 0:
            sub_idx = np.random.permutation(idx)[:n_remains]
            tmp_x = np.vstack([tmp_x, x[sub_idx]])
            tmp_y = np.hstack([tmp_y, y[sub_idx]])
        balance_x.append(tmp_x)
        balance_y.append(tmp_y)
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)

    return balance_x, balance_y


def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    """
    Generate a generator that return a batch of inputs and targets.
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


def iterate_seq_minibatches(inputs, targets, batch_size, seq_length, stride):
    """
    Generate a generator that return a batch of sequence inputs and targets.
    """
    assert len(inputs) == len(targets)
    n_loads = (batch_size * stride) + (seq_length - stride)
    for start_idx in range(0, len(inputs) - n_loads + 1, (batch_size * stride)):
        seq_inputs = np.zeros((batch_size, seq_length) + inputs.shape[1:],
                              dtype=inputs.dtype)
        seq_targets = np.zeros((batch_size, seq_length) + targets.shape[1:],
                               dtype=targets.dtype)
        for b_idx in xrange(batch_size):
            start_seq_idx = start_idx + (b_idx * stride)
            end_seq_idx = start_seq_idx + seq_length
            seq_inputs[b_idx] = inputs[start_seq_idx:end_seq_idx]
            seq_targets[b_idx] = targets[start_seq_idx:end_seq_idx]
        flatten_inputs = seq_inputs.reshape((-1,) + inputs.shape[1:])
        flatten_targets = seq_targets.reshape((-1,) + targets.shape[1:])
        yield flatten_inputs, flatten_targets


def iterate_batch_seq_minibatches(inputs, targets, batch_size, seq_length):
    assert len(inputs) == len(targets)
    n_inputs = len(inputs)
    batch_len = n_inputs // batch_size

    epoch_size = batch_len // seq_length
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or seq_length")

    seq_inputs = np.zeros((batch_size, batch_len) + inputs.shape[1:],
                          dtype=inputs.dtype)
    seq_targets = np.zeros((batch_size, batch_len) + targets.shape[1:],
                           dtype=targets.dtype)

    for i in range(batch_size):
        seq_inputs[i] = inputs[i*batch_len:(i+1)*batch_len]
        seq_targets[i] = targets[i*batch_len:(i+1)*batch_len]

    for i in range(epoch_size):
        x = seq_inputs[:, i*seq_length:(i+1)*seq_length]
        y = seq_targets[:, i*seq_length:(i+1)*seq_length]
        flatten_x = x.reshape((-1,) + inputs.shape[1:])
        flatten_y = y.reshape((-1,) + targets.shape[1:])
        yield flatten_x, flatten_y


def iterate_list_batch_seq_minibatches(inputs, targets, batch_size, seq_length):
    for idx, each_data in enumerate(itertools.izip(inputs, targets)):
        each_x, each_y = each_data
        seq_x, seq_y = [], []
        for x_batch, y_batch in iterate_seq_minibatches(inputs=each_x, 
                                                        targets=each_y, 
                                                        batch_size=1, 
                                                        seq_length=seq_length, 
                                                        stride=1):
            seq_x.append(x_batch)
            seq_y.append(y_batch)
        seq_x = np.vstack(seq_x)
        seq_x = seq_x.reshape((-1, seq_length) + seq_x.shape[1:])
        seq_y = np.hstack(seq_y)
        seq_y = seq_y.reshape((-1, seq_length) + seq_y.shape[1:])
        
        for x_batch, y_batch in iterate_batch_seq_minibatches(inputs=seq_x, 
                                                              targets=seq_y, 
                                                              batch_size=batch_size, 
                                                              seq_length=1):
            x_batch = x_batch.reshape((-1,) + x_batch.shape[2:])
            y_batch = y_batch.reshape((-1,) + y_batch.shape[2:])
            yield x_batch, y_batch

def convert_to_nepochs(inputs, targets, nepochs_per_sample=5):
    """
    inputs: raw signal from 1 night (nsamples, 3000, 1)
    targets: sleep_stage (nsamples)
    
    return sample in shape (n_newsamples, 3000*nepochs_per_sample, 1) with sleep_stage of the middle epoch
    """
    new_x = []
    new_y = []
    for index in range(len(inputs)):
        start_index = index-int(nepochs_per_sample/2)
        end_index = index+int(nepochs_per_sample/2)+1
        
        if len(inputs[start_index:end_index]) == nepochs_per_sample:
            new_x.append(np.concatenate(inputs[start_index:end_index]))
            new_y.append(targets[index])
            
    new_x, new_y = np.array(new_x), np.array(new_y)
    assert len(new_x) == len(new_y)
    assert new_x.shape[1] == nepochs_per_sample*inputs.shape[1]
    assert new_x.shape[2] == inputs.shape[2]
    
    return new_x, new_y

    
