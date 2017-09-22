""" Reducing clutter etc. etc. etc. """

import gzip
import pickle
import numpy as np
import sys
import tensorflow as tf
from collections import defaultdict


def get_tf_session():
    """ Returning a session. Set options here if desired. """
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1,
                               intra_op_parallelism_threads=1)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def get_available_gpus():
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

    print("AVAILABLE GPUS: ", get_available_gpus())
    return session


def load_dataset(name):
    """ Given dataset, return train, valid, test sets. 
    
    For MNIST, the pickled data is from Python 2, gah. Fortunately this blog
    helped: http://www.mlblog.net/2016/09/reading-mnist-in-python3.html. The
    train, val, and test are both tuples with the first and second elements as
    the data and labels. Arrange things in a dictionary. Also for now we're
    combining the training and validation into the `train` set as there's no
    need for validation here. GANs are actually resistant to overfitting as they
    can never actually see the real images.
    """
    if name == 'mnist':
        with gzip.open('../data/mnist.pkl.gz','rb') as ff:
            u = pickle._Unpickler( ff )
            u.encoding = 'latin1'
            train, val, test = u.load()
        data = {}
        data['X_train'] = np.concatenate( (train[0],val[0]), axis=0 )
        data['y_train'] = np.concatenate( (train[1],val[1]), axis=0 )
        data['X_test'] = test[0]
        data['y_test'] = test[1]
        print("X_train.shape: {} (+valid)".format(data['X_train'].shape))
        print("Y_train.shape: {} (+valid)".format(data['y_train'].shape))
        print("X_test.shape:  {}".format(data['X_test'].shape))
        print("y_test.shape:  {}".format(data['y_test'].shape))
        assert (0.0 <= np.min(data['X_train']) and np.max(data['X_train']) <= 1.0)
        return data
    else:
        raise ValueError("Dataset name {} is not valid".format(name))


def list_of_minibatches(data, bsize, shuffle=True):
    """ Forms a list of minibatches for each element in `data` to avoid
    repeatedly shuffling and sampling during training.

    Assumes `data` is a dictionary with `X_train` and `y_train` keys, and
    returns a dictionary of the same length.
    """
    data_lists = defaultdict(list)
    assert 'X_train' in data.keys() and 'y_train' in data.keys()
    N = data['X_train'].shape[0]
    indices = np.random.permutation(N)
    X_train = data['X_train'][indices]
    y_train = data['y_train'][indices]

    for i in range(0, N-bsize, bsize):
        data_lists['X_train'].append(X_train[i:i+bsize, :])
        data_lists['y_train'].append(y_train[i:i+bsize])

    first = data_lists['X_train'][0].shape
    last  = data_lists['X_train'][-1].shape
    assert first == last, "{} vs {} w/bs {}".format(first, last, bsize)
    return data_lists
