import pickle as cPickle
import gzip
import numpy as np
import os
from scipy.misc import imread
import random

def mnist(datasets_dir='./utils/datasets', withseparatedim=False):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = cPickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], -1)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], -1)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], -1)
    train_y = train_y.astype('int32')
    print('... done loading data')
    
    train_idxs = np.random.permutation(train_x.shape[0])
    train_x = train_x[train_idxs]
    train_y = train_y[train_idxs]
    print('... done permutation training set')
    
    if withseparatedim:
        mask_reshape = [-1, 1, 28, 28]
        return (train_x.reshape(mask_reshape), train_y,
               valid_x.reshape(mask_reshape), valid_y,
               test_x.reshape(mask_reshape), test_y)
    else:
        return train_x, train_y, valid_x, valid_y, test_x, test_y
