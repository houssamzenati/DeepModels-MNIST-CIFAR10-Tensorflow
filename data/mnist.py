# MNIST Downloader

import logging
import pickle
import os
import os.path
import gzip
import urllib.request
import numpy as np


logger = logging.getLogger(__name__)

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"


def get_train(digit_to_ignore=None):
    return _get_dataset("train", digit_to_ignore)

def get_test(centered=False):
    return _get_dataset("test", centered=centered)

def get_shape_input():
    return (None, 28, 28, 1)

def get_shape_label():
    return (None,)

def num_classes():
    return 10

def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):
        return

    logger.warn("Downloading {} ... ".format(file_name))
    urllib.request.urlretrieve(url_base + file_name, file_path)
    logger.debug("Done")
    
def download_mnist():
    for v in key_file.values():
       _download(v)
        
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    logger.debug("Converting {} to NumPy Array ...".format(file_name))
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    logger.debug("Done")
    
    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    logger.debug("Converting {} to NumPy Array ...".format(file_name))
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 784)
    logger.debug("Done")
    
    return data
    
def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    logger.debug("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    logger.debug("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    

def _get_dataset(split, digit_to_ignore=None, normalize=True, flatten=False, channels_first=False, squeeze=False, one_hot_label=False, centered=False):

    assert split == 'test' or split == 'train'
    if not os.path.exists(save_file):
        init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    key_img = split + '_img'
    key_lbl = split + '_label'

    if normalize:
        dataset[key_img] = dataset[key_img].astype(np.float32)
        dataset[key_img] /= 255.0

    if centered:
        dataset[key_img] = dataset[key_img].astype(np.float32)
        dataset[key_img] = dataset[key_img]*2. - 1.

    if one_hot_label:
        dataset[key_lbl] = _change_one_hot_label(dataset[key_lbl])
    
    if not flatten:
        dataset[key_img] = dataset[key_img].reshape(-1, 1, 28, 28)

    if not channels_first:
        if flatten:
            logger.error('Set flatten=False to use channels_first')
            return
        dataset[key_img] = dataset[key_img].transpose([0, 2, 3, 1])

    if squeeze:
        dataset[key_img] = np.squeeze(dataset[key_img])

    if digit_to_ignore != None:
        dataset[key_img] = dataset[key_img][(dataset[key_lbl] != digit_to_ignore)]
        dataset[key_lbl] = dataset[key_lbl][(dataset[key_lbl] != digit_to_ignore)]

    return (dataset[key_img], dataset[key_lbl])