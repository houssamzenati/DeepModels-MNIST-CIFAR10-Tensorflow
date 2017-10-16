# CIFAR10 Downloader

import logging
import pickle
import math
import os
import errno
import tarfile
import shutil

import numpy as np

import urllib3

logger = logging.getLogger(__name__)


def get_train(class_to_ignore=None):
    return _get_dataset("train", class_to_ignore)

def get_test(centered=True):
    return _get_dataset("test", centered=centered)

def get_shape_input():
    return (None, 32, 32, 3)

def get_shape_label():
    return (None,)

def num_classes():
    return 10

def _unpickle_file(filename):
    logger.debug("Loading pickle file: {}".format(filename))

    with open(filename, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    # Reorder the data
    img = data[b'data']
    img = img.reshape([-1, 3, 32, 32])
    img = img.transpose([0, 2, 3, 1])
    # Load labels
    lbl = np.array(data[b'labels'])

    return img, lbl


def _get_dataset(split, class_to_ignore=None, centered=False):
    assert split == "test" or split == "train"
    path = "data"
    dirname = "cifar-10-batches-py"
    data_url = "http://10.217.128.198/datasets/cifar-10-python.tar.gz"

    if not os.path.exists(os.path.join(path, dirname)):
        # Extract or download data
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        
        file_path = os.path.join(path, data_url.split('/')[-1])
        if not os.path.exists(file_path):
            # Download
            logger.warn("Downloading {}".format(data_url))
            with urllib3.PoolManager().request('GET', data_url, preload_content=False) as r, \
                 open(file_path, 'wb') as w:
                    shutil.copyfileobj(r, w)

        logger.warn("Unpacking {}".format(file_path))
        # Unpack data
        tarfile.open(name=file_path, mode="r:gz").extractall(path)

    # Import the data
    filenames = ["test_batch"] if split == "test" else \
                ["data_batch_{}".format(i) for i in range(1, 6)]
    
    imgs = []
    lbls = []
    for f in filenames:
        img, lbl = _unpickle_file(os.path.join(path, dirname, f))
        
        if class_to_ignore is not None:
            img = img[lbl!=class_to_ignore]
            lbl = lbl[lbl!=class_to_ignore]
            
        imgs.append(img)
        lbls.append(lbl)
        
    # Now we flatten the arrays
    imgs = np.concatenate(imgs)
    lbls = np.concatenate(lbls)

    # Convert images to [0..1] range
    imgs = imgs.astype(np.float32)/255.0

    if centered:
        imgs = imgs.astype(np.float32)*2. - 1.

    return imgs, lbls


