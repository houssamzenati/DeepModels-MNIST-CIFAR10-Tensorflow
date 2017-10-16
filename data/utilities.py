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

BASE_URL = "http://10.217.128.198/datasets/"
def ensure_dataset_exists(files, dirname):
    path = os.path.join("data", dirname)
    rv = [os.path.join(path, f) for f in files]
    
    logger.info("Retrieving dataset from {}".format(path))
    if not os.path.exists(path):
        # Extract or download data
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        
        for f, file_path in zip(files, rv):
            data_url = BASE_URL + dirname + "/" + f
            if not os.path.exists(file_path):
                logger.warn("Downloading {}".format(data_url))
                with urllib3.PoolManager().request('GET', data_url, preload_content=False) as r, \
                    open(file_path, 'wb') as w:
                        shutil.copyfileobj(r, w)
    return rv


# Convert data into a stream of never-ending data
def finite_generator(data, batch_size):
    x, y = data
    i = 0 # Type: int
    while True:
        j = i + batch_size
        # If we wrap around the back of the dataset:
        if j >= x.shape[0]:
            yield (x[i:x.shape[0],...], y[i:x.shape[0]])
            break
        else:
            yield (x[i:j,...], y[i:j])
            i = j

# Convert data into a stream of never-ending data
def infinite_generator(data, batch_size):
    x, y = data
    i = 0 # Type: int
    while True:
        j = i + batch_size
        # If we wrap around the back of the dataset:
        if j >= x.shape[0]:
            rv = list(range(i, x.shape[0])) + list(range(0, j - x.shape[0]))
            yield (x[rv,...], y[rv])
            i = j - x.shape[0]
        else:
            yield (x[i:j,...], y[i:j])
            i = j
