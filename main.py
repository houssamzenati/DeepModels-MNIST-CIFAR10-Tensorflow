#! python3
# Originally written by Gaurav Manek

import argparse
import importlib
import logging
import os
import shutil
import urllib3
import zipfile

import data

# Logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s @%(name)s] %(message)s', datefmt='%H:%M:%S'))
logging.basicConfig(level=logging.DEBUG, handlers=[console])
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logger = logging.getLogger("AnomalyDetection")



def run(args):
    print("""
 ______   _____       _____       ____   
|_     `.|_   _|     / ___ `.   .'    '. 
  | | `. \ | |      |_/___) |  |  .--.  |
  | |  | | | |   _   .'____.'  | |    | |
 _| |_.' /_| |__/ | / /_____  _|  `--'  |
|______.'|________| |_______|(_)'.____.' 
                                         
""")

    has_effect = False

    if args.datasets:
        download_data()
        has_effect = True

    if args.pretrained:
        if args.example:
            download_pretrained(args.example)
        else:
            for example in ["gan"]: # To be changed: we'll have pretrained weights for AE and VAE
                download_pretrained(example)

        has_effect = True

    if args.example and args.dataset and args.split:
        try:
            mod_name = "{}.{}".format(args.example, args.split)
            logger.info("Running script at {}".format(mod_name))

            mod = importlib.import_module(mod_name)
            mod.run(args.dataset, int(args.ignore))

        except Exception as e:
            logger.exception(e)
            logger.error("Uhoh, the script halted with an error.")
    else:
        if not has_effect:
            logger.error("Script halted without any effect. To run code, use command:\npython3 main.py <example name> {train, test}")

def download_data():
    from data import mnist, cifar10  # To be changed: we'll have MNIST and other datasets

    for mod in [mnist, cifar10]: # To be changed: we'll have MNIST and other datasets
        logger.info("Downloading dataset: {}".format(mod.__name__))
        mod.get_train()
        mod.get_test()

def download_pretrained(example):
    logger.info("Downloading pretrained weights for: {}".format(example))

    data_url = "http://10.217.128.198/pretrained/{}.zip".format(example)
    # Check if .../example/ exists
    if not os.path.isdir(example):
        logger.error("Example {} does not exist.".format(example))
        return

    logs_path = os.path.join(example, "train_logs")
    file_path = os.path.join(example, "train_logs.zip")

    # Try to delete the directory
    try:
        shutil.rmtree(logs_path)
    except:
        pass

    # Recreate the directory
    try:
        os.mkdir(logs_path)
    except Exception as e:
        logger.error("Error creating directory {}.".format(logs_path))
        logger.exception(e)
        return

    # Download
    logger.warn("Downloading {}".format(data_url))
    with urllib3.PoolManager().request('GET', data_url, preload_content=False) as r, open(file_path, 'wb') as w:
        shutil.copyfileobj(r, w)
    logger.warn("Unpacking {}".format(file_path))
    # Unpack data
    with zipfile.ZipFile(file_path, 'r') as zipf:
        zipf.extractall(logs_path)
    os.remove(file_path)

    pass

def path(d):
    try:
        assert os.path.isdir(d)
        return d
    except Exception as e:
        raise argparse.ArgumentTypeError("Example {} cannot be located.".format(d))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run examples from the DL 2.0 Anomaly Detector.')
    parser.add_argument('--datasets', action='store_true', help='download all datasets')
    parser.add_argument('--pretrained', action='store_true', help='download pretrained weights for an example if specified, or all examples otherwise')
    parser.add_argument('example', nargs="?", type=path, help='the folder name of the example you want to run')
    parser.add_argument('dataset', nargs="?", choices=['mnist', 'cifar10'], help='the name of the dataset you want to run the experiments on')
    parser.add_argument('split', nargs="?", choices=['train', 'test'], help='train the example or evaluate it')
    parser.add_argument('ignore', nargs="?", choices=list(str(i) for i in range(0,10)), help='the label you want to ignore for mnist or cifar10')
    #parser.add_argument('nb_epochs', nargs="?", type=int, help='the label you want to ignore for mnist or cifar10')

    run(parser.parse_args())
