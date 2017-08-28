import shutil
import os

output_path = 'output'
log_path = 'log'

import train 

if __name__ == '__main__':
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.mkdir(log_path)
    print("start training")
    train.train_model()