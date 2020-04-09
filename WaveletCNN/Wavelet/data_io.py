"""
            File that is responsible for initializing the variables in conf.py file from the config file.

            Functions:
            ReadList(list_file) : To convert the protocol file into a dataframe to feed it into other functions like batchGenerator
            Read_conf(conf_file): To read the config file and initialize the conf.py file
            Str_to_bool():


"""

import configparser as ConfigParser
from optparse import OptionParser
import numpy as np
import pandas as pd

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1";  



def ReadList(list_file):

    filehandle = open(list_file, 'r')

    protocol = []

    while True:

        # read a single line

        line = (filehandle.readline())

        protocol.append(line)

        if not line:

            break

    # close the pointer to that file
    filehandle.close()

    protocol = [s[:-1] for s in protocol]

    protocol = pd.DataFrame([s.split(' ') for s in protocol])

    protocol.columns = ['speaker_id', 'file_id', 'blah', 'system_id', 'label']

    protocol = protocol[['speaker_id', 'file_id', 'system_id', 'label']]

    protocol.dropna(inplace=True)

    protocol.drop_duplicates(subset="file_id", keep='first', inplace=True)

    for count, i in enumerate(protocol.iloc[:, -2]):
        if i == '-':
            protocol.iloc[count, -2] = 0
            protocol.iloc[count, -1] = 0
        elif i == 'A01':
            protocol.iloc[count, -2] = 1
            protocol.iloc[count, -1] = 1
        elif i == 'A02':
            protocol.iloc[count, -2] = 2
            protocol.iloc[count, -1] = 1
        elif i == 'A03':
            protocol.iloc[count, -2] = 3
            protocol.iloc[count, -1] = 1
        elif i == 'A04':
            protocol.iloc[count, -2] = 4
            protocol.iloc[count, -1] = 1
        elif i == 'A05':
            protocol.iloc[count, -2] = 5
            protocol.iloc[count, -1] = 1
        elif i == 'A06':
            protocol.iloc[count, -2] = 6
            protocol.iloc[count, -1] = 1

    return protocol


def read_conf(cfg_file=None):
    parser = OptionParser()
    parser.add_option("--cfg")  # Mandatory
    (options, args) = parser.parse_args()
    if cfg_file is None:
        cfg_file = options.cfg
    if options.cfg is None and cfg_file is None:
        cfg_file = r'/home/rohita/rohit/spoof/spoof_deep_features/spoof_detection_deep_features/WaveletCNN/Wavelet/cfg/Wavelet_ASV.cfg'
    Config = ConfigParser.ConfigParser()
    Config.read(cfg_file)

    # [data]
    options.train_lst = Config.get('data', 'train_lst')
    options.dev_lst = Config.get('data', 'dev_lst')
    # options.lab_dict = Config.get('data', 'lab_dict')
    options.train_data_folder = Config.get('data', 'train_data_folder')
    options.dev_data_folder = Config.get('data', 'dev_data_folder')
    options.output_folder = Config.get('data', 'output_folder')
    options.pt_file = Config.get('data', 'pt_file')

    # [windowing]
    options.fs = Config.get('windowing', 'fs')
    options.cw_len = Config.get('windowing', 'cw_len')
    options.cw_shift = Config.get('windowing', 'cw_shift')

    # [cnn]
    options.cnn_N_filt = Config.get('cnn', 'cnn_N_filt')
    options.cnn_len_filt = Config.get('cnn', 'cnn_len_filt')
    options.cnn_max_pool_len = Config.get('cnn', 'cnn_max_pool_len')
    options.cnn_use_laynorm_inp = Config.get('cnn', 'cnn_use_laynorm_inp')
    options.cnn_use_batchnorm_inp = Config.get('cnn', 'cnn_use_batchnorm_inp')
    options.cnn_use_laynorm = Config.get('cnn', 'cnn_use_laynorm')
    options.cnn_use_batchnorm = Config.get('cnn', 'cnn_use_batchnorm')
    options.cnn_act = Config.get('cnn', 'cnn_act')
    options.cnn_drop = Config.get('cnn', 'cnn_drop')

    # [dnn]
    options.fc_lay = Config.get('dnn', 'fc_lay')
    options.fc_drop = Config.get('dnn', 'fc_drop')
    options.fc_use_laynorm_inp = Config.get('dnn', 'fc_use_laynorm_inp')
    options.fc_use_batchnorm_inp = Config.get('dnn', 'fc_use_batchnorm_inp')
    options.fc_use_batchnorm = Config.get('dnn', 'fc_use_batchnorm')
    options.fc_use_laynorm = Config.get('dnn', 'fc_use_laynorm')
    options.fc_act = Config.get('dnn', 'fc_act')

    # [class]
    options.class_lay = Config.get('class', 'class_lay')
    options.class_drop = Config.get('class', 'class_drop')
    options.class_use_laynorm_inp = Config.get('class', 'class_use_laynorm_inp')
    options.class_use_batchnorm_inp = Config.get('class', 'class_use_batchnorm_inp')
    options.class_use_batchnorm = Config.get('class', 'class_use_batchnorm')
    options.class_use_laynorm = Config.get('class', 'class_use_laynorm')
    options.class_act = Config.get('class', 'class_act')

    # [optimization]
    options.lr = Config.get('optimization', 'lr')
    options.batch_size = Config.get('optimization', 'batch_size')
    options.N_epochs = Config.get('optimization', 'N_epochs')
    options.N_batches = Config.get('optimization', 'N_batches')
    options.N_dev_batches = Config.get('optimization', 'N_dev_batches')
    options.N_eval_epoch = Config.get('optimization', 'N_eval_epoch')
    options.seed = Config.get('optimization', 'seed')

    return options


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError