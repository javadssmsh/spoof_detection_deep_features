import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1";  
import keras
import soundfile as sf
from conf import *
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras import backend as K
from model import getModel
import librosa as lb
from data_io import batchGenerator_scores
import math
from sklearn.metrics import confusion_matrix


def log_softmax(x):
    return x - tf.log(tf.reduce_sum(tf.exp(x), -1, keep_dims=True))


input_shape = wlen,1
model = getModel(input_shape, out_dim)
model.load_weights(pt_file)

validation_generator = batchGenerator_scores(Batch_dev, dev_data_folder, flac_lst_dev, snt_dev, wlen,out_dim)
y = model.predict_generator(validation_generator, steps = (snt_dev/Batch_dev), verbose = 1)
print(len(y),len(flac_lst_dev))
logits = y[:24844]
log_probs = log_softmax(logits)
flac_lst_dev['scores'] = y[:,0]-y[:,1]
y_pred = tf.argmax(log_probs,1)

print(confusion_matrix(list(flac_lst_dev['label']), y_pred))

file1 = open(output_folder+'/dev_scores.txt',"w")
for i in range(flac_lst_dev.shape[0]):
    if flac_lst_dev.iloc[i,2] == 0:
        L = "{} - bonafide {} \n".format(flac_lst_dev.iloc[i,1] , flac_lst_dev.iloc[i,-1])
    else:
        L = "{} A0{} spoof {} \n".format(flac_lst_dev.iloc[i,1], flac_lst_dev.iloc[i,2], flac_lst_dev.iloc[i,-1])

    file1.writelines(L) 
file1.close()