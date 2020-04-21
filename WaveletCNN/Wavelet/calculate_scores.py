import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1";  
import keras
import soundfile as sf
from conf import *
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
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
input_shape = wlen,1
model = getModel(input_shape, out_dim)
model.load_weights(pt_file)

validation_generator = batchGenerator_scores(Batch_dev, dev_data_folder, flac_lst_dev, snt_dev, wlen,out_dim)
y = model.predict_generator(validation_generator, steps = (snt_dev/Batch_dev), verbose = 1)
print(len(y),len(flac_lst_dev))
y1 = [math.log(i) for i in y[:24844,0]] 
y2 = [math.log(s) for s in y[:24844,1]]
flac_lst_dev['scores'] = [y1[i]-y2[i] for i in range(len(y1))]
y_pred = [1 if i>0 else 0 for i in flac_lst_dev['scores']]
# print(flac_lst_dev['label'][0:5], y_pred[:5])
print(confusion_matrix(list(flac_lst_dev['label']), y_pred))

file1 = open(output_folder+'/dev_scores.txt',"w")
# df = pd.DataFrame(columns=['file_id','spoof_id','label','scores'])
# df['file_id'] = flac_lst_dev['file_id']
# df['spoof_id'] = flac_lst_dev['system_id']
# df['label'] = ['bonafide' if s == 0 else 'spoof' for s in flac_lst_dev['label']]
# print(df.get_value(1,2))
# df['scores'] = flac_lst_dev['scores']
for i in range(flac_lst_dev.shape[0]):
    if flac_lst_dev.iloc[i,2] == 0:
        L = "{} - bonafide {} \n".format(flac_lst_dev.iloc[i,1] , flac_lst_dev.iloc[i,-1])
    else:
        L = "{} A0{} spoof {} \n".format(flac_lst_dev.iloc[i,1], flac_lst_dev.iloc[i,2], flac_lst_dev.iloc[i,-1])

    file1.writelines(L) 
file1.close()