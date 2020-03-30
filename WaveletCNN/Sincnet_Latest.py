#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";  


# In[3]:


import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)


# In[4]:


import warnings
warnings.simplefilter(action='ignore', category=Warning)
import numpy as np
# import librosa as lb
from tensorflow import keras
import tensorflow as tf

from matplotlib import pyplot as plt
from keras.optimizers import Adam,SGD
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Lambda
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import AveragePooling1D
from keras.layers import BatchNormalization
from keras.layers.merge import add, concatenate
from keras.utils import plot_model
from keras.utils import to_categorical,Sequence
import pandas as pd
import math
import sincnet
# from tensorflow import set_random_seed
from keras import models, layers
import numpy as np
import sincnet
from keras.layers import Dense, Dropout, Activation
from keras.layers import MaxPooling1D,MaxPooling2D, Conv1D, LeakyReLU, BatchNormalization, Dense, Flatten
from keras.layers import InputLayer, Input
from keras.models import Model
# from tensorflow.python.keras.utils.data_utils import Sequence
from sklearn.preprocessing import StandardScaler
import pywt
from sklearn.decomposition import PCA
import sys
import tensorflow.contrib.slim as slim
# import tfwavelets
# sys.path.insert(1, '/content/drive/My Drive/SA/Code/spoof_detection_deep_features/WaveletCNN/cwt-tensorflow')
# from cwt import cwtMortlet, cwtRicker,mortletWavelet, rickerWavelet
# tf.compat.v1.enable_eager_execution()


# import pydot


# In[5]:


def generator(data, labels, batch_size):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[audio1,label1], [audio2,label2],...].
    """
    

    total_batches = int(data.shape[0])
    for i in range(total_batches):
        X_train,y_train = create_batches_rnd(data, labels, batch_size)
        yield X_train, y_train


# In[6]:


def generator_val(data, labels, batch_size):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[audio1,label1], [audio2,label2],...].
    """
    

    total_batches = int(data.shape[0])
    for i in range(total_batches):
        X_train,y_train = create_batches_rnd(data, labels, batch_size)
        yield X_train, y_train


# In[7]:


def create_batches_rnd(data,labels,batch_size):
    wlen = 3200
    fact_amp = 0.2
    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    sig_batch=np.zeros([batch_size,wlen])
#     wpt_batch = np.zeros([batch_size,wlen,1])
    lab_batch=[]
    signal_id_arr=np.random.randint(data.shape[0], size=batch_size)
    rand_amp_arr = np.random.uniform(1.0-fact_amp,1+fact_amp,batch_size)
    for i in range(batch_size): 
        # select a random sentence from the list 
        #[fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst[snt_id_arr[i]])
        #signal=signal.astype(float)/32768
#         [signal, fs] = sf.read(data_folder+wav_lst[snt_id_arr[i]])
        signal = data[signal_id_arr[i]]
        # accesing to a random chunk
        signal_len=signal.shape[0]
        signal_beg=np.random.randint(signal_len-wlen-1) #randint(0, snt_len-2*wlen-1)
        signal_end=signal_beg+wlen
        sig_batch[i,:]=signal[signal_beg:signal_end]*rand_amp_arr[i]
#         wpt = Wavelet_1d(signal[signal_beg:signal_end])
#         wpt_batch[i,:] = wpt
        y=labels[signal_id_arr[i]]
#         yt = to_categorical(y, num_classes=out_dim)
        lab_batch.append(y)
    a, b = np.shape(sig_batch)
    sig_batch = sig_batch.reshape((a, b, 1))
    return sig_batch, np.array(lab_batch)


# In[8]:


def res_conv_block(X,in_channels,out_channels,stage,block,dilation=1):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    X_shortcut = X
    
    X = BatchNormalization(name=bn_name_base+'a')(X)
    X = Activation('relu')(X)
    X = Conv1D(in_channels, 3, padding='valid',use_bias = False, name= conv_name_base+'a')(X)
    X = BatchNormalization(name=bn_name_base+'b')(X)
    X = Activation('relu')(X)
    X = Conv1D(in_channels, 3, padding='valid',use_bias = False, name= conv_name_base+'b')(X)
    print(X.shape)
    paddings = tf.constant([[0, 0],   # the batch size dimension
                          [2, 2],   # top and bottom of image
                          [0, 0]])  # the channels dimension
    X = Lambda(lambda x: tf.pad(x, paddings, mode='CONSTANT',
                        constant_values=0.0))(X)
    X = concatenate([X , X_shortcut])
    X = BatchNormalization(name = bn_name_base+'c')(X)
    X = Activation('relu')(X)
    X = Conv1D(out_channels, 3, padding='valid',use_bias = False, dilation_rate = dilation, name = conv_name_base+'c')(X)

    return X


# In[9]:


input_shape = None,3200,1
    

inputs = tf.placeholder(tf.float32, shape=input_shape, name= 'the_input')


sinc = sincnet.SincConv1D(64, 251, 16000)(inputs)
sinc_pool = MaxPooling1D(pool_size=3,name = 'sinc_pool')(sinc)
sinc_norm = BatchNormalization(momentum=0.05, name = 'sinc_norm')(sinc_pool)
sinc_layer_norm = sincnet.LayerNorm(name = 'sinc_layer_norm')(sinc_norm)
sinc_relu = LeakyReLU(alpha=0.2, name = 'sinc_relu')(sinc_layer_norm)

sinc_conv = Conv1D(64, 5, strides=2, padding='valid',kernel_initializer = keras.initializers.glorot_uniform(seed=0))(sinc_relu)
# sinc_pool_1 = MaxPooling1D(pool_size=3,name = 'sinc_pool_1')(sinc_conv)
sinc_norm_1 = BatchNormalization(momentum=0.05, name = 'sinc_norm_1')(sinc_conv)
sinc_layer_norm_1 = sincnet.LayerNorm(name = 'sinc_layer_norm_1')(sinc_norm_1)
sinc_relu_1 = LeakyReLU(alpha=0.2, name = 'sinc_relu_1')(sinc_layer_norm_1)
 
#concate level one and level two decomposition
# concate_level_2 = concatenate([relu_1_2,sinc_relu_1])
# print(concate_level_2.shape)
res_conv_1 = res_conv_block(sinc_relu_1, 128, 16, 1, 'a', 4)
res_conv_2 = res_conv_block(res_conv_1, 16, 8, 2, 'a', 8)
res_conv_3 = res_conv_block(res_conv_2, 8, 4, 3, 'a', 16)
res_conv_4 = res_conv_block(res_conv_3, 4, 2, 4, 'a', 32)
res_conv_5 = res_conv_block(res_conv_4, 2, 1, 5, 'a', 64)

res_norm = BatchNormalization(name='res_norm')(res_conv_5)
res_relu = Activation('relu')(res_norm)


pool_5_1 = AveragePooling1D(pool_size=3, padding='same', name='avg_pool_5_1')(res_relu)
flat_5_1 = Flatten(name='flat_5_1')(pool_5_1) 

fc_5 = Dense(2048, name='fc_5',kernel_initializer = keras.initializers.glorot_uniform(seed=0))(flat_5_1)
norm_5 = BatchNormalization(name='norm_5')(fc_5)
relu_5 = Activation('relu', name='relu_5')(norm_5)
drop_5 = Dropout(0.5, name='drop_5')(relu_5)

fc_6 = Dense(2048, name='fc_6',kernel_initializer = keras.initializers.glorot_uniform(seed=0))(drop_5)
norm_6 = BatchNormalization(name='norm_6')(fc_6)
relu_6 = Activation('relu', name='relu_6')(norm_6)
drop_6 = Dropout(0.5, name='drop_6')(relu_6)

output = Dense(2, activation=tf.nn.softmax)(drop_6)


# In[10]:


def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

model_summary()


# In[11]:


labels = tf.placeholder(tf.float32, shape=(None,2))

from keras.metrics import categorical_accuracy as accuracy


with tf.name_scope('Accuracy'):
    acc= tf.reduce_mean(accuracy(labels, output))

from keras.objectives import categorical_crossentropy
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(categorical_crossentropy(labels, output))


# In[12]:


X_train = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_train_train.npy")
y_train = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_train_train_labels.npy")


y_train1 = list()
for i in y_train:
    if i == b'bonafide':
        y_train1.append(1)
    else:
        y_train1.append(0)
y_train = to_categorical(y_train)
# X_dev_train = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev_train.npy")
# X_dev_val = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev_val.npy")
# y_dev_train = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev_train_labels.npy")
# y_dev_val = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev_val_labels.npy")
# dev_wpt_levels_data_train = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev_wpt_levels_data_train.npy")
# dev_wpt_levels_data_val = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev_wpt_levels_data_val.npy")
# wpt_levels_data_train = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_train_wpt_levels_data_train.npy")
X_val = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev_train.npy")
y_val = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev_train_labels.npy")
# y_val = list(y_val)
# y_val1 = list()
# for i in y_val:
#     if i == b'bonafide':
#         y_val1.append(1)
#     else:
#         y_val1.append(0)
y_val = to_categorical(y_val)
# wpt_levels_data_val = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_train_wpt_levels_data_val.npy")


# ## Normalize data

# In[14]:


row_sums = X_train.sum(axis=1)
X_train = X_train / row_sums[:, np.newaxis]
row_sums = X_val.sum(axis=1)
X_val = X_val / row_sums[:, np.newaxis]


# In[17]:


with tf.name_scope('RMSProp'):
    # Gradient Descent
    optimizer = tf.train.RMSPropOptimizer(1e-3)
    train_step = optimizer.minimize(loss)
    # Op to calculate every variable gradient

tf.summary.scalar("loss", loss)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)
# Create summaries to visualize weights
# for var in tf.trainable_variables():
#     tf.summary.histogram(var.name, var)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Initialize all variables
init_op = tf.global_variables_initializer()

batch_size = 128
logs_path = '/home/rohita/rohit/spoof/npy_data_asvspoof/Sincnet'

# Run training loop
with sess.as_default():
    
    sess.run(init_op)
    
    train_summary_writer = tf.summary.FileWriter(logs_path+'/Train_whole_dataset_RMS',
                                            graph=tf.get_default_graph())
    val_summary_writer = tf.summary.FileWriter(logs_path+'/Val_whole_dataset_RMS')
    gen = generator(X_train, y_train, batch_size)
    gen_val = generator_val(X_val, y_val, batch_size)
    total_batch_train = int(X_train.shape[0]/batch_size)
    total_batch_val = int(X_val.shape[0]/batch_size)
    for epoch in range(100):
        for i in range(total_batch_train):
            X_batch, y_batch = next(gen)
            feed_dict = {inputs: X_batch, labels: y_batch, tf.keras.backend.learning_phase(): 1}
            sess.run(train_step,feed_dict)
            loss_train, acc_train, summary = (sess.run([loss, acc, merged_summary_op],feed_dict))
            train_summary_writer.add_summary(summary, epoch * total_batch_train + i)
            print("Epoch: "+str(epoch)+"step: "+str(i)+"Training loss: ",loss_train," ","Training accuracy"," ",acc_train)
        
#             loss_val,acc_val = (sess.run([loss,acc_value],feed_dict={inputs: X_batch, labels: y_batch, tf.keras.backend.learning_phase(): 0}))
#             print("Epoch: "+str(i)+"step: "+str(step)+"loss: ",loss_val," ","accuracy"," ",acc_val)
            
                    
        for i in range(total_batch_val):
            X_batch, y_batch = next(gen_val)
            loss_val, acc_val, summary = (sess.run([loss, acc, merged_summary_op],feed_dict={inputs: X_batch, labels: y_batch, tf.keras.backend.learning_phase(): 0}))
            val_summary_writer.add_summary(summary, epoch * total_batch_val + i)
            print("val loss: ",loss_val," ","val accuracy"," ",acc_val)


# In[ ]:




