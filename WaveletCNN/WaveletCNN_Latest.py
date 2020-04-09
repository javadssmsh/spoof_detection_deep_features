#!/usr/bin/env python
# coding: utf-8


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";  



import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)


# In[2]:


import warnings
warnings.simplefilter(action='ignore', category=Warning)
import numpy as np
import librosa as lb
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
# import tfwavelets
# sys.path.insert(1, '/content/drive/My Drive/SA/Code/spoof_detection_deep_features/WaveletCNN/cwt-tensorflow')
# from cwt import cwtMortlet, cwtRicker,mortletWavelet, rickerWavelet
# tf.compat.v1.enable_eager_execution()


# import pydot


# In[ ]:




# In[3]:



def framing_windowing(signal):
    pre_emphasis = 0.97
    frame_size = 3200
    frame_stride = 160
    nfilt = 20
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    frame_length, frame_step = frame_size, frame_stride  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
#     print(num_frames)
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)
    # print(frames.shape)
    return frames

# In[4]:


def tkeo(a):

    """
    Calculates the TKEO of a given recording by using 2 samples.
    See Li et al., 2007
    Arguments:
    a 			--- 1D numpy array.
    Returns:
    1D numpy array containing the tkeo per sample
    """
    # Create two temporary arrays of equal length, shifted 1 sample to the right
    # and left and squared:
    i = a[1:-1]*a[1:-1]
    j = a[2:]*a[:-2]
    # Calculate the difference between the two temporary arrays:
    aTkeo = i-j
    return aTkeo

def Wavelet_1d(signal):
    import numpy
    nfilt = 3200
    NFFT = 8191
    sample_rate = 16000
    # signal = tkeo(signal)
    mel = lb.filters.mel(sr=sample_rate, n_fft=NFFT, n_mels=nfilt)
    audio_features = []
    tke = tkeo(signal)
    data_std = StandardScaler().fit_transform(tke.reshape(-1,1)).reshape(1,-1)[0]            
    wptree = pywt.WaveletPacket(data=data_std, wavelet='db1', mode='symmetric')
    level = wptree.maxlevel
    levels = wptree.get_level(level, order = "freq")            
        #Feature extraction for each node
    frame_features = []        
    for node in levels:
        data_wp = node.data
            # Features group
        frame_features.extend(data_wp)
#     print(np.array(frame_features).shape)
    mag_frames = numpy.absolute(frame_features)  # Magnitude of the FFT
    pow_frames = numpy.abs((mag_frames) ** 2)
    mel_scaled_features = mel.dot(pow_frames)
    log_energy = numpy.log10(mel_scaled_features)
    log_energy = pd.DataFrame(log_energy)
    pd.set_option('use_inf_as_null', True)
    log_energy=log_energy.fillna(log_energy.mean())
    return np.array(log_energy)

# In[5]:
def generator_part1(data, labels,batch_size):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[audio1,label1], [audio2,label2],...].
    """
    x_samples = []
    y_samples = []
    for count,i in enumerate(data):
        x_samples = framing_windowing(i)
        num_samples = len(x_samples)
        y_samples = [labels[count]]*num_samples
#         print("yo")
 
      # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
        for offset in range(0, num_samples, batch_size):
        # Get the samples you'll use in this batch
            batch_samples = x_samples[offset:offset+batch_size]

        # Initialise X_train and y_train arrays for this batch
            X_train = []
            y_train = []
            wpt = []
        # For each example
            for count,x_sample in enumerate(batch_samples):
            # audio (X) and label (y)
                audio =  x_sample
#             print(level1.shape)
                label = y_samples[count]
            
            # Add example to arrays
                X_train.append(audio)
                y_train.append(label)
                wpt.append(Wavelet_1d(audio))


        # Make sure they're numpy arrays (as opposed to lists)
            X_train = np.array(X_train)
            y_train = np.array(y_train)
#             print(y_train)
#             y_train = to_categorical(y_train)
            wpt = np.array(wpt)
#             print(y_train.shape)
        # The generator-y part: yield the next training batch            
            yield X_train,wpt, y_train
        
        

def generator_val_part1(data, labels):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[audio1,label1], [audio2,label2],...].
    """
    x_samples = []
    y_samples = []
    for count,i in enumerate(data):
        x_samples = framing_windowing(i)
        num_samples = len(x_samples)
        y_samples = [labels[count]]*num_samples
        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
        
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = x_samples[offset:offset+batch_size]

            # Initialise X_train and y_train arrays for this batch
            X_train = []
            y_train = []
            wpt = []
            # For each example
            for count,x_sample in enumerate(batch_samples):
                # audio (X) and label (y)
                audio =  x_sample
#                 print(level1.shape)
                label = y_samples[count]
                
                # Add example to arrays
                X_train.append(audio)
                y_train.append(label)
                wpt.append(Wavelet_1d(audio))


            # Make sure they're numpy arrays (as opposed to lists)
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            wpt = np.array(wpt)
#             print("hello")
            # The generator-y part: yield the next training batch            
            yield X_train, wpt, y_train






input_shape = None,3200
    
input_ = tf.placeholder(tf.float32, shape=input_shape, name= 'the_input')
reshaped_input_ =  Reshape((3200,1))(input_)
input_l1 = tf.placeholder(tf.float32, shape=(None,3200,1), name= 'input_l1')

# level one decomposition starts
print("hi")

conv_1 = Conv1D(64, 251, strides=1, padding='valid',name = 'conv_1')(input_l1)
pool_1 = MaxPooling1D(pool_size=3,name = 'pool_1')(conv_1)
norm_1 = BatchNormalization(momentum=0.05, name = 'norm_1')(pool_1)
relu_1 = LeakyReLU(alpha=0.2, name = 'relu_1')(norm_1)


conv_1_2 = Conv1D(64, 5, strides=2, padding='valid',name = 'conv_1_2')(relu_1)
# pool_1_2 = MaxPooling1D(pool_size=3,name = 'pool_1')(sinc_conv)
norm_1_2 = BatchNormalization(momentum=0.05, name = 'norm_1_2')(conv_1_2)
relu_1_2 = LeakyReLU(alpha=0.2, name = 'relu_1_2')(norm_1_2) 

# paddings = tf.constant([[0, 0],   # the batch size dimension
#                         [261, 260],   # top and bottom of image
#                         [0, 0]])  # the channels dimension
# padded_input = Lambda(lambda x: tf.pad(x, paddings, mode='CONSTANT',
#                       constant_values=0.0))(relu_1_2)

# level two decomposition starts
sinc = sincnet.SincConv1D(64, 251, 16000)(reshaped_input_)
sinc_pool = MaxPooling1D(pool_size=3,name = 'sinc_pool')(sinc)
sinc_norm = BatchNormalization(momentum=0.05, name = 'sinc_norm')(sinc_pool)
sinc_layer_norm = sincnet.LayerNorm(name = 'sinc_layer_norm')(sinc_norm)
sinc_relu = LeakyReLU(alpha=0.2, name = 'sinc_relu')(sinc_layer_norm)

sinc_conv = Conv1D(64, 5, strides=2, padding='valid')(sinc_layer_norm)
# sinc_pool_1 = MaxPooling1D(pool_size=3,name = 'sinc_pool_1')(sinc_conv)
sinc_norm_1 = BatchNormalization(momentum=0.05, name = 'sinc_norm_1')(sinc_conv)
sinc_layer_norm_1 = sincnet.LayerNorm(name = 'sinc_layer_norm_1')(sinc_norm_1)
sinc_relu_1 = LeakyReLU(alpha=0.2, name = 'sinc_relu_1')(sinc_layer_norm_1)


    
#concate level one and level two decomposition
concate_level_2 = concatenate([relu_1_2,sinc_relu_1])
conv_2 = Conv1D(64, 5, strides=1, padding='valid',name = 'conv_2')(concate_level_2)
# pool_1 = MaxPooling1D(pool_size=3,name = 'pool_2')(sinc_conv)
norm_2 = BatchNormalization(momentum=0.05, name = 'norm_2')(conv_2)
relu_2 = LeakyReLU(alpha=0.2, name = 'relu_2')(norm_2)



conv_2_2 = Conv1D(64, 5, strides=1, padding='valid',name = 'conv_2_2')(relu_2)
# pool_2_2 = MaxPooling1D(pool_size=3,name = 'pool_1')(sinc_conv)
norm_2_2 = BatchNormalization(momentum=0.05, name = 'norm_2_2')(conv_1_2)
relu_2_2 = LeakyReLU(alpha=0.2, name = 'relu_2_2')(norm_1_2) 

#level three decomposition starts 


pool_5_1 = AveragePooling1D(pool_size=3, padding='same', name='avg_pool_5_1')(relu_2_2)
flat_5_1 = Flatten(name='flat_5_1')(pool_5_1) 

fc_5 = Dense(2048, name='fc_5')(flat_5_1)
norm_5 = BatchNormalization(name='norm_5')(fc_5)
relu_5 = Activation('relu', name='relu_5')(norm_5)
drop_5 = Dropout(0.5, name='drop_5')(relu_5)

fc_6 = Dense(2048, name='fc_6')(drop_5)
norm_6 = BatchNormalization(name='norm_6')(fc_6)
relu_6 = Activation('relu', name='relu_6')(norm_6)
drop_6 = Dropout(0.5, name='drop_6')(relu_6)

output = Dense(2, activation=tf.nn.softmax)(drop_6)


labels = tf.placeholder(tf.float32, shape=(None,2))
from keras.metrics import categorical_accuracy as accuracy
acc_value = tf.reduce_mean(accuracy(labels, output))
# correct_pred = tf.equal(output, labels)
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
from keras.objectives import categorical_crossentropy
loss = tf.reduce_mean(categorical_crossentropy(labels, output))


# In[9]:
saver = tf.train.Saver()

X_train = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_train_train.npy")
y_train = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_train_train_labels.npy")
y_train = to_categorical(y_train)
# X_dev_train = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev_train.npy")
# X_dev_val = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev_val.npy")
# y_dev_train = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev_train_labels.npy")
# y_dev_val = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev_val_labels.npy")
# dev_wpt_levels_data_train = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev_wpt_levels_data_train.npy")
# dev_wpt_levels_data_val = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev_wpt_levels_data_val.npy")
# wpt_levels_data_train = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_train_wpt_levels_data_train.npy")
X_val = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_train_val.npy")
y_val = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_train_val_labels.npy")
y_val = to_categorical(y_val)
# wpt_levels_data_val = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_train_wpt_levels_data_val.npy")

# In[20]:


# X_train = np.concatenate([X_train,X_dev_train,X_dev_val])
# wpt_levels_data_train = np.concatenate([wpt_levels_data_train,dev_wpt_levels_data_train,dev_wpt_levels_data_val])


# In[10]:


batch_size = 8
# X_batch = np.array((batch_size,75673))
# y_train = np.array((batch_size))
train_step = tf.train.RMSPropOptimizer(0.001, decay=0.9, epsilon=1e-8).minimize(loss)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Run training loop
with sess.as_default():
    for i in range(100):
        step = 0
        for X_batch,wpt, y_batch in generator_part1(X_train, y_train, 301):
            step = step + 1
            feed_dict = {input_: X_batch, labels: y_batch, input_l1: wpt, K.learning_phase(): 1}
            sess.run(train_step,feed_dict)
#             print(sess.run(output,feed_dict))
            if step % 1 == 0:
                loss_val,acc_val = (sess.run([loss,acc_value],feed_dict={input_: X_batch, labels: y_batch, input_l1: wpt, K.learning_phase(): 1}))
                print("Epoch: "+ str(i)+" Step: "+str(step)+"Training loss: "+str(loss_val)+" "+"Training accuracy"+" "+str(acc_val))
                val_acc_val = (sess.run(acc_value,feed_dict={input_: X_batch, labels: y_batch, input_l1: wpt, K.learning_phase(): 0}))
                print("Epoch: "+ str(i)+" Step: "+str(step)+" validation accuracy"+" "+str(val_acc_val))
            
        X_batch, wpt, y_batch = generator_val_part1(X_val, y_val, 100)
        test_acc_val = (sess.run(acc_value,feed_dict={input_: X_batch, labels: y_batch, input_l1: wpt, K.learning_phase(): 0}))
        print("Epoch: "+ str(i)+" test accuracy "+str(test_acc_val))
        save_path = saver.save(sess, "/home/rohita/rohit/spoof/npy_data_asvspoof/WaveletCNN_Latest/WCNN_"+str(i)+".ckpt")
        print("Model saved in file: %s" % save_path)


# In[11]:


# X_train = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_train_train.npy")
# y_train = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_train_train_labels.npy")
# wpt_levels_data_train = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_train_wpt_levels_data_train.npy")


