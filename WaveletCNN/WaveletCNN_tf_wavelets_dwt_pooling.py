#!/usr/bin/env python
# coding: utf-8

# ## Setup for google colab


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1";  


# In[2]:


import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)


# In[3]:


import warnings
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
import pandas as pd
import math
import numpy as np
from tensorflow.python.keras.layers import MaxPooling1D,MaxPooling2D,AveragePooling1D, Conv1D, LeakyReLU, BatchNormalization, Dense, Flatten, concatenate, Activation
from tensorflow.python.keras.layers import InputLayer, Input, Layer, Lambda, Dropout
from keras.utils import to_categorical
from tensorflow.python.keras.models import Model
from sklearn.preprocessing import StandardScaler
import tfwavelets
import keras
import tensorflow.contrib.slim as slim


# In[4]:


def generator(data, labels, batch_size):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[audio1,label1], [audio2,label2],...].
    """
    total_batches = int(data.shape[0])
    for i in range(total_batches):
        X_train,y_train = create_batches_rnd(data, labels, batch_size)
        yield X_train, y_train


# In[5]:


def generator_val(data, labels, batch_size):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[audio1,label1], [audio2,label2],...].
    """
    total_batches = int(data.shape[0])
    for i in range(total_batches):
        X_train,y_train = create_batches_rnd(data, labels, batch_size)
        yield X_train, y_train


# In[6]:


def create_batches_rnd(data,labels,batch_size):
    wlen = 3200
    fact_amp = 0.2
    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    sig_batch=np.zeros([batch_size,wlen])
    lab_batch=[]
    signal_id_arr=np.random.randint(data.shape[0], size=batch_size)
    rand_amp_arr = np.random.uniform(1.0-fact_amp,1+fact_amp,batch_size)
    for i in range(batch_size): 
        signal = data[signal_id_arr[i]]
        # accesing to a random chunk
        signal_len=signal.shape[0]
        signal_beg=np.random.randint(signal_len-wlen-1) #randint(0, snt_len-2*wlen-1)
        signal_end=signal_beg+wlen
        sig_batch[i,:]=signal[signal_beg:signal_end]*rand_amp_arr[i]
        y=labels[signal_id_arr[i]]
        lab_batch.append(y)
    a, b = np.shape(sig_batch)
    sig_batch = sig_batch.reshape((a, b, 1))
    return sig_batch, np.array(lab_batch)


# In[7]:


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


# In[8]:


def dwt(x,db2):
    # x1_ = tf.placeholder(tf.float32, shape=(None,3200,3), name= 'x1')
    dwt = tfwavelets.nodes.dwt1d(x,db2,1)
    return dwt

    


def iwt(x,db2):
    
    idwt = tfwavelets.nodes.idwt1d(x,db2,1)
    return idwt
db2 = tfwavelets.dwtcoeffs.Wavelet(
    tfwavelets.dwtcoeffs.Filter(np.array([-0.12940952255092145,
                     0.22414386804185735,
                     0.836516303737469,
                     0.48296291314469025]), 3),
    tfwavelets.dwtcoeffs.Filter(np.array([-0.48296291314469025,
                     0.836516303737469,
                     -0.22414386804185735,
                     -0.12940952255092145]), 0),
    tfwavelets.dwtcoeffs.Filter(np.array([0.48296291314469025,
                     0.836516303737469,
                     0.22414386804185735,
                     -0.12940952255092145]), 0),
    tfwavelets.dwtcoeffs.Filter(np.array([-0.12940952255092145,
                     -0.22414386804185735,
                     0.836516303737469,
                     -0.48296291314469025]), 3)
)

class DWT_Pooling(Layer):
    """
    # Input shape :
        
            4D tensor of shape: (batch_size, signal, channels)
        
            
    # Output shape
        
            4D tensor of shape: (batch_size, signal/2, channels*4)
        
    """

    def __init__(self,**kwargs):
        super(DWT_Pooling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DWT_Pooling, self).build(input_shape) 

    def call(self, x):
        return dwt(x,db2)

    def compute_output_shape(self, input_shape):
        
        return (input_shape[0], input_shape[1]//2, input_shape[2]*4)


class IWT_UpSampling(Layer):
    """
    # Input shape :
        
            4D tensor of shape: (batch_size, signal, channels)
        
    # Output shape
        
            4D tensor of shape: (batch_size, singal*2, channels/4)
        
    """

    def __init__(self, **kwargs):
        super(IWT_UpSampling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(IWT_UpSampling, self).build(input_shape) 

    def call(self, x):
        return iwt(x,db2)

    def compute_output_shape(self, input_shape):
        
        return (input_shape[0], input_shape[1]*2, input_shape[2]//4)


# In[10]:


# def unetWavelet(input_size = (3200,1)):


input_shape = None,3200,1
    

inputs = tf.placeholder(tf.float32, shape=input_shape, name= 'the_input')
input_l1 = DWT_Pooling()(inputs)
input_l2 = DWT_Pooling()(input_l1)
input_l3 = DWT_Pooling()(input_l2)
input_l4 = DWT_Pooling()(input_l3)
# level one decomposition starts
conv_1 = Conv1D(filters=64, kernel_size= 3, padding="same")(input_l1)
norm_1 = BatchNormalization(name='norm_1')(conv_1)
relu_1 = Activation('relu', name='relu_1')(norm_1)

conv_1_2 = Conv1D(filters=64, kernel_size= 3, padding="same")(relu_1)#strides = 2, padding="same")(relu_1)
norm_1_2 = BatchNormalization(name='norm_1_2')(conv_1_2)
relu_1_2 = Activation('relu', name='relu_1_2')(norm_1_2)

# level two decomposition starts
conv_a = Conv1D(filters=64, kernel_size= 3, padding="same")(input_l2)
norm_a = BatchNormalization(name='norm_a')(conv_a)
relu_a = Activation('relu', name='relu_a')(norm_a)

# concate level one and level two decomposition
concate_level_2 = concatenate([relu_1_2, relu_a,])
conv_2 = Conv1D(filters=128, kernel_size= 3, padding="same")(concate_level_2)
norm_2 = BatchNormalization(name='norm_2')(conv_2)
relu_2 = Activation('relu', name='relu_2')(norm_2)

conv_2_2 = Conv1D(filters=128, kernel_size= 3, padding="same")(relu_2)#strides = 2, padding="same")(relu_2)
norm_2_2 = BatchNormalization(name='norm_2_2')(conv_2_2)
relu_2_2 = Activation('relu', name='relu_2_2')(norm_2_2)

# level three decomposition starts 
conv_b = Conv1D(filters=64, kernel_size= 3, padding="same")(input_l3)
norm_b = BatchNormalization(name='norm_b')(conv_b)
relu_b = Activation('relu', name='relu_b')(norm_b)

conv_b_2 = Conv1D(filters=128, kernel_size= 3, padding="same")(relu_b)
norm_b_2 = BatchNormalization(name='norm_b_2')(conv_b_2)
relu_b_2 = Activation('relu', name='relu_b_2')(norm_b_2)

# concate level two and level three decomposition 
concate_level_3 = concatenate([relu_2_2, relu_b_2])
conv_3 = Conv1D(filters=256, kernel_size= 3, padding="same")(concate_level_3)
norm_3 = BatchNormalization(name='nomr_3')(conv_3)
relu_3 = Activation('relu', name='relu_3')(norm_3)

conv_3_2 = Conv1D(filters=256, kernel_size= 3, padding="same")(relu_3)#strides = 2, padding="same")(relu_3)
norm_3_2 = BatchNormalization(name='norm_3_2')(conv_3_2)
relu_3_2 = Activation('relu', name='relu_3_2')(norm_3_2)

# level four decomposition start
conv_c = Conv1D(filters=64, kernel_size= 3, padding="same")(input_l4)
norm_c = BatchNormalization(name='norm_c')(conv_c)
relu_c = Activation('relu', name='relu_c')(norm_c)

conv_c_2 = Conv1D(filters=256, kernel_size= 3, padding="same")(relu_c)
norm_c_2 = BatchNormalization(name='norm_c_2')(conv_c_2)
relu_c_2 = Activation('relu', name='relu_c_2')(norm_c_2)

conv_c_3 = Conv1D(filters=256, kernel_size= 3, padding="same")(relu_c_2)
norm_c_3 = BatchNormalization(name='norm_c_3')(conv_c_3)
relu_c_3 = Activation('relu', name='relu_c_3')(norm_c_3)

# concate level level three and level four decomposition
concate_level_4 = concatenate([relu_3_2, relu_c_3])
conv_4 = Conv1D(filters=256, kernel_size= 3, padding="same")(concate_level_4)
norm_4 = BatchNormalization(name='norm_4')(conv_4)
relu_4 = Activation('relu', name='relu_4')(norm_4)

conv_4_2 = Conv1D(filters=256, kernel_size= 3, strides = 2, padding="same")(relu_4)
norm_4_2 = BatchNormalization(name='norm_4_2')(conv_4_2)
relu_4_2 = Activation('relu', name='relu_4_2')(norm_4_2)

conv_5_1 = Conv1D(filters=128, kernel_size= 3, padding="same")(relu_4_2)
norm_5_1 = BatchNormalization(name='norm_5_1')(conv_5_1)
relu_5_1 = Activation('relu', name='relu_5_1')(norm_5_1)

res_conv_1 = res_conv_block(relu_5_1, 128, 64, 1, 'a', 4)
res_conv_2 = res_conv_block(res_conv_1, 64, 32, 2, 'a', 8)
res_conv_3 = res_conv_block(res_conv_2, 32, 16, 3, 'a', 16)
res_conv_4 = res_conv_block(res_conv_3, 16, 4, 4, 'a', 32)
res_conv_5 = res_conv_block(res_conv_4, 4, 2, 5, 'a', 64)

res_norm = BatchNormalization(1,name='res_norm')(res_conv_5)
res_relu = Activation('relu')(res_norm)


pool_5_1 = AveragePooling1D(pool_size=7 , strides=2, padding='same', name='avg_pool_5_1')(res_relu)
flat_5_1 = Flatten(name='flat_5_1')(pool_5_1) 


output = Dense(2, activation='softmax', name='fc_5')(flat_5_1)

# model = Model(input = inputs, output = output)
    
# return model


# In[11]:


def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

model_summary()


# In[12]:


labels = tf.placeholder(tf.float32, shape=(None,2))

from keras.metrics import categorical_accuracy as accuracy


with tf.name_scope('Accuracy'):
    acc= tf.reduce_mean(accuracy(labels, output))

from keras.objectives import categorical_crossentropy
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(categorical_crossentropy(labels, output))


# In[13]:


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


# In[ ]:


with tf.name_scope('Adam'):
    # Gradient Descent
    optimizer = tf.train.AdamOptimizer(1e-3)
    train_step = optimizer.minimize(loss)
    # Op to calculate every variable gradient

tf.summary.scalar("loss", loss)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)
# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Initialize all variables
init_op = tf.global_variables_initializer()

batch_size = 100
logs_path = '/home/rohita/rohit/spoof/npy_data_asvspoof/WCNN'

# Run training loop
with sess.as_default():
    
    sess.run(init_op)
    
    train_summary_writer = tf.summary.FileWriter(logs_path+'/Train_whole_dataset_Adam_lr_1e-3_100',
                                            graph=tf.get_default_graph())
    val_summary_writer = tf.summary.FileWriter(logs_path+'/Val_whole_dataset_Adam_lr_1e-3_100')
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
            


# ## Tensorboard event log for Train and validation

# In[24]:


# count = 0
# for e in tf.train.summary_iterator('/home/rohita/rohit/spoof/npy_data_asvspoof/MWCNN/Train/events.out.tfevents.1585208776.anands-MS-7A93'):
#     for v in e.summary.value:
#         if v.tag == 'loss':
#             count = count + 1
#             print("Step: "+str(count),v)
#         elif v.tag == 'accuracy':
#             print(v)


# # In[26]:


# count = 0
# for e in tf.train.summary_iterator('/home/rohita/rohit/spoof/npy_data_asvspoof/MWCNN/Val/events.out.tfevents.1585208859.anands-MS-7A93'):
#     for v in e.summary.value:
#         if v.tag == 'loss':
#             count = count + 1
#             print("Step: "+str(count),v)
#         elif v.tag == 'accuracy':
#             print(v)


# # In[ ]:





# In[ ]:





# In[ ]:




