#!/usr/bin/env python
# coding: utf-8


# In[1]:


import warnings
warnings.simplefilter(action='ignore', category=Warning)
import numpy as np
import librosa as lb
import tensorflow as tf
import pandas as pd
import math
# from tensorflow import set_random_seed
import numpy as np
from sklearn.preprocessing import StandardScaler
import pywt
from sklearn.decomposition import PCA
import sys


# In[ ]:


def framing_windowing(signal):
    pre_emphasis = 0.97
    frame_size = 256
    frame_stride = 128
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


# In[2]:


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


# In[3]:


def Wavelet_1d(signal):
    import numpy
    pre_emphasis = 0.97
    frame_size = 256
    frame_stride = 128
    nfilt = 20
    NFFT = 511
    sample_rate = 16000
    signal = framing_windowing(signal)
# signal = tkeo(signal)
    mel = lb.filters.mel(sr=sample_rate, n_fft=NFFT, n_mels=nfilt)
    audio_features1 = []
    audio_features2 = []
    audio_features3 = []
    audio_features4 = []
# print("signal shape",signal.shape)
    for f in signal:
        tke = tkeo(f)
        data_std = StandardScaler().fit_transform(tke.reshape(-1,1)).reshape(1,-1)[0]            
        wptree = pywt.WaveletPacket(data=data_std, wavelet='db1', mode='symmetric')
        level1 = wptree.get_level(1, order = "freq")
        level2 = wptree.get_level(2, order = "freq")
        level3 = wptree.get_level(3, order = "freq")
        level4 = wptree.get_level(4, order = "freq")
        # print("level1 data array:",np.array(level1).shape)
        # print("level2 data array:",np.array(level2).shape)
        # print("level3 data array:",np.array(level3).shape)
        # print("level4 data array:",np.array(level4).shape)          
          #Feature extraction for each node
        frame_features1 = []
        frame_features2 = []
        frame_features3 = []
        frame_features4 = []        
        for node in level1:
            data_wp = node.data
          # print("WP data:",np.array(data_wp).shape)
          # Features group
            frame_features1.extend(data_wp)
        # print("frame_features1",np.array(frame_features1).shape)
        mag_frames = numpy.absolute(frame_features1)  # Magnitude of the FFT
        pow_frames = numpy.abs((mag_frames) ** 2)
        z = mel.shape[1] - pow_frames.shape[0]
        pow_frames = np.pad(pow_frames,[(0,z)],'constant', constant_values=0)
        # print("pow_frames",pow_frames.shape)
        mel_scaled_features = mel.dot(pow_frames)
        audio_features1.append(mel_scaled_features)


        for node in level2:
            data_wp = node.data
          # print("WP data:",np.array(data_wp).shape)
          # Features group
            frame_features2.extend(data_wp)
        mag_frames = numpy.absolute(frame_features2)  # Magnitude of the FFT
        pow_frames = numpy.abs((mag_frames) ** 2)
        mel_scaled_features = mel.dot(pow_frames)
        audio_features2.append(mel_scaled_features)
        for node in level3:
            data_wp = node.data
          # print("WP data:",np.array(data_wp).shape)
          # Features group
            frame_features3.extend(data_wp)
        mag_frames = numpy.absolute(frame_features3)  # Magnitude of the FFT
        pow_frames = numpy.abs((mag_frames) ** 2)
        mel_scaled_features = mel.dot(pow_frames)
        audio_features3.append(mel_scaled_features)
        for node in level4:
            data_wp = node.data
          # print("WP data:",np.array(data_wp).shape)
          # Features group
            frame_features4.extend(data_wp)
        mag_frames = numpy.absolute(frame_features4)  # Magnitude of the FFT
        pow_frames = numpy.abs((mag_frames) ** 2)
        mel_scaled_features = mel.dot(pow_frames)
        audio_features4.append(mel_scaled_features)
    
  # print("audio_features1:",np.array(audio_features1).shape)
  # print("audio_features1:",np.array(audio_features2).shape)
  # print("audio_features1:",np.array(audio_features3).shape)
  # print("audio_features1:",np.array(audio_features4).shape)
#     print("hello")
    log_energy1 = numpy.log10(audio_features1)
    log_energy1 = pd.DataFrame(log_energy1)
    log_energy2 = numpy.log10(audio_features2)
    log_energy2 = pd.DataFrame(log_energy2)
    log_energy3 = numpy.log10(audio_features3)
    log_energy3 = pd.DataFrame(log_energy3)
    log_energy4 = numpy.log10(audio_features4)
    log_energy4 = pd.DataFrame(log_energy4)
    pd.set_option('use_inf_as_null', True)
    log_energy1=log_energy1.fillna(log_energy1.mean())
    log_energy1 = np.array(log_energy1)
    log_energy2=log_energy2.fillna(log_energy2.mean())
    log_energy2 = np.array(log_energy2)
    log_energy3=log_energy3.fillna(log_energy3.mean())
    log_energy3 = np.array(log_energy3)
    log_energy4=log_energy4.fillna(log_energy4.mean())
    log_energy4 = np.array(log_energy4)
  # print("log_energy1: ",log_energy1.shape)
  # print("log_energy2: ",log_energy2.shape)
  # print("log_energy3: ",log_energy3.shape)
  # print("log_energy4: ",log_energy4.shape)
  # signals_level1[count] = np.array(log_energy1)
  # signals_level2[count] = np.array(log_energy2)
  # signals_level3[count] = np.array(log_energy3)
  # signals_level4[count] = np.array(log_energy4)
  # print(signals_level1.shape)
  # print(signals_level2.shape)
  # print(signals_level3.shape)
  # print(signals_level4.shape)  

    return log_energy1,log_energy2,log_energy3,log_energy4

def Wavelet_out_shape(input_shapes):
    # print('in to shape')
    return [tuple([None, 399, 20]), tuple([None, 399, 20]), 
            tuple([None, 399, 20]), tuple([None, 399, 20])]


# In[4]:


# train_labels = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/train_label.npy")
audio_train = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev_train.npy")
audio_val = np.load("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev_val.npy")
# audio_train = audio_train.astype('float32')
# audio_val = audio_val('float32')

# In[77]:


wpt_levels_data_train = []
for count,i in enumerate(audio_train):
    if count%10 == 0:
        print(count)
    level1,level2,level3,level4 = Wavelet_1d(i)
    wpt_levels_data_train.append([level1,level2,level3,level4])
wpt_levels_data_train = np.array(wpt_levels_data_train)
np.save("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev_wpt_levels_data_train.npy",wpt_levels_data_train)
print(wpt_levels_data_train.shape)

wpt_levels_data_val = []
for count,i in enumerate(audio_val):
    if count%10 == 0:
        print(count)
    level1,level2,level3,level4 = Wavelet_1d(i)
    wpt_levels_data_val.append([level1,level2,level3,level4])
wpt_levels_data_val = np.array(wpt_levels_data_val)
np.save("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev_wpt_levels_data_val.npy",wpt_levels_data_val)
print(wpt_levels_data_val.shape)


# In[ ]:




