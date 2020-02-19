import numpy as np
import pywt
import librosa as lb
import speechpy as sp
from keras.utils import plot_model,to_categorical
from sklearn.decomposition import PCA
import keras 
import numpy as np
import librosa as lb
import sys
import os
import pandas as pd
from keras.utils.training_utils import multi_gpu_model
from keras.utils import to_categorical,Sequence
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import History 
from keras.utils import plot_model,to_categorical
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
# from iter_window import window 
import speechpy as sp
# import statistics
from keras import backend as K
from keras.layers import Dense, Activation, Flatten
from sklearn.preprocessing import StandardScaler

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

def mwpc(signal,sample_rate):
    pre_emphasis = 0.97
    frame_size = 256
    frame_stride = 128
    nfilt = 20
    NFFT = 511
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    frame_length, frame_step = frame_size, frame_stride  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
#     print(num_frames)
    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]
    frames *= numpy.hamming(frame_length)
#     print(frames.shape)
    audio_features = []
    mel = lb.filters.mel(sr=sample_rate, n_fft=NFFT, n_mels=nfilt)
    for f in frames:
        tke = tkeo(f)
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
#         print(len(frame_features))
        mag_frames = numpy.absolute(frame_features)  # Magnitude of the FFT
        pow_frames = numpy.abs((mag_frames) ** 2)
        mel_scaled_features = mel.dot(pow_frames)
        audio_features.append(mel_scaled_features)
    
    
#     print("hello")
    log_energy = numpy.log10(audio_features)
    log_energy = pd.DataFrame(log_energy)
    pd.set_option('use_inf_as_null', True)
    log_energy=log_energy.fillna(log_energy.mean())
#     print(log_energy)
    
    pca = PCA(n_components=12)
    mwpc = pca.fit_transform(log_energy)
#     print(mwpc.shape)
    
    return mwpc

train_labels = np.load("/media/hinton/F8E62A5EE62A1D7E/rohit/spoof/spoof_deep_features/train_label.npy")
trimmed_audio = np.load("/media/hinton/F8E62A5EE62A1D7E/rohit/spoof/original_datasets/npy_data_asvspoof/trimmed_audio.npy")

delta = []
delta2 = []
value = []
mwpc_train = np.empty((16000,21240))
mean_value = []

for count,f in enumerate(trimmed_audio):
    
    mwpc_feat = mwpc(f,16000)
    delta=np.array(lb.feature.delta(mwpc_feat))
#     print(delta.shape)
    delta2=np.array(lb.feature.delta(mwpc_feat, order=2))
#     print(delta2.shape)
    value = np.concatenate([mwpc_feat,delta,delta2], axis=1)
#     print(len(value[1]))
    mean_value = sp.processing.cmvnw(value).reshape(1,-1)
#     print(len(mean_value[0]))
    mwpc_train[count] = mean_value

    if count%100==0:
        print(count)
        
        
np.save("/media/hinton/F8E62A5EE62A1D7E/rohit/spoof/spoof_deep_features/mwpc_train.npy",mwpc_train,allow_pickle=True)        