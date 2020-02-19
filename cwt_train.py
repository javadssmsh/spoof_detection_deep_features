import numpy as np
import pywt
import librosa as lb
import speechpy as sp
from scipy.fftpack import dct



def cwt(x,n):
    scales = range(1,n)
    waveletname = 'morl'
    signal = x
    coeff, freq = pywt.cwt(signal, np.asarray(scales), waveletname, 1)
    coeff_ = coeff[:,:int(np.floor(n / 2 + 1))]
    return coeff_



def mfcc_cwt(signal,sample_rate,num_ceps):
    pre_emphasis = 0.97
    frame_size = 0.025
    frame_stride = 0.01
    nfilt = 20
    NFFT = 128
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
#     frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
#     signal_length = len(emphasized_signal)
#     frame_length = int(round(frame_length))
#     frame_step = int(round(frame_step))
#     num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

#     pad_signal_length = num_frames * frame_step + frame_length
#     z = np.zeros((pad_signal_length - signal_length))
#     pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

#     indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
#     frames = pad_signal[indices.astype(np.int32, copy=False)]
#     frames *= np.hamming(frame_length)
    mag_frames = np.absolute(cwt(emphasized_signal, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    mfcc = dct(filter_banks, type=2, axis=1,n=num_ceps, norm='ortho') # Keep 2-13
    return mfcc

train_labels = np.load("/media/hinton/F8E62A5EE62A1D7E/rohit/spoof/spoof_deep_features/train_label.npy")
trimmed_audio = np.load("/media/hinton/F8E62A5EE62A1D7E/rohit/spoof/original_datasets/npy_data_asvspoof/trimmed_audio.npy")

delta = []
delta2 = []
value = []
cwt_feat_train = np.empty((16000,7620))
shape = []
mean_value = []

for count,f in enumerate(trimmed_audio):
#     print(f.shape)
    
    delta=np.array(lb.feature.delta(mfcc_cwt(f,16000, num_ceps=30)))
    
    delta2=np.array(lb.feature.delta(mfcc_cwt(f,16000, num_ceps=30), order=2))
    value = np.concatenate([delta,delta2], axis=1)
#     print(value.shape,delta.shape,delta2.shape)
#     shape = np.append(shape,(value.shape[0]))
    mean_value = sp.processing.cmvnw(value).reshape(1,-1)
#     print(len(mean_value[0]))
#     padded_value =pad_sequences(mean_value,maxlen=74340,dtype='float32')
    cwt_feat_train[count] = mean_value

    if count%100==0:
        print(count)
        
        
np.save("/media/hinton/F8E62A5EE62A1D7E/rohit/spoof/spoof_deep_features/cwt_train.npy",cwt_feat_train,allow_pickle=True)        