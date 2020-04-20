import soundfile as sf
import os
import webrtcvad
import contextlib
import sys
import collections
import wave
from conf import *
from data_io import vad

print("Train data folder: ",train_data_folder)
print("Train data folder: ",flac_lst_train.iloc[0:5,:])
print("Dev data folder: ",dev_data_folder)
print("Dev train_lst: ",flac_lst_dev.iloc[0:5,:])
vad(train_data_folder,flac_lst_train)
vad(dev_data_folder, flac_lst_dev)