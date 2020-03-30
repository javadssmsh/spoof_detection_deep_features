#!/usr/bin/env python
# coding: utf-8

# In[ ]:





import keras 
import numpy as np
import librosa as lb
import pandas as pd
import os
import sys
# import pandas as pd
from keras.utils import to_categorical
# import soundfile as sf
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import History 
from keras.utils import plot_model,to_categorical
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
# import speechpy as sp
# import statistics
from keras import backend as K
from keras.layers import Dense, Activation, Flatten




# ## Dev set

# In[1]:


filename = "/home/rohita/rohit/spoof/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"

# open the file for reading
filehandle = open(filename, 'r')
dev_protocol = []
while True:
    # read a single line
    line = (filehandle.readline())
    dev_protocol.append(line)
    if not line:
        break

# close the pointer to that file
filehandle.close()


# In[4]:


dev_protocol = [s[:-1] for s in dev_protocol]

dev_protocol = pd.DataFrame([s.split(' ') for s in dev_protocol])

dev_protocol.columns = ['speaker_id','file_id', 'blah','system_id', 'label']

dev_protocol = dev_protocol[['speaker_id', 'file_id', 'system_id', 'label']]

dev_protocol = dev_protocol.dropna()


# In[5]:


#import names of files in dataset
path = r'/home/rohita/rohit/spoof/ASVspoof2019/LA/ASVspoof2019_LA_dev/flac'
files = []
missing=[]
print(path)
for r, d, f in os.walk(path):
    for file in f:
        if '.flac' in file  :        
            files.append(os.path.join(r, file))
        else:
            missing.append(file)
print(len(files))


files = [s.split('/') for s in files]

files = [s[-1] for s in files]

files = [s[:-5] for s in files]

files1 = [s.split(' ') for s in files]

for s in files:
    if len(s)>12:
        print(s)
        files.remove(s)

dev_file_id = list(dev_protocol.iloc[:,1])


# In[ ]:


dev_labels = []
dev_audio = []
for count,audio in enumerate(files):
    index = dev_file_id.index(audio)
    if bool(index) == True:
        dev_audio.append(lb.load('/home/rohita/rohit/spoof/ASVspoof2019/LA/ASVspoof2019_LA_dev/flac/'+audio+'.flac',sr=16000))
        dev_labels.append(dev_protocol.iloc[index,3])
    if count%100 == 0 :
        print(count)
np.save("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev.npy",np.array(dev_audio))
np.save("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_dev_labels.npy",dev_labels)




