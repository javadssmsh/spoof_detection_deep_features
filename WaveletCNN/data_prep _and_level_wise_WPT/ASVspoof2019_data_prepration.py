#!/usr/bin/env python
# coding: utf-8




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


# In[2]:


filename = "/home/rohita/rohit/spoof/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

# open the file for reading
filehandle = open(filename, 'r')
train_protocol = []
while True:
    # read a single line
    line = (filehandle.readline())
    train_protocol.append(line)
    if not line:
        break

# close the pointer to that file
filehandle.close()


# In[3]:


train_protocol = [s[:-1] for s in train_protocol]


# In[4]:


train_protocol = pd.DataFrame([s.split(' ') for s in train_protocol])


# In[ ]:




# In[5]:


train_protocol.columns = ['speaker_id','file_id', 'blah','system_id', 'label']


# In[6]:


train_protocol = train_protocol[['speaker_id', 'file_id', 'system_id', 'label']]


# In[ ]:





# In[7]:


train_protocol = train_protocol.dropna()


# In[46]:




# In[8]:


#import names of files in dataset
path = r'/home/rohita/rohit/spoof/ASVspoof2019/LA/ASVspoof2019_LA_train/flac'
files = []
missing=[]
print(path)
for r, d, f in os.walk(path):
    for file in f:
        if '.flac' in file  :        
            files.append(os.path.join(r, file))
        else:
            missing.append(file)


# In[9]:


files = [s.split('/') for s in files]


# In[10]:


files = [s[-1] for s in files]


# In[11]:


files = [s[:-5] for s in files]


# In[12]:


files1 = [s.split(' ') for s in files]


# In[13]:


for s in files:
    if len(s)>12:
        print(s)
        files.remove(s)


# In[ ]:


train_file_id = list(train_protocol.iloc[:,1])


# In[79]:



# In[ ]:


train_labels = []
train_audio = []
for count,audio in enumerate(files):
    index = train_file_id.index(audio)
    if bool(index) == True:
        train_audio.append(lb.load('/home/rohita/rohit/spoof/ASVspoof2019/LA/ASVspoof2019_LA_train/flac/'+audio+'.flac',sr=16000))
        train_labels.append(train_protocol.iloc[index,3])
    if count%100 == 0 :
        print(count)
np.save("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_train.npy",np.array(train_audio))
np.save("/home/rohita/rohit/spoof/npy_data_asvspoof/ASVspoof2019_train_labels.npy",train_labels)


# In[83]:





