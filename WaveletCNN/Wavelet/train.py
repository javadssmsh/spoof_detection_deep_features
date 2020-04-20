import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1";  
import soundfile as sf
from conf import *
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import numpy as np
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras import backend as K
K.clear_session()
from modelWCNN import getModel
import librosa as lb
from data_io import batchGenerator, batchGenerator_val
from keras.callbacks import LearningRateScheduler
  
def lr_scheduler(epoch, lr):
    if epoch == 1:
        decay_rate = 1
        return lr * decay_rate
    elif epoch == 2:
        decay_rate = 10
        return lr*decay_rate
    elif epoch == 3:
        decay_rate = 10
        return lr*decay_rate
    elif epoch == 4:
        decay_rate = 0.1
        return lr*decay_rate
    elif epoch == 5:
        decay_rate = 1
        return lr*decay_rate
    else:
        decay_rate = 1
        return lr*decay_rate
    return lr




if __name__ == "__main__":

    

    print('N_filt ' + str(cnn_N_filt))
    print('N_filt len ' + str(cnn_len_filt))
    print('FS ' + str(fs))
    print('WLEN ' + str(wlen))
    print('learning_rate: ',lr)

    input_shape = (wlen, 1)
    out_dim = class_lay[0]

    model = getModel(input_shape, out_dim)
    optimizer = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-8)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    checkpoints_path = os.path.join(output_folder, 'checkpoints')

    tb = TensorBoard(log_dir=os.path.join(output_folder, 'logs', 'WCNN'))
    lrate = LearningRateScheduler(lr_scheduler, verbose=1)
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(checkpoints_path, 'WCNN.hdf5'),
        verbose=1,
        save_best_only=False,
        save_weights_only=True)

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    callbacks = [tb, checkpointer,lrate]

    if pt_file != 'none':
        model.load_weights(pt_file)

    train_generator = batchGenerator(batch_size, train_data_folder, flac_lst_train, snt_train, wlen, 0.2, out_dim)
    validation_generator = batchGenerator_val(Batch_dev, dev_data_folder, flac_lst_dev, snt_dev, wlen, 0.2, out_dim)
    model.fit_generator(train_generator, steps_per_epoch=N_batches, epochs=N_epochs, verbose=1,
                        validation_data=validation_generator, validation_steps=N_dev_batches, callbacks=callbacks)
