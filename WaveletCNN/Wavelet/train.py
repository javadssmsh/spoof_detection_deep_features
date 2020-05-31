import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";
import soundfile as sf
from conf import *
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.callbacks import Callback

import numpy as np
import tensorflow as tf
from keras.optimizers import RMSprop
from keras import backend as K
import keras

K.clear_session()
from model import getModel_wavelets
import librosa as lb
from data_io import batchGenerator, batchGenerator_val
from keras.callbacks import LearningRateScheduler
from keras.metrics import categorical_accuracy as accuracy
from test import Validation


def lr_scheduler(epoch, lr):
    if epoch in [0,4,9]:
        decay_rate = 1
        return lr * decay_rate
    elif epoch in [1,6]:
        decay_rate = 10
        return lr * decay_rate
    elif epoch in [2,7]:
        decay_rate = 10
        return lr * decay_rate
    elif epoch in [3,8,5]:
        decay_rate = 0.1
        return lr * decay_rate
    else:
        decay_rate = 1
        return lr * decay_rate
    return lr


def softmax_cross_entropy_logits(targets, logs):
    return tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logs)

def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed

class ValidationCallback(Callback):
    def __init__(self, Batch_dev, data_folder, wav_lst_te, wlen, wshift, class_lay):
        self.wav_lst_te = wav_lst_te
        self.data_folder = data_folder
        self.wlen = wlen
        self.wshift = wshift
        self.Batch_dev = Batch_dev
        self.class_lay = class_lay

    def on_epoch_end(self, epoch, logs={}):
        val = Validation(self.Batch_dev, self.data_folder, self.wav_lst_te, self.wlen, self.wshift, self.class_lay,
                         self.model)
        val.validate(epoch)


if __name__ == "__main__":

    print('N_filt ' + str(cnn_N_filt))
    print('N_filt len ' + str(cnn_len_filt))
    print('FS ' + str(fs))
    print('WLEN ' + str(wlen))
    print('learning_rate: ', lr)

    input_shape = (wlen, 1)
    out_dim = class_lay[0]

    model = getModel_wavelets(input_shape, out_dim)

    optimizer = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-8)
    model.compile(loss=softmax_cross_entropy_logits, optimizer=optimizer, metrics=['accuracy'])

    checkpoints_path = os.path.join(output_folder, 'checkpoints')

    tb = TensorBoard(log_dir=os.path.join(output_folder, 'logs', 'Wavelet_filtersize_32'))
    lrate = LearningRateScheduler(lr_scheduler, verbose=1)
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(checkpoints_path, 'Wavelet_32.hdf5'),
        verbose=1,
        save_best_only=False,
        save_weights_only=True)

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    validation = ValidationCallback(Batch_dev, dev_data_folder, flac_lst_dev, wlen, wshift, class_lay)
    callbacks = [tb, checkpointer, lrate]

    if pt_file != 'none':
        model.load_weights(pt_file)

    train_generator = batchGenerator(batch_size, train_data_folder, flac_lst_train, snt_train, wlen, out_dim)
    validation_generator = batchGenerator_val(Batch_dev, dev_data_folder, flac_lst_dev, snt_dev, wlen, out_dim)
    model.fit_generator(train_generator, steps_per_epoch=N_batches, epochs=N_epochs, verbose=1,
                        validation_data=validation_generator, validation_steps=N_dev_batches, callbacks=callbacks)
