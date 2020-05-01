# from tensorflow import set_random_seed
from keras import models, layers
import numpy as np
import sincnet
# from keras.layers import Dense, Dropout, Activation
from keras.layers import MaxPooling1D, Conv1D, LeakyReLU, BatchNormalization, Dense, Flatten
from keras.layers import InputLayer, Input, Lambda
from keras.models import Model
import tensorflow as tf
import tfwavelets
from conf import *


def log_softmax(x):
    return x - tf.log(tf.reduce_sum(tf.exp(x), -1, keep_dims=True))


def getModel(input_shape, out_dim):
    #
    inputs = Input(input_shape)
    x = sincnet.SincConv1D(cnn_N_filt[0], cnn_len_filt[0], fs)(inputs)

    x = MaxPooling1D(pool_size=cnn_max_pool_len[0])(x)
    if cnn_use_batchnorm[0]:
        x = BatchNormalization(momentum=0.05)(x)
    if cnn_use_laynorm[0]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv1D(cnn_N_filt[1], cnn_len_filt[1], strides=1, padding='valid')(x)
    x = MaxPooling1D(pool_size=cnn_max_pool_len[1])(x)
    if cnn_use_batchnorm[1]:
        x = BatchNormalization(momentum=0.05)(x)
    if cnn_use_laynorm[1]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv1D(cnn_N_filt[2], cnn_len_filt[2], strides=1, padding='valid')(x)
    x = MaxPooling1D(pool_size=cnn_max_pool_len[2])(x)
    if cnn_use_batchnorm[2]:
        x = BatchNormalization(momentum=0.05)(x)
    if cnn_use_laynorm[2]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)

    # DNN
    x = Dense(fc_lay[0])(x)
    if fc_use_batchnorm[0]:
        x = BatchNormalization(momentum=0.05, epsilon=1e-5)(x)
    if fc_use_laynorm[0]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(fc_lay[1])(x)
    if fc_use_batchnorm[1]:
        x = BatchNormalization(momentum=0.05, epsilon=1e-5)(x)
    if fc_use_laynorm[1]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(fc_lay[2])(x)
    if fc_use_batchnorm[2]:
        x = BatchNormalization(momentum=0.05, epsilon=1e-5)(x)
    if fc_use_laynorm[2]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # DNN final
    x = layers.Dense(out_dim)(x)
    # prediction = Lambda(lambda x: log_softmax(x))(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model

db4 = tfwavelets.dwtcoeffs.Wavelet(
        tfwavelets.dwtcoeffs.Filter(np.array([-0.010597401784997278,
                                                       0.032883011666982945,
                                                       0.030841381835986965,
                                                       -0.18703481171888114,
                                                       -0.02798376941698385,
                                                       0.6308807679295904,
                                                       0.7148465705525415,
                                                       0.23037781330885523]), 7),
        tfwavelets.dwtcoeffs.Filter(np.array([-0.23037781330885523,
                                                       0.7148465705525415,
                                                       -0.6308807679295904,
                                                       -0.02798376941698385,
                                                       0.18703481171888114,
                                                       0.030841381835986965,
                                                       -0.032883011666982945,
                                                       -0.010597401784997278]), 0),
        tfwavelets.dwtcoeffs.Filter(np.array([0.23037781330885523,
                                                       0.7148465705525415,
                                                       0.6308807679295904,
                                                       -0.02798376941698385,
                                                       -0.18703481171888114,
                                                       0.030841381835986965,
                                                       0.032883011666982945,
                                                       -0.010597401784997278]), 0),
        tfwavelets.dwtcoeffs.Filter(np.array([-0.010597401784997278,
                                                       -0.032883011666982945,
                                                       0.030841381835986965,
                                                       0.18703481171888114,
                                                       -0.02798376941698385,
                                                       -0.6308807679295904,
                                                       0.7148465705525415,
                                                       -0.23037781330885523]), 7)

)

trainable_wavelet = tfwavelets.dwtcoeffs.TrainableWavelet(db4)


def getModel_wavelets(input_shape, out_dim):
    #
    inputs = Input(input_shape)
    x = Lambda(lambda x: tfwavelets.nodes.dwt1d(x, trainable_wavelet, 4))(inputs)
    # print(x.shape)
    # x = MaxPooling1D(pool_size=cnn_max_pool_len[0])(x)
    # if cnn_use_batchnorm[0]:
    #     x = BatchNormalization(momentum=0.05)(x)
    # if cnn_use_laynorm[0]:
    #     x = sincnet.LayerNorm()(x)
    # x = LeakyReLU(alpha=0.2)(x)

    x = Conv1D(cnn_N_filt[1], cnn_len_filt[1], strides=1, padding='valid')(x)
    x = MaxPooling1D(pool_size=cnn_max_pool_len[1])(x)
    if cnn_use_batchnorm[1]:
        x = BatchNormalization(momentum=0.05)(x)
    if cnn_use_laynorm[1]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv1D(cnn_N_filt[2], cnn_len_filt[2], strides=1, padding='valid')(x)
    x = MaxPooling1D(pool_size=cnn_max_pool_len[2])(x)
    if cnn_use_batchnorm[2]:
        x = BatchNormalization(momentum=0.05)(x)
    if cnn_use_laynorm[2]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)

    # DNN
    x = Dense(fc_lay[0])(x)
    if fc_use_batchnorm[0]:
        x = BatchNormalization(momentum=0.05, epsilon=1e-5)(x)
    if fc_use_laynorm[0]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(fc_lay[1])(x)
    if fc_use_batchnorm[1]:
        x = BatchNormalization(momentum=0.05, epsilon=1e-5)(x)
    if fc_use_laynorm[1]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(fc_lay[2])(x)
    if fc_use_batchnorm[2]:
        x = BatchNormalization(momentum=0.05, epsilon=1e-5)(x)
    if fc_use_laynorm[2]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # DNN final
    x = layers.Dense(out_dim)(x)
    # prediction = Lambda(lambda x: log_softmax(x))(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model
