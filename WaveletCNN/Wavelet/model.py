# from tensorflow import set_random_seed
from keras import models, layers
import numpy as np
import sincnet
# from keras.layers import Dense, Dropout, Activation
from keras.layers import MaxPooling1D, Conv1D, LeakyReLU, BatchNormalization, Dense, Flatten
from keras.layers import InputLayer, Input, Lambda, concatenate
from keras.models import Model
from kymatio.keras import Scattering1D
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
        tfwavelets.dwtcoeffs.Filter(np.array([0.0,
                                             0.0,
                                             0.0,
                                             0.0,
                                             0.014426282505624435,
                                             0.014467504896790148,
                                             -0.07872200106262882,
                                             -0.04036797903033992,
                                             0.41784910915027457,
                                             0.7589077294536541,
                                             0.41784910915027457,
                                             -0.04036797903033992,
                                             -0.07872200106262882,
                                             0.014467504896790148,
                                             0.014426282505624435,
                                             0.0,
                                             0.0,
                                             0.0]), 17),
        tfwavelets.dwtcoeffs.Filter(np.array([-0.0019088317364812906,
                                               -0.0019142861290887667,
                                               0.016990639867602342,
                                               0.01193456527972926,
                                               -0.04973290349094079,
                                               -0.07726317316720414,
                                               0.09405920349573646,
                                               0.4207962846098268,
                                               -0.8259229974584023,
                                               0.4207962846098268,
                                               0.09405920349573646,
                                               -0.07726317316720414,
                                               -0.04973290349094079,
                                               0.01193456527972926,
                                               0.016990639867602342,
                                               -0.0019142861290887667,
                                               -0.0019088317364812906,
                                               0.0]), 0),
        tfwavelets.dwtcoeffs.Filter(np.array([0.0019088317364812906,
                                               -0.0019142861290887667,
                                               -0.016990639867602342,
                                               0.01193456527972926,
                                               0.04973290349094079,
                                               -0.07726317316720414,
                                               -0.09405920349573646,
                                               0.4207962846098268,
                                               0.8259229974584023,
                                               0.4207962846098268,
                                               -0.09405920349573646,
                                               -0.07726317316720414,
                                               0.04973290349094079,
                                               0.01193456527972926,
                                               -0.016990639867602342,
                                               -0.0019142861290887667,
                                               0.0019088317364812906,
                                               0.0]), 0),
        tfwavelets.dwtcoeffs.Filter(np.array([0.0,
                                               0.0,
                                               0.0,
                                               0.0,
                                               0.014426282505624435,
                                               -0.014467504896790148,
                                               -0.07872200106262882,
                                               0.04036797903033992,
                                               0.41784910915027457,
                                               -0.7589077294536541,
                                               0.41784910915027457,
                                               0.04036797903033992,
                                               -0.07872200106262882,
                                               -0.014467504896790148,
                                               0.014426282505624435,
                                               0.0,
                                               0.0,
                                               0.0]), 7)

)

trainable_wavelet = tfwavelets.dwtcoeffs.TrainableWavelet(db4)


def getModel_wavelets(input_shape, out_dim):
    #
    inputs = Input(input_shape)
    x1 = Lambda(lambda x: tfwavelets.nodes.dwt1d(x, trainable_wavelet, 1))(inputs)
    x2 = Lambda(lambda x: tfwavelets.nodes.dwt1d(x, trainable_wavelet, 2))(inputs)
    x3 = Lambda(lambda x: tfwavelets.nodes.dwt1d(x, trainable_wavelet, 3))(inputs)
    x4 = Lambda(lambda x: tfwavelets.nodes.dwt1d(x, trainable_wavelet, 4))(inputs)
    x5 = Lambda(lambda x: tfwavelets.nodes.dwt1d(x, trainable_wavelet, 5))(inputs)
    x6 = Lambda(lambda x: tfwavelets.nodes.dwt1d(x, trainable_wavelet, 6))(inputs)
    x = concatenate([x1,x2,x3,x4,x5,x6])
    # print(x.shape)
#     x = MaxPooling1D(pool_size=cnn_max_pool_len[0])(x)
#     if cnn_use_batchnorm[0]:
#         x = BatchNormalization(momentum=0.05)(x)
#     if cnn_use_laynorm[0]:
#         x = sincnet.LayerNorm()(x)
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
    x = layers.Dense(out_dim,activation='softmax')(x)
#     prediction = Lambda(lambda x: log_softmax(x))(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model


