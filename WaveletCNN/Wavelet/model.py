"""
            File to define Sincnet the model for training

            Functions:
            get_model(input_shape,out_dim): Defines the model

"""


from keras import models, layers
import sincnet
from keras.layers import MaxPooling1D, Conv1D, LeakyReLU, BatchNormalization, Dense, Flatten
from keras.layers import InputLayer, Input, Lambda
from keras.models import Model
from conf import *
import tensorflow as tf

def log_softmax(x):
    return x - tf.log(tf.reduce_sum(tf.exp(x), -1, keep_dims=True))


def getModel(input_shape, out_dim):
    """
                Function defines the Sincnet model

                Parameters:
                input_shape : input dimension of the chunk of audio to train on
                out_dim (int) : output_dimension for the labels used to make the labels categorical

                Returns:
                model : returns the model to train
                """

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

    #DNN
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

    #DNN final
    prediction = layers.Dense(out_dim,activation='softmax')(x)
#     prediction =  Lambda(lambda x: log_softmax(x))(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.summary()
    return model