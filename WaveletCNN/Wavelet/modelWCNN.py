
import tensorflow as tf
from keras.layers import Conv1D, LeakyReLU, BatchNormalization, Dense, Flatten, AveragePooling1D
from keras.layers import InputLayer, Input, concatenate, Lambda
from keras.models import Model
import tfwavelets
from conf import *


def res_conv_block(X, in_channels, out_channels, stage, block, dilation=1):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    X_shortcut = X

    X = BatchNormalization(name=bn_name_base + 'a')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = Conv1D(in_channels, 3, padding='valid', use_bias=False, name=conv_name_base + 'a')(X)
    X = BatchNormalization(name=bn_name_base + 'b')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = Conv1D(in_channels, 3, padding='valid', use_bias=False, name=conv_name_base + 'b')(X)
    print(X.shape)
    paddings = tf.constant([[0, 0],  # the batch size dimension
                            [2, 2],  # top and bottom of image
                            [0, 0]])  # the channels dimension
    X = Lambda(lambda x: tf.pad(x, paddings, mode='CONSTANT',
                                constant_values=0.0))(X)
    X = concatenate([X, X_shortcut])
    X = BatchNormalization(name=bn_name_base + 'c')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = Conv1D(out_channels, 3, padding='valid', use_bias=False, dilation_rate=dilation, name=conv_name_base + 'c')(X)

    return X


db2 = tfwavelets.dwtcoeffs.Wavelet(
    tfwavelets.dwtcoeffs.Filter(np.array([-0.12940952255092145,
                                          0.22414386804185735,
                                          0.836516303737469,
                                          0.48296291314469025]), 3),
    tfwavelets.dwtcoeffs.Filter(np.array([-0.48296291314469025,
                                          0.836516303737469,
                                          -0.22414386804185735,
                                          -0.12940952255092145]), 0),
    tfwavelets.dwtcoeffs.Filter(np.array([0.48296291314469025,
                                          0.836516303737469,
                                          0.22414386804185735,
                                          -0.12940952255092145]), 0),
    tfwavelets.dwtcoeffs.Filter(np.array([-0.12940952255092145,
                                          -0.22414386804185735,
                                          0.836516303737469,
                                          -0.48296291314469025]), 3)
)


def getModel(input_shape, out_dim):
    inputs = Input(input_shape)

    conv_i = Conv1D(filters=128, kernel_size=5, strides=2, padding="same")(inputs)
    norm_i = BatchNormalization(name='norm_i')(conv_i)
    relu_i = LeakyReLU(alpha=0.2)(norm_i)

    res_conv_i_1 = res_conv_block(relu_i, 128, 64, 1, 'i', 1)
    res_conv_i_2 = res_conv_block(res_conv_i_1, 64, 64, 2, 'i', 1)
    res_conv_i_3 = res_conv_block(res_conv_i_2, 64, 64, 3, 'i', 1)
    res_conv_i_4 = res_conv_block(res_conv_i_3, 64, 64, 4, 'i', 1)
    res_conv_i_5 = res_conv_block(res_conv_i_4, 64, 64, 5, 'i', 1)
    res_conv_i_6 = res_conv_block(res_conv_i_5, 64, 64, 6, 'i', 1)

    paddings = tf.constant([[0, 0],  # the batch size dimension
                            [6, 6],  # top and bottom of image
                            [0, 0]])  # the channels dimension
    res_conv_i_6 = Lambda(lambda x: tf.pad(x, paddings, mode='CONSTANT',
                                           constant_values=0.0))(res_conv_i_6)

    input_l1 = Lambda(lambda x: tfwavelets.nodes.dwt1d(x, db2, 1))(inputs)
    input_l2 = Lambda(lambda x: tfwavelets.nodes.dwt1d(x, db2, 2))(inputs)
    input_l3 = Lambda(lambda x: tfwavelets.nodes.dwt1d(x, db2, 3))(inputs)
    input_l4 = Lambda(lambda x: tfwavelets.nodes.dwt1d(x, db2, 4))(inputs)
    # level one decomposition starts
    conv_1 = Conv1D(filters=64, kernel_size=3, padding="same")(input_l1)
    norm_1 = BatchNormalization(name='norm_1')(conv_1)
    relu_1 = LeakyReLU(alpha=0.2)(norm_1)

    conv_1_2 = Conv1D(filters=64, kernel_size=3, padding="same")(relu_1)  # strides = 2, padding="same")(relu_1)
    norm_1_2 = BatchNormalization(name='norm_1_2')(conv_1_2)
    relu_1_2 = LeakyReLU(alpha=0.2)(norm_1_2)

    # level two decomposition starts
    conv_a = Conv1D(filters=64, kernel_size=3, padding="same")(input_l2)
    norm_a = BatchNormalization(name='norm_a')(conv_a)
    relu_a = LeakyReLU(alpha=0.2)(norm_a)

    # concate level one and level two decomposition
    concate_level_2 = concatenate([relu_1_2, relu_a, ])
    conv_2 = Conv1D(filters=128, kernel_size=3, padding="same")(concate_level_2)
    norm_2 = BatchNormalization(name='norm_2')(conv_2)
    relu_2 = LeakyReLU(alpha=0.2)(norm_2)

    conv_2_2 = Conv1D(filters=128, kernel_size=3, padding="same")(relu_2)  # strides = 2, padding="same")(relu_2)
    norm_2_2 = BatchNormalization(name='norm_2_2')(conv_2_2)
    relu_2_2 = LeakyReLU(alpha=0.2)(norm_2_2)

    # level three decomposition starts
    conv_b = Conv1D(filters=64, kernel_size=3, padding="same")(input_l3)
    norm_b = BatchNormalization(name='norm_b')(conv_b)
    relu_b = LeakyReLU(alpha=0.2)(norm_b)

    conv_b_2 = Conv1D(filters=128, kernel_size=3, padding="same")(relu_b)
    norm_b_2 = BatchNormalization(name='norm_b_2')(conv_b_2)
    relu_b_2 = LeakyReLU(alpha=0.2)(norm_b_2)

    # concate level two and level three decomposition
    concate_level_3 = concatenate([relu_2_2, relu_b_2])
    conv_3 = Conv1D(filters=256, kernel_size=3, padding="same")(concate_level_3)
    norm_3 = BatchNormalization(name='nomr_3')(conv_3)
    relu_3 = LeakyReLU(alpha=0.2)(norm_3)

    conv_3_2 = Conv1D(filters=256, kernel_size=3, padding="same")(relu_3)  # strides = 2, padding="same")(relu_3)
    norm_3_2 = BatchNormalization(name='norm_3_2')(conv_3_2)
    relu_3_2 = LeakyReLU(alpha=0.2)(norm_3_2)

    # level four decomposition start
    conv_c = Conv1D(filters=64, kernel_size=3, padding="same")(input_l4)
    norm_c = BatchNormalization(name='norm_c')(conv_c)
    relu_c = LeakyReLU(alpha=0.2)(norm_c)

    conv_c_2 = Conv1D(filters=256, kernel_size=3, padding="same")(relu_c)
    norm_c_2 = BatchNormalization(name='norm_c_2')(conv_c_2)
    relu_c_2 = LeakyReLU(alpha=0.2)(norm_c_2)

    conv_c_3 = Conv1D(filters=256, kernel_size=3, padding="same")(relu_c_2)
    norm_c_3 = BatchNormalization(name='norm_c_3')(conv_c_3)
    relu_c_3 = LeakyReLU(alpha=0.2)(norm_c_3)

    # concate level level three and level four decomposition
    concate_level_4 = concatenate([relu_3_2, relu_c_3])
    conv_4 = Conv1D(filters=256, kernel_size=3, padding="same")(concate_level_4)
    norm_4 = BatchNormalization(name='norm_4')(conv_4)
    relu_4 = LeakyReLU(alpha=0.2)(norm_4)

    conv_4_2 = Conv1D(filters=128, kernel_size=3, strides=2, padding="same")(relu_4)
    norm_4_2 = BatchNormalization(name='norm_4_2')(conv_4_2)
    relu_4_2 = LeakyReLU(alpha=0.2)(norm_4_2)

    conv_5_1 = Conv1D(filters=64, kernel_size=3, padding="same")(relu_4_2)
    norm_5_1 = BatchNormalization(name='norm_5_1')(conv_5_1)
    relu_5_1 = LeakyReLU(alpha=0.2)(norm_5_1)

    concat_res = concatenate([relu_5_1, res_conv_i_6])

    res_conv_1 = res_conv_block(relu_5_1, 128, 64, 1, 'a', 4)
    res_conv_2 = res_conv_block(res_conv_1, 64, 32, 2, 'a', 8)
    res_conv_3 = res_conv_block(res_conv_2, 32, 16, 3, 'a', 16)
    res_conv_4 = res_conv_block(res_conv_3, 16, 4, 4, 'a', 32)
    res_conv_5 = res_conv_block(res_conv_4, 4, 2, 5, 'a', 64)

    res_norm = BatchNormalization(name='res_norm')(res_conv_5)
    res_relu = LeakyReLU(alpha=0.2)(res_norm)

    pool_5_1 = AveragePooling1D(pool_size=7, strides=2, padding='same', name='avg_pool_5_1')(res_relu)
    flat_5_1 = Flatten(name='flat_5_1')(pool_5_1)

    output = Dense(out_dim, activation='softmax', name='fc_7')(flat_5_1)
    model = Model(inputs=inputs, outputs=output)
    model.summary()
    return model
