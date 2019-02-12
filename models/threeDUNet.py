from __future__ import division

from tensorflow import keras

Model = keras.models.Model
Input = keras.layers.Input
Conv3D = keras.layers.Conv3D
MaxPooling3D = keras.layers.MaxPooling3D
UpSampling3D = keras.layers.UpSampling3D
Cropping3D = keras.layers.Cropping3D
ZeroPadding3D = keras.layers.ZeroPadding3D
BatchNormalization = keras.layers.BatchNormalization
Activation = keras.layers.Activation
Adam = keras.optimizers.Adam
concatenate = keras.layers.concatenate
ModelCheckpoint = keras.callbacks.ModelCheckpoint
K = keras.backend
losses = keras.losses

K.set_image_data_format('channels_last')


def get_crop_shape(target, refer):
    # depth, the 1st dimension
    cd = (target.get_shape()[0] - refer.get_shape()[0]).value
    assert (cd >= 0)
    if cd % 2 != 0:
        cd1, cd2 = int(cd / 2), int(cd / 2) + 1
    else:
        cd1, cd2 = int(cd / 2), int(cd / 2)
    # width, the 3rd dimension
    cw = (target.get_shape()[3] - refer.get_shape()[3]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)
    return (cd1, cd2), (ch1, ch2), (cw1, cw2)


def conv_3D_bn_relu(nd, filter_conv, inputs=None):
    conv = Conv3D(nd, filter_conv, padding='same')(inputs)  # , kernel_initializer='he_normal'
    # bn = BatchNormalization()(conv)
    relu = Activation('relu')(conv)
    return relu


def get_3Dunet(img_shape, num_classes):
    inputs = Input(shape=img_shape)
    concat_axis = -1
    filter_conv_1 = [3, 3, 3]
    filter_conv_2 = [1, 1, 1]
    filter_mp_1 = [2, 2, 2]
    filter_mp_2 = [1, 2, 2]

    conv1 = conv_3D_bn_relu(64, filter_conv_1, inputs)
    conv1 = conv_3D_bn_relu(64, filter_conv_1, conv1)
    conv1 = conv_3D_bn_relu(64, filter_conv_1, conv1)
    trans_conv1 = conv_3D_bn_relu(64, filter_conv_2, conv1)

    pool1 = MaxPooling3D(pool_size=filter_mp_1)(conv1)
    conv2 = conv_3D_bn_relu(128, filter_conv_1, pool1)
    conv2 = conv_3D_bn_relu(128, filter_conv_1, conv2)
    trans_conv2 = conv_3D_bn_relu(32, filter_conv_2, conv2)

    pool2 = MaxPooling3D(pool_size=filter_mp_2)(conv2)
    conv3 = conv_3D_bn_relu(256, filter_conv_1, pool2)
    conv3 = conv_3D_bn_relu(256, filter_conv_1, conv3)
    trans_conv3 = conv_3D_bn_relu(32, filter_conv_2, conv3)

    pool3 = MaxPooling3D(pool_size=filter_mp_1)(conv3)
    conv4 = conv_3D_bn_relu(384, filter_conv_1, pool3)
    conv4 = conv_3D_bn_relu(384, filter_conv_1, conv4)

    up_conv4 = UpSampling3D(size=filter_mp_1)(conv4)
    up5 = concatenate([up_conv4, trans_conv3], axis=concat_axis)
    conv5 = conv_3D_bn_relu(256, filter_conv_1, up5)
    conv5 = conv_3D_bn_relu(256, filter_conv_1, conv5)
    up_conv5 = UpSampling3D(size=filter_mp_2)(conv5)

    up5 = concatenate([up_conv5, trans_conv2], axis=concat_axis)
    conv6 = conv_3D_bn_relu(128, filter_conv_1, up5)
    conv6 = conv_3D_bn_relu(128, filter_conv_1, conv6)
    up_conv6 = UpSampling3D(size=filter_mp_1)(conv6)

    up6 = concatenate([up_conv6, trans_conv1], axis=concat_axis)
    conv7 = conv_3D_bn_relu(64, filter_conv_1, up6)
    conv7 = conv_3D_bn_relu(64, filter_conv_1, conv7)

    conv8 = Conv3D(num_classes, 1, activation='sigmoid', padding='same')(conv7)  # , kernel_initializer='he_normal'
    model = Model(inputs=inputs, outputs=conv8)
    return model
