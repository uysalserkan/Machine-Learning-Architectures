import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Input, BatchNormalization, ReLU, Conv2DTranspose, Cropping2D, Add, Activation


def encoder_block(input_layer, kernels, filters, pad, pool_size):
    """
    Buraya açıklama gelecek. Park etmeyiniz..
    """
    x = ZeroPadding2D(padding=pad)(input_layer)
    # ? same -> valid olabilir.
    x = Conv2D(filters=filters, kernel_size=kernels, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)

    return x


def Encoder(img_height=224, img_width=224, channels=3):
    """
    Açıklama buraya gelecek.
    """

    # Local function variables
    KERNEL = 3
    FILTER_SIZES = [64, 128, 256, 256, 256]
    PADDING = 1
    POOL_SIZE = 2

    img_input = Input(shape=(img_height, img_width, channels))
    encoded_levels = []

    x = img_input

    # F1, F2, F3, F4, F5 blocks.
    for FILTER_SIZE in FILTER_SIZES:
        x = encoder_block(
            input_layer=img_input,
            filters=FILTER_SIZE,
            kernels=(KERNEL, KERNEL),
            pool_size=(POOL_SIZE, POOL_SIZE),
            pad=(PADDING, PADDING),
        )
        encoded_levels.append(x)

    return img_input, encoded_levels


def Decoder(encoder_levels, class_size=12):
    """
    Buraya açıklama gelecek.
    """
    T_KERNEL_SIZE = (4, 4)
    T_STRIDES = (2, 2)
    CROP_SIZE = (1, 1)
    KERNEL_SIZE = (1, 1)

    [f1, f2, f3, f4, f5] = encoder_levels

    # Upsample-1
    o = Conv2DTranspose(filters=class_size,
                        kernel_size=T_KERNEL_SIZE, strides=T_STRIDES)(f5)
    o = Cropping2D(cropping=CROP_SIZE)(o)

    # load pool
    o2 = f4
    o2 = Conv2D(filters=class_size, kernel_size=KERNEL_SIZE,
                activation='relu', padding='same')(o2)

    # adding 2 layers
    o = Add()([o, o2])

    # Upsample-2
    o = Conv2DTranspose(filters=class_size,
                        kernel_size=T_KERNEL_SIZE, strides=T_STRIDES)(o)
    o = Cropping2D(cropping=CROP_SIZE)(o)

    # load pool
    o2 = f3
    o2 = Conv2D(filters=class_size, kernel_size=KERNEL_SIZE,
                activation='relu', padding='same')(o2)

    # adding 2 layers
    o = Add()([o, o2])

    o = Conv2DTranspose(filters=class_size,
                        kernel_size=(8, 8), strides=(8, 8))(o)

    o = Activation('softmax')(o)

    return o


def FCN8(class_size, img_height=224, img_width=224, channels=3):
    img_input, encoded_levels = Encoder(
        img_height=img_height, img_width=img_width, channels=channels)
    output = Decoder(encoder_levels=encoded_levels, class_size=12)
    model = Model(img_input, output)

    return model


model = FCN8(12)
