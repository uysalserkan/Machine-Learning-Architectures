import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, GlobalAveragePooling2D, Dense, Input, Activation
from tensorflow.keras.models import Model


def encoder_block(input_layer, kernels, filters, pool_size, conv_iter, pad=1):
    '''
        Params:
          input_layer: Required
          kernels: Required
          filters: Required
          pad: Required
          pool_size: Required.

        Return:
          A keras layer.
    '''

    x = input_layer
    # x = ZeroPadding2D(padding=pad)(input_layer)
    for _ in range(conv_iter):
        x = Conv2D(filters=filters, kernel_size=kernels, padding='same')(x)

    x = MaxPooling2D(pool_size=pool_size)(x)

    return x


def Encoder(input_layer, kernels, pool_size, pad=1):
    '''
        Params:
          input_layer: Required
          kernels: Required
          filters: Required
          pad: Required
          pool_size: Required.

        Return:
          A keras layer.
    '''

    FILTERS = [64, 128, 256, 512, 512]
    COV_ITERS = [2, 2, 3, 3, 3]

    x = input_layer
    for filters, conv_iter in zip(FILTERS, COV_ITERS):
        x = encoder_block(kernels=kernels, conv_iter=conv_iter,
                          input_layer=x, filters=filters, pool_size=pool_size)

    return x


def Decoder(encoded_layer, class_size):
    FILTER_SIZE = 128
    o = Conv2D(filters=FILTER_SIZE, kernel_size=(
        7, 7), padding='same')(encoded_layer)
    o = Conv2D(filters=FILTER_SIZE, kernel_size=(1, 1), padding='same')(o)
    o = Conv2D(filters=class_size, kernel_size=(1, 1), padding='same')(o)
    o = Conv2D(filters=class_size, kernel_size=(1, 1), padding='same')(o)

    return Activation('softmax')(o)


def FCN32(img_height=224, img_width=224, channels=3):
    input_lay = Input(shape=(img_height, img_width, channels))
    pooled = Encoder(input_layer=input_lay, kernels=(3, 3), pool_size=(2, 2))
    output = Decoder(encoded_layer=pooled, class_size=12)
    return Model(input_lay, output)


model = FCN32()
model.summary()
