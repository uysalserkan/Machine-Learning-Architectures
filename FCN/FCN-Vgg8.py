import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation, BatchNormalization, Add
from tensorflow.keras.applications.vgg16 import VGG16


def load_vgg(input_shape, class_size):
    '''
      It loads weights='imagenet'.
    '''

    return VGG16(include_top=False, weights='imagenet', input_shape=input_shape, classes=class_size)


def Decoder(model, class_size):
    '''
      buraya açıklama park edilecek.
    '''

    FILTER_SIZE = 4096
    T_KERNEL_SIZE = (4, 4)
    PADDING = 'same'
    STRIDE = 2

    [p3, p4, p5] = [model.get_layer('block3_pool').output,
                    model.get_layer('block4_pool').output,
                    model.get_layer('block5_pool').output
                    ]

    x = Conv2D(
        filters=FILTER_SIZE,
        kernel_size=(7, 7),
        padding=PADDING
    )(p5)

    x = Conv2D(
        filters=FILTER_SIZE,
        kernel_size=(1, 1),
        padding=PADDING
    )(x)

    x = Conv2D(
        filters=class_size,
        kernel_size=(1, 1),
        padding=PADDING
    )(x)

    x = Conv2DTranspose(
        filters=class_size,
        kernel_size=T_KERNEL_SIZE,
        strides=STRIDE,
        padding=PADDING
    )(x)

    x = BatchNormalization(axis=1)(x)

    x2 = Conv2D(
        filters=class_size,
        kernel_size=(1, 1),
        padding=PADDING
    )(p4)

    x = Add()([x, x2])

    x = Conv2DTranspose(
        filters=class_size,
        kernel_size=T_KERNEL_SIZE,
        strides=STRIDE,
        padding=PADDING
    )(x)

    x = BatchNormalization(axis=1)(x)

    x2 = Conv2D(
        filters=class_size,
        kernel_size=(1, 1),
        padding=PADDING
    )(p3)

    x = Add()([x, x2])

    x = Conv2DTranspose(
        filters=class_size,
        kernel_size=(16, 16),
        strides=8,
        padding=PADDING
    )(x)

    x = Activation('softmax')(x)

    return x


def FCN8(input_shape, class_size):
    vgg_model = load_vgg(input_shape=input_shape, class_size=class_size)
    input_layer = vgg_model.get_layer(name='input_1').output

    output_layer = Decoder(model=vgg_model, class_size=10)
    return Model(input_layer, output_layer)


model = FCN8(input_shape=(224, 224, 3), class_size=10)
model.summary()
