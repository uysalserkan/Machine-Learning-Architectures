import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, ReLU, Add, Input, \
    AveragePooling2D, GlobalAveragePooling2D, Reshape, UpSampling2D, Concatenate, Activation, Flatten
from tensorflow.keras.models import Model


def Conv_block(input_layer, filters):
    '''
      Açıklama park olacak, PARK ETME.
    '''

    KERNEL_1 = (1, 1)
    KERNEL_2 = (3, 3)
    DILATION_2 = (2, 2)
    PADDING = 'same'
    K_INIT = 'he_normal'
    ALPHA = 0.2
    f1, f2, f3 = filters

    # Block1
    x = Conv2D(
        filters=f1,
        kernel_size=KERNEL_1,
        padding=PADDING,
        kernel_initializer=K_INIT
    )(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=ALPHA)(x)

    # Block2
    x = Conv2D(
        filters=f2,
        kernel_initializer=K_INIT,
        kernel_size=KERNEL_2,
        dilation_rate=DILATION_2,
        padding=PADDING
    )(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=ALPHA)(x)

    # Block3
    x = Conv2D(
        filters=f3,
        kernel_size=KERNEL_1,
        padding=PADDING,
        kernel_initializer=K_INIT
    )(x)
    x = BatchNormalization()(x)

    # primitive block
    prim = Conv2D(
        filters=f3,
        kernel_size=KERNEL_2,
        kernel_initializer=K_INIT,
        padding=PADDING
    )(input_layer)
    prim = BatchNormalization()(prim)

    # adding section
    x = Add()([x, prim])
    x = ReLU()(x)

    return x


def Generate_base_features(input_layer):
    '''
      Açıklama park olacak, PARK ETME.
    '''

    base = Conv_block(input_layer=input_layer, filters=[32, 32, 64])
    base = Conv_block(input_layer=base, filters=[64, 64, 128])
    base = Conv_block(input_layer=base, filters=[128, 128, 256])

    return base


def Pyramid_feature_maps(input_layer):
    FILTER_SIZE = 64
    KERNEL_SIZE = (1, 1)
    YELLOW_POOL = (2, 2)
    BLUE_POOL = (4, 4)
    GREEN_POOL = (8, 8)
    INTERPOLATION = 'bilinear'

    red_block = GlobalAveragePooling2D()(input_layer)
    red_block = Reshape(target_shape=(1, 1, 256))(red_block)
    red_block = Conv2D(
        filters=FILTER_SIZE,
        kernel_size=KERNEL_SIZE,
    )(red_block)
    red_block = UpSampling2D(size=256, interpolation=INTERPOLATION)(red_block)

    yellow_block = AveragePooling2D(pool_size=(YELLOW_POOL))(input_layer)
    yellow_block = Conv2D(
        filters=FILTER_SIZE,
        kernel_size=KERNEL_SIZE
    )(yellow_block)
    yellow_block = UpSampling2D(
        size=2, interpolation=INTERPOLATION)(yellow_block)

    blue_block = AveragePooling2D(pool_size=BLUE_POOL)(input_layer)
    blue_block = Conv2D(
        filters=FILTER_SIZE,
        kernel_size=KERNEL_SIZE
    )(blue_block)
    blue_block = UpSampling2D(size=4, interpolation=INTERPOLATION)(blue_block)

    green_block = AveragePooling2D(pool_size=GREEN_POOL)(input_layer)
    green_block = Conv2D(
        filters=FILTER_SIZE,
        kernel_size=KERNEL_SIZE
    )(green_block)
    green_block = UpSampling2D(
        size=8, interpolation=INTERPOLATION)(green_block)

    return tf.keras.layers.concatenate([input_layer, red_block, yellow_block, blue_block, green_block])


def Last_conv_layer(input_layer):
    x = Conv2D(filters=3, kernel_size=3, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    return Flatten()(x)


def PSPNet(img_height=256, img_width=256, channel=3):
    input_layer = Input(shape=(img_height, img_width, channel))
    base = Generate_base_features(input_layer=input_layer)
    pyramid_layer = Pyramid_feature_maps(input_layer=base)
    output = Last_conv_layer(input_layer=pyramid_layer)

    return Model(input_layer, output)


model = PSPNet()
model.summary()
