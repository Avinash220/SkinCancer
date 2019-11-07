import os
import tensorflow as tf
#  import keras
import tensorflow.keras as keras
from tensorflow.keras import Sequential, layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def channel_split(x, num_splits=2):
    """split the input tensor into two equal dimension

    Args:
        x: the input tensor
        num_splits: the numbers of tensors after splitting

    """
    if num_splits == 2:
        return tf.split(x, axis=3, num_or_size_splits=num_splits)
    else:
        raise ValueError('The num_splits is 2')


def channel_shuffle(x, groups):
    """channel shuffle operation.

    Args:
        x: the input tensor
        groups: input branch number

    """
    _, height, width, channels = x.shape
    channels_per_group = channels // groups
    x = tf.reshape(x, [-1, height, width, groups, channels_per_group])
    x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, height, width, channels])
    return x


class SELayer(keras.Model):
    """this is the implement of SE unit."""
    def __init__(self, out_channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc = Sequential([
            layers.Dense(out_channels//reduction),
            layers.Activation('relu'),
            layers.Dense(out_channels),
            layers.Activation('sigmoid')
        ])

    def call(self, inputs, training=None):
        _, _, _, c = inputs.shape
        out = self.avg_pool(inputs)
        out = self.fc(out)
        out = tf.reshape(out, [-1, 1, 1, c])
        return inputs * out


class ShuffleNetUnit(keras.Model):
    """this is the implement of shufflenet v2 unit including stride=1 and 2."""
    def __init__(self, out_channels, stride=1, se=False):
        super(ShuffleNetUnit, self).__init__()
        self.stride = stride
        self.se = se
        self.out_channels = out_channels//2
        self.residual = Sequential([
            layers.Conv2D(self.out_channels, (1, 1), use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.DepthwiseConv2D((3, 3), strides=self.stride, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Conv2D(self.out_channels, (1, 1), use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        if stride == 1:
            self.short_cut = Sequential()
        else:
            self.short_cut = Sequential([
                layers.DepthwiseConv2D((3, 3), strides=self.stride, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.Conv2D(self.out_channels, (1, 1), use_bias=False),
                layers.BatchNormalization(),
                layers.Activation('relu')
            ])
        if self.se:
            self.se_layer = SELayer(self.out_channels)

    def call(self, inputs, training=None):
        if self.stride == 1:
            residual, short_cut = channel_split(inputs)
        else:
            residual, short_cut = inputs, inputs
        residual = self.residual(residual)
        short_cut = self.short_cut(short_cut)
        if self.se:
            residual = self.se_layer(residual)
        out = layers.concatenate([residual, short_cut], axis=-1)
        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2(keras.Model):
    """ShuffleNet v2 implement."""
    def __init__(self, scale, se=False, num_classes=1000):
        super(ShuffleNetV2, self).__init__()
        self.se = se
        if scale == 0.5:
            out_channels = [48, 96, 192, 1024]
        elif scale == 1:
            out_channels = [116, 232, 464, 1024]
        elif scale == 1.5:
            out_channels = [176, 352, 704, 1024]
        elif scale == 2:
            out_channels = [244, 488, 976, 2048]
        else:
            raise ValueError('The value of scale must be of [0.5, 1, 1.5, 2]')
        self.conv1 = Sequential([
            layers.Conv2D(24, (3, 3), strides=2, padding='same', use_bias=False),
            layers.BatchNormalization()
        ])
        self.max_pool = layers.MaxPool2D((3, 3), strides=2, padding='same')
        self.stage2 = self._make_stage(3, out_channels[0])
        self.stage3 = self._make_stage(7, out_channels[1])
        self.stage4 = self._make_stage(3, out_channels[2])
        self.conv5 = Sequential([
            layers.Conv2D(out_channels[3], (1, 1), use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.max_pool(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.conv5(out)
        out = self.avg_pool(out)
        out = self.fc(out)
        return out

    def _make_stage(self, num_stages, out_channels):
        layers = []
        layers.append(ShuffleNetUnit(out_channels, stride=2, se=self.se))
        for i in range(num_stages):
            layers.append(ShuffleNetUnit(out_channels, stride=1, se=self.se))
        return Sequential(layers)


