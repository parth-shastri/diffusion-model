import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
import config


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = tf.keras.layers.Conv2D(width, kernel_size=1)(x)
        x = tf.keras.layers.BatchNormalization(center=False, scale=False)(x)
        x = tf.keras.layers.Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = tf.keras.layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = tf.keras.layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = tf.keras.layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(PositionalEmbedding, self).__init__()
        self.dim = dim

    def __call__(self, position, *args, **kwargs):
        half_dim = tf.cast(self.dim // 2, tf.float32)
        embeddings = tf.math.log(10000.) / (half_dim - 1)
        embeddings = tf.exp(tf.range(half_dim) * -embeddings)
        embeddings = position[:, None] * embeddings[None, :]
        embeddings = tf.concat([tf.sin(embeddings), tf.cos(embeddings)], axis=-1)
        return embeddings


# generate upsample block and downsample block
# Make a UNet class and init these blocks into layer lists
# while building the Unet keep the track of the output of each downsample block in a list.
# while upsampling, reverse this list and concat the output of each upsampled layer with the tracked list.

# take an image before passing it to the UNet add positional embeddings to it
# use the PositionalEmbeddings class
# the embedded out => (embed_dim, embed_dim) is passed on to the UNet

class UNet(tf.keras.Model):
    def __init__(self, dim, num_channels, num_channels_per_layer, block_depth):
        super(UNet, self).__init__()
        self.dim = dim
        self.widths = num_channels_per_layer
        self.num_channels = num_channels

        self.downsample_blocks = []
        for width in self.widths[:-1]:
            self.downsample_blocks.append(DownBlock(width, block_depth))

        self.upsample_blocks = []
        for width in reversed(self.widths[:-1]):
            self.upsample_blocks.append(UpBlock(width, block_depth))

        self.bottle_necks = []
        for _ in range(block_depth):
            self.bottle_necks.append(ResidualBlock(self.widths[-1]))

        self.time_mlp = tf.keras.Sequential([
            PositionalEmbedding(self.dim),
            layers.UpSampling2D(size=config.IMAGE_SIZE, interpolation="nearest")
        ])
        self.input_conv = layers.Conv2D(self.widths[0], kernel_size=1)
        self.last_layer = tf.keras.layers.Conv2D(self.num_channels, kernel_size=1, kernel_initializer="zeros")

    def model(self):
        ims = tf.keras.Input(shape=(64, 64, 3))
        t = tf.keras.Input(shape=(1, 1))
        out = self([ims, t])
        return tf.keras.Model(inputs=[ims, t], outputs=out)

    def __call__(self, inputs, *args, **kwargs):
        skips = []
        x, time = inputs
        e = self.time_mlp(time)
        x = self.input_conv(x)
        x = layers.Concatenate()([x, e])

        for layer in self.downsample_blocks:
            x = layer([x, skips])

        # skips = reversed(skips[:-1])

        for bottle_neck in self.bottle_necks:
            x = bottle_neck(x)

        for layer in self.upsample_blocks:
            x = layer([x, skips])
            # x = tf.keras.layers.Concatenate()([x, skip])

        x = self.last_layer(x)
        return x


if __name__ == "__main__":
    model = UNet(config.EMBEDDING_DIM, 3, config.WIDTHS, config.BLOCK_DEPTH)
    print(len(model.upsample_blocks), len(model.downsample_blocks))
    out = model([tf.keras.Input(shape=(64, 64, 3)), tf.keras.Input(shape=(1, 1))])
    print(model.model().summary())





