import os
import config
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer


class ResidualBlock(Model):
    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({"width": self.width})
        return config

    def __init__(self, width, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.width = width
        self.res = tf.keras.layers.Conv2D(width, kernel_size=1)
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.conv1 = tf.keras.layers.Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )
        self.conv2 = tf.keras.layers.Conv2D(width, kernel_size=3, padding="same")

    def call(self, inputs, *args, **kwargs):
        input_width = inputs.shape[3]

        if input_width == self.width:
            residual = inputs
        else:
            residual = self.res(inputs)

        x = self.bn(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = tf.keras.layers.Add()([x, residual])
        return x


class DownBlock(Layer):
    def __init__(self, width, block_depth, **kwargs):
        super(DownBlock, self).__init__(**kwargs)
        self.width = width
        self.block_depth = block_depth
        self.res_blocks = []
        for _ in range(block_depth):
            self.res_blocks.append(ResidualBlock(self.width))
        self.pool = tf.keras.layers.AveragePooling2D(pool_size=2)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for res_block in self.res_blocks:
            x = res_block(x)
        skip = x
        x = self.pool(x)
        return x, skip

    def get_config(self):
        config = super(DownBlock, self).get_config()
        config.update({"width": self.width, "block_depth": self.block_depth})
        return config


class UpBlock(Layer):
    def __init__(self, width, block_depth, **kwargs):
        super(UpBlock, self).__init__(**kwargs)
        self.width = width
        self.block_depth = block_depth
        self.res_blocks = []
        for _ in range(block_depth):
            self.res_blocks.append(ResidualBlock(self.width))
        self.up = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")

    def call(self, inputs, *args, **kwargs):
        x, skip = inputs
        x = self.up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        for res_block in self.res_blocks:
            x = res_block(x)
        return x

    def get_config(self):
        config = super(UpBlock, self).get_config()
        config.update({"width": self.width, "block_depth": self.block_depth})
        return config


class PositionalEmbedding(Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def call(self, position, *args, **kwargs):
        half_dim = tf.cast(self.dim // 2, tf.float32)
        embeddings = tf.math.log(10000.0) / (half_dim - 1)
        embeddings = tf.exp(tf.range(half_dim) * -embeddings)
        embeddings = position[:, None] * embeddings[None, :]
        embeddings = tf.concat([tf.sin(embeddings), tf.cos(embeddings)], axis=-1)
        return embeddings

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({"dim": self.dim})
        return config


# generate upsample block and downsample block
# Make a UNet class and init these blocks into layer lists
# while building the Unet keep the track of the output of each downsample block in a list.
# while upsampling, reverse this list and concat the output of each upsampled layer with the tracked list.

# take an image before passing it to the UNet add positional embeddings to it
# use the PositionalEmbeddings class
# the embedded out => (embed_dim, embed_dim) is passed on to the UNet

class UNet(Model):
    def __init__(self, dim, num_channels, num_channels_per_layer, block_depth):
        super().__init__()
        self.dim = dim
        self.widths = num_channels_per_layer
        self.num_channels = num_channels
        self.block_depth = block_depth
        self.skips = []
        self.downsample_blocks = []
        for width in self.widths[:-1]:
            self.downsample_blocks.append(DownBlock(width, block_depth, name=f"down_block_{width}"))

        self.upsample_blocks = []
        for width in reversed(self.widths[:-1]):
            self.upsample_blocks.append(UpBlock(width, block_depth, name=f"up_block_{width}"))

        self.bottle_necks = []
        for i in range(block_depth):
            self.bottle_necks.append(ResidualBlock(self.widths[-1], name=f"bottle_neck_{i}"))

        self.time_mlp = tf.keras.Sequential(
            [
                PositionalEmbedding(self.dim),
                layers.UpSampling2D(size=config.IMAGE_SIZE, interpolation="nearest"),
            ]
        )
        self.concat = layers.Concatenate(name="mlp_concat")
        self.input_conv = layers.Conv2D(self.widths[0], kernel_size=1, name="input_conv")
        self.last_layer = tf.keras.layers.Conv2D(
            self.num_channels, kernel_size=1, kernel_initializer="zeros", name="out_conv"
        )

    def model(self):
        ins = (tf.keras.Input(shape=(64, 64, 3)), tf.keras.Input(shape=(1, 1)))
        model = Model(inputs=ins, outputs=self.call(ins))
        return model

    def summary(self, line_length=None, positions=None, print_fn=None):
        model = self.model()
        return model.summary()

    def call(self, inputs, training=None, mask=None):
        ins, time = inputs
        e = self.time_mlp(time)
        x = self.input_conv(ins)
        x = self.concat([x, e])

        for layer in self.downsample_blocks:
            x, skip = layer(x)
            self.skips.append(skip)

        for bottle_neck in self.bottle_necks:
            x = bottle_neck(x)

        rev_skips = reversed(self.skips)
        for skip, layer in zip(rev_skips, self.upsample_blocks):
            x = layer([x, skip])
            # x = tf.keras.layers.Concatenate()([x, skip])

        out = self.last_layer(x)
        return out

    def get_config(self):
        return {
            "dim": self.dim,
            "num_channels": self.num_channels,
            "num_channels_per_layer": self.widths,
            "self.block_depth": self.block_depth
        }


if __name__ == "__main__":
    model = UNet(config.EMBEDDING_DIM, 3, config.WIDTHS, config.BLOCK_DEPTH)
    # print(len(model.upsample_blocks), len(model.downsample_blocks))
    # model.build(input_shape=[(None, 64, 64, 3), (None, 1, 1)])
    # out = model.call([tf.keras.Input(shape=(64, 64, 3)), tf.keras.Input(shape=(1, 1))])
    print(model.summary())
    print(len(model.trainable_variables))

    # ones test
    ones = tf.ones((1, config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
    t_ones = tf.ones((1, 1, 1))
    with tf.device("/CPU:0"):
        inferred_ones = model((ones, t_ones))


