import config
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow_datasets import load
import matplotlib.pyplot as plt


def preprocess_data(data):
    height = tf.shape(data["image"])[0]
    width = tf.shape(data["image"])[1]
    crop_size = tf.minimum(height, width)
    image = tf.image.crop_to_bounding_box(
        data["image"],
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size,
    )

    image = tf.image.resize(
        image, [config.IMAGE_SIZE, config.IMAGE_SIZE], antialias=True
    )
    # normalize the image
    image /= 255.0
    image = -1. + (2. * image)
    return tf.clip_by_value(image, -1.0, 1.0)


def get_dataset(split):

    return (
        load(config.DATA_NAME, split=split, shuffle_files=True)
        .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
        # .take(4)
        .cache()
        .repeat(config.DATA_REPETITIONS)
        .shuffle(10 * config.BATCH_SIZE)
        .batch(config.BATCH_SIZE, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


train_dataset = get_dataset("train[:80%]+validation[:80%]+test[:80%]")
val_dataset = get_dataset("train[:20%]+validation[:20%]+test[:20%]")

if __name__ == "__main__":
    for data in train_dataset:
        print(data.shape)
        # print(data.numpy)
        data = (data[0] + 1.) / 2.
        data = data.numpy()
        # img = Image.fromarray(data.astype(np.uint8))
        # img.show()
        fig, ax = plt.subplots()
        ax.imshow(data)
        fig.show()
        break
