# the test and eval scripts along with the loss
import config
import tensorflow as tf
from model import UNet
from tensorflow import keras
# from tqdm import tqdm
from utils import corrupt_image, get_alphas, inference_samples, linear_beta_schedule


def loss_fn(noise, noise_pred, loss_type="l1"):
    # loss function for the training of the diffusion model
    # that denoises the input noisy image
    if loss_type == "l1":
        loss = tf.keras.losses.mae(noise, noise_pred)
    elif loss_type == "l2":
        loss = tf.keras.losses.mse(noise, noise_pred)
    elif loss_type == "huber":
        loss = tf.keras.losses.huber(noise, noise_pred)
    else:
        raise NotImplementedError(f"The loss {loss_type} is not implemented.")

    return loss


@tf.function
def train_fn(img_batch, model, betas, optimizer):
    # test the model for one pass of the dataset
    # return the losses after
    # batch_shape = tf.gather(img_batch.shape, 0)
    t_batch = tf.random.uniform((config.BATCH_SIZE, 1, 1), 1, config.TIME_STEPS)
    noise = tf.random.normal((config.BATCH_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
    noisy_image_batch = corrupt_image(
        img_batch, tf.expand_dims(t_batch, axis=-1), betas, noise=noise
    )
    with tf.GradientTape() as tape:
        noise_pred = model([noisy_image_batch, t_batch])
        step_loss = loss_fn(noise, noise_pred, loss_type="l1")

    gradients = tape.gradient(step_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return step_loss


@tf.function
def eval_fn(num_images, model):
    return inference_samples(model, num_images)


if __name__ == "__main__":
    model = UNet(
        config.EMBEDDING_DIM,
        num_channels=3,
        num_channels_per_layer=config.WIDTHS,
        block_depth=config.BLOCK_DEPTH,
    )

    images = inference_samples(model, 4)
    print(images.shape)
