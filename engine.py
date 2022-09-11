# the train and eval scripts along with the loss
import tensorflow as tf
from tensorflow import keras
import config
from utils import get_alphas, corrupt_image, linear_beta_schedule, inference_samples
from tqdm import tqdm
from model import UNet


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


def train_fn(dataset, model, betas, optimizer):
    # train the model for one pass of the dataset
    # return the losses after
    loss_arr = []
    for img_batch in tqdm(dataset, total=len(dataset)):
        with tf.GradientTape() as tape:
            t_batch = tf.random.uniform((config.BATCH_SIZE, 1, 1), 1, config.TIME_STEPS)
            noise = tf.random.normal(img_batch.shape)
            noisy_image_batch = corrupt_image(img_batch, tf.expand_dims(t_batch, axis=-1), betas, noise=noise)
            noise_pred = model([noisy_image_batch, t_batch])
            step_loss = loss_fn(noise, noise_pred, loss_type="l1")
            loss_arr.append(step_loss)

        gradients = tape.gradient(step_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return sum(loss_arr) / len(loss_arr)


def eval_fn():
    pass


if __name__ == "__main__":
    model = UNet(config.EMBEDDING_DIM,
                 num_channels=3,
                 num_channels_per_layer=config.WIDTHS,
                 block_depth=config.BLOCK_DEPTH)

    images = inference_samples(model, 4)
    print(images.shape)