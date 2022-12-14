# The code for all the util functions required for training and sampling from the
# diffusion model
import config
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
import matplotlib.pyplot as plt
import requests
# from model import UNet

def get_alphas(betas):
    alphas = 1.0 - betas
    alphas_cum = tf.math.cumprod(alphas, axis=0)
    alphas_cum_prev = tf.pad(alphas_cum[:-1], [[1, 0]], constant_values=1.0)
    alphas_sqrt = tf.sqrt(alphas_cum)
    sqrt_one_minus_alphas_cum = tf.sqrt(1.0 - alphas_cum)
    return alphas_sqrt, sqrt_one_minus_alphas_cum, alphas_cum_prev, alphas


def extract(tensor, t):
    tensor_t = tf.gather(tensor, tf.cast(t, dtype=tf.int32))
    return tensor_t


def corrupt_image(image, t, betas, noise=None):
    if noise is None:
        noise = tf.random.normal(tf.shape(image))
    sqrt_a_, sqrt_one_minus_a_, _, _ = get_alphas(betas)
    sqrt_a_t, sqrt_one_minus_a_t = extract(sqrt_a_, t), extract(sqrt_one_minus_a_, t)
    # print(sqrt_a_t)
    cor_img = sqrt_a_t * image + sqrt_one_minus_a_t * noise
    return cor_img


def inference_samples(model, num_images):
    # take one sample from the learned markov chain
    x_t = tf.random.normal((num_images, config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
    betas = linear_beta_schedule(config.TIME_STEPS)
    sqrt_a_, sqrt_one_minus_a_, a_t_minus_one_, alphas = get_alphas(betas)
    posterior_betas = (1. - a_t_minus_one_ / sqrt_one_minus_a_**2) * betas
    sqrt_recip_alphas = tf.sqrt(1.0 / alphas)

    for t in range(config.TIME_STEPS):

        posterior_beta_t = extract(posterior_betas, t)
        beta_t = extract(betas, t)
        sqrt_a_t, sqrt_one_minus_a_t = extract(sqrt_a_, t), extract(
            sqrt_one_minus_a_, t
        )
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t)
        # x_t_minus_one = (1.0 / sqrt_a_t) * (
        #     x_t
        #     - (sqrt_one_minus_a_t**2)
        #     * model([x_t, tf.fill((x_t.shape[0], 1, 1), tf.cast(t, tf.float32))])
        #     / sqrt_one_minus_a_t
        # )

        model_mean = sqrt_recip_alphas_t * (
            x_t - beta_t * model([x_t, tf.fill((x_t.shape[0], 1, 1), tf.cast(t, tf.float32))]) / sqrt_one_minus_a_t
        )

        if t == 0:
            x_t_minus_one = model_mean
        else:
            z = tf.random.normal(x_t.shape)
            x_t_minus_one = model_mean + posterior_beta_t * z

        x_t = x_t_minus_one

    return x_t


def cosine_beta_schedule(timesteps, s=0.08):
    steps = timesteps + 1
    t = tf.linspace(0, timesteps, steps)
    alpha_cum = tf.cos((t / timesteps) + s) / ((1 + s) * np.pi * 0.5) ** 2
    alpha_cum = alpha_cum / alpha_cum[0]
    betas = 1 - (alpha_cum[1:] / alpha_cum[:-1])
    betas = tf.clip_by_value(betas, 0.0001, 0.9999)
    return tf.cast(betas, tf.float32)


def linear_beta_schedule(timesteps):
    """
    Schedule used in the original DDPMs paper
    :param timesteps:
    :return:
    """
    beta_start = 0.0001
    beta_end = 0.02
    return tf.linspace(beta_start, beta_end, timesteps, name="betas_t")


if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = Image.open(requests.get(url, stream=True).raw)
    img = img.resize((128, 128))
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = 2 * img - 1
    print(np.min(img), np.max(img))
    betas = linear_beta_schedule(200)
    # print(betas)
    # print(betas.dtype)
    a_t_, one_minus_a_t_, a_t_minus_one, _ = get_alphas(betas)
    # print(a_t_.dtype)
    # fig, axs = plt.subplots(1, 5, tight_layout=True)
    for i, t_step in enumerate([0, 50, 100, 150, 199]):
        img_ = corrupt_image(img, t_step, betas)
        # img_ = np.clip(img_, -1, 1)
        img_ = (img_ + 1) / 2
        img_ *= 255.0
        img_ = img_.numpy().astype(np.uint8)
        plt.imshow(img_)
        plt.show()
        # axs[i].axis("off")
        # axs[i].imshow(img_)
    # fig.show()

    # pil_img_ = Image.fromarray(np.array(img_, dtype=np.uint8))
    # pil_img_.show()

    # model = UNet(config.EMBEDDING_DIM, 3, config.WIDTHS, config.BLOCK_DEPTH).model()
    loaded_model = tf.keras.models.load_model("models/diffusion_model_epochs_10_oxford_data")

    with tf.device("cpu"):
        # inferred_img = inference_samples(model, 1)
        inferred_img_loaded = inference_samples(loaded_model, 1)
        # normal = tf.random.normal((1, 64, 64, 3))
        # t_ones = tf.ones((1, 1, 1))
        # normal_test = model([normal, t_ones])

    fig, axs = plt.subplots()
    inf_img = (1 + inferred_img_loaded) / 2
    inf_img *= 255
    inf_img = inf_img.numpy()[0].astype(np.uint8)
    axs.imshow(inf_img)
    fig.show()