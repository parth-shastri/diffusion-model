import datetime
import time

import config
import tensorflow as tf
from dataloader import train_dataset, val_dataset
from engine import eval_fn, train_fn
from model import UNet
from tensorflow import keras
from tensorflow_addons.optimizers import AdamW
from utils import cosine_beta_schedule, linear_beta_schedule


def main():
    model = UNet(
        config.EMBEDDING_DIM,
        num_channels=3,
        num_channels_per_layer=config.WIDTHS,
        block_depth=config.BLOCK_DEPTH,
    )

    optimizer = AdamW(config.LR, config.DECAY)

    if config.BETA_SCHEDULE == "linear":
        betas = linear_beta_schedule(config.TIME_STEPS)
    elif config.BETA_SCHEDULE == "cosine":
        betas = cosine_beta_schedule(config.TIME_STEPS)
    else:
        raise NotImplementedError(f"{config.BETA_SCHEDULE} not implemented.")

    # tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f"models/logs/{current_time}/train"
    eval_log_dir = f"models/logs/{current_time}/test"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

    # Metrics
    loss_metric = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)

    for epoch in range(config.NUM_EPOCHS):
        print("Training Epoch: {}".format(epoch + 1))
        train_fn(train_dataset, model, betas, optimizer, loss_metric)
        print("Loss for Epoch {}: {:.2}".format(epoch + 1, loss_metric.result()))
        if (epoch+1) % 2 == 0:
            print("Evaluating...")
            start = time.perf_counter()
            imgs = eval_fn(num_images=4, model=model)
            end = time.perf_counter()
            print(f"Time taken: {end - start}")

            with eval_summary_writer.as_default():
                tf.summary.image(
                    "diffusion training",
                    imgs,
                    step=epoch,
                    max_outputs=4,
                    description="The images generated after every 2 epochs.",
                )

        with train_summary_writer.as_default():
            tf.summary.scalar("loss", loss_metric.result(), step=epoch)

        loss_metric.reset_states()

    model.save("diffusion_model_epochs_10_oxford_data")


if __name__ == "__main__":
    main()
