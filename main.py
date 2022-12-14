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
from tensorflow.keras.utils import Progbar
import matplotlib.pyplot as plt
import numpy as np


def main():
    model = UNet(
        config.EMBEDDING_DIM,
        num_channels=3,
        num_channels_per_layer=config.WIDTHS,
        block_depth=config.BLOCK_DEPTH,
    )
    model = model.model()
    # model.build(input_shape=[(None, config.IMAGE_SIZE, config.IMAGE_SIZE, 3), (None, 1, 1)])
    optimizer = AdamW(weight_decay=config.DECAY, learning_rate=config.LR)

    if config.BETA_SCHEDULE == "linear":
        betas = linear_beta_schedule(config.TIME_STEPS)
    elif config.BETA_SCHEDULE == "cosine":
        betas = cosine_beta_schedule(config.TIME_STEPS)
    else:
        raise NotImplementedError(f"{config.BETA_SCHEDULE} not implemented.")

    # tensorboard
    current_time = datetime.datetime.now().strftime("%Y_%m_%d-%H~%M~%S")
    train_log_dir = f"models/logs/{current_time}/train"
    eval_log_dir = f"models/logs/{current_time}/test"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

    # Metrics
    loss_metric = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)

    for epoch in range(config.NUM_EPOCHS):
        print("Training Epoch: {}".format(epoch + 1))
        # keras progress bar
        pb = Progbar(len(train_dataset) * config.BATCH_SIZE, stateful_metrics=["train_loss"])
        for idx, img_batch in enumerate(train_dataset):
            # test for one step
            loss_of_step = train_fn(img_batch, model, betas, optimizer)
            loss_metric.update_state(loss_of_step)
            pb.update((idx+1) * config.BATCH_SIZE, values=[("train_loss", loss_metric.result())])

        # print("Loss for Epoch {}: {:.2}".format(epoch + 1, loss_metric.result()))
        if (epoch+1) % 2 == 0:
            print("Evaluating...")
            start = time.perf_counter()
            imgs = eval_fn(num_images=4, model=model)
            end = time.perf_counter()
            imgs = (imgs + 1) / 2
            imgs *= 255
            # plt.imshow(imgs[0].numpy().astype(np.uint8))
            # plt.show()
            print(f"Time taken: {end - start}")

            with eval_summary_writer.as_default():
                tf.summary.image(
                    "diffusion training",
                    tf.cast(imgs, tf.uint8),
                    step=epoch,
                    max_outputs=4,
                    description="The images generated after every 2 epochs.",
                )

        with train_summary_writer.as_default():
            tf.summary.scalar("loss", loss_metric.result(), step=epoch)

        loss_metric.reset_states()

    model.save("models/diffusion_model_epochs_{}_oxford_data".format(config.NUM_EPOCHS), save_traces=True)


if __name__ == "__main__":
    main()
