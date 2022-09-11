import tensorflow as tf
from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow import keras
import config
from utils import linear_beta_schedule, cosine_beta_schedule
from engine import train_fn
from model import UNet
from dataloader import train_dataset, val_dataset


def main():
    model = UNet(config.EMBEDDING_DIM,
                 num_channels=3,
                 num_channels_per_layer=config.WIDTHS,
                 block_depth=config.BLOCK_DEPTH)

    optimizer = AdamW(config.LR,
                     config.DECAY)

    if config.BETA_SCHEDULE == "linear":
        betas = linear_beta_schedule(config.TIME_STEPS)
    elif config.BETA_SCHEDULE == "cosine":
        betas = cosine_beta_schedule(config.TIME_STEPS)
    else:
        raise NotImplementedError(f"{config.BETA_SCHEDULE} not implemented.")

    for epoch in range(config.NUM_EPOCHS):
        print("Training Epoch: {}".format(epoch + 1))
        loss = train_fn(train_dataset, model, betas, optimizer)
        print("Loss for Epoch {}: {:.2}".format(epoch+1, loss))


if __name__ == "__main__":
    main()

