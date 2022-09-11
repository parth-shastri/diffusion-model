# architecture
EMBEDDING_DIM = 32
WIDTHS = [32, 64, 96, 128]
BLOCK_DEPTH = 2
IMAGE_SIZE = 64
# datasets
DATA_NAME = "oxford_flowers102"
NUM_EPOCHS = 10
DATA_REPETITIONS = 5
BATCH_SIZE = 4
# algorithm
BETA_START = 0.0001
BETA_END = 0.02
TIME_STEPS = 200
BETA_SCHEDULE = "linear"
# optimizer
LR = 1e-3
DECAY = 1e-4

# 64->32->16->8->8->16->32->64
