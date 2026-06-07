import os

# IMAGE SETTINGS

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 5

NUM_CLASSES = 3

CLASS_NAMES = ["cat", "dog", "others"]

# DATA PATHS

TRAIN_DIR = "data/train"

VAL_DIR = "data/val"

TEST_DIR = "data/test"

# MODEL PATHS

MODEL_DIR = "models"

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_model.keras"
)

# TRAINING SETTINGS

LEARNING_RATE = 0.00001

PATIENCE_EARLY_STOPPING = 3

PATIENCE_REDUCE_LR = 2

REDUCE_LR_FACTOR = 0.3