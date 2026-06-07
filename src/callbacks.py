import os
import tensorflow as tf

from src.config import (
    BEST_MODEL_PATH,
    PATIENCE_EARLY_STOPPING,
    PATIENCE_REDUCE_LR,
    REDUCE_LR_FACTOR
)

# BASE DIRECTORY
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# LOG DIRECTORY
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

TRAINING_LOG_PATH = os.path.join(LOG_DIR, "training_log.csv")

print(f"\nTraining log path: {TRAINING_LOG_PATH}\n")

# CALLBACKS
def get_callbacks(stage="stage1"):

    callbacks = []

    # MODEL CHECKPOINT (BEST MODEL SAVING)
    if stage == "stage1":
        monitor_metric = "val_accuracy"
        mode = "max"
    else:
        monitor_metric = "val_loss"
        mode = "min"

    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor=monitor_metric,
            save_best_only=True,
            mode=mode,
            verbose=1
        )
    )

    # EARLY STOPPING
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE_EARLY_STOPPING,
            restore_best_weights=True,
            verbose=1
        )
    )

    # REDUCE LEARNING RATE
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=PATIENCE_REDUCE_LR,
            verbose=1
        )
    )

    # CSV LOGGER
    callbacks.append(
        tf.keras.callbacks.CSVLogger(
            filename=TRAINING_LOG_PATH,
            separator=",",
            append=(stage == "stage2")  # important: don't overwrite stage1 logs
        )
    )

    print(f"\nCallbacks loaded for {stage} successfully.\n")

    return callbacks