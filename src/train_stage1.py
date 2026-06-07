import os

from src.config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    EPOCHS,
    MODEL_DIR,
    BEST_MODEL_PATH
)

from src.data_loader import load_dataset
from src.model import build_model
from src.callbacks import get_callbacks
from src.evaluate import evaluate_model

# CREATE MODEL DIRECTORY
os.makedirs(MODEL_DIR, exist_ok=True)

# LOAD DATA
print("\nLoading datasets...\n")

# TRAIN DATASET
train_ds, class_names = load_dataset(
    TRAIN_DIR,
    shuffle=True,
    augment=True
)

# VALIDATION DATASET
val_ds, _ = load_dataset(
    VAL_DIR,
    class_names=class_names,
    shuffle=False,
    augment=False
)

# TEST DATASET
test_ds, _ = load_dataset(
    TEST_DIR,
    class_names=class_names,
    shuffle=False,
    augment=False
)

# PRINT CLASS MAPPING
print("\nClass Mapping:")
print(class_names)

print("\nClasses:", class_names)

# BUILD MODEL (FROZEN BACKBONE)
print("\nBuilding Stage 1 model...\n")
model = build_model()
model.summary()

# CALLBACKS (STAGE 1)
callbacks = get_callbacks(stage="stage1")

# TRAIN STAGE 1
print("\nStarting Stage 1 training...\n")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# EVALUATION
evaluate_model(model, test_ds)

print("\nStage 1 completed.\n")
print(f"Best model saved at: {BEST_MODEL_PATH}")