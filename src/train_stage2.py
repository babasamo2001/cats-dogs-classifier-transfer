import tensorflow as tf

from src.config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    EPOCHS,
    BEST_MODEL_PATH
)

from src.data_loader import load_dataset
from src.callbacks import get_callbacks
from src.evaluate import evaluate_model

# LOAD DATA
print("\nLoading datasets...\n")

train_ds, class_names = load_dataset(TRAIN_DIR, shuffle=True, augment=True)
val_ds, _ = load_dataset(VAL_DIR, shuffle=False, augment=False)
test_ds, _ = load_dataset(TEST_DIR, shuffle=False, augment=False)

print("\nClasses:", class_names)

# LOAD BEST STAGE 1 MODEL
print("\nLoading best Stage 1 model...\n")

model = tf.keras.models.load_model(BEST_MODEL_PATH)

# RECOVER BACKBONE
print("\nUnfreezing backbone for fine-tuning...\n")

base_model = model.get_layer("MobileNetV3Small")

base_model.trainable = True

# Freeze early layers of backbone, fine-tune 30 last layers only
for layer in base_model.layers[:-30]:
    layer.trainable = False

# RECOMPILE FOR FINE-TUNING
print("\nRecompiling model...\n")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# CALLBACKS (STAGE 2)
callbacks = get_callbacks(stage="stage2")

# TRAIN STAGE 2
print("\nStarting Stage 2 fine-tuning...\n")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    initial_epoch=EPOCHS,
    callbacks=callbacks
)

# FINAL EVALUATION
evaluate_model(model, test_ds)

print("\nStage 2 completed.\n")