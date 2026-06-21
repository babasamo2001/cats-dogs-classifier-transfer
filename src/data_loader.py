import tensorflow as tf

from src.config import IMG_SIZE, BATCH_SIZE
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# DATA AUGMENTATION
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# LOAD DATASET
def load_dataset(
    path,
    class_names=None,
    shuffle=True,
    augment=False
):

    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        color_mode="rgb",
        class_names=class_names
    )

    # Save actual class names
    class_names = ds.class_names

    print(f"\nDataset Loaded: {path}")
    print(f"Class Mapping: {class_names}")

    # PREPROCESSING
    def preprocess(image, label):

        # Convert image to float32
        image = tf.cast(image, tf.float32)

        # Data augmentation
        if augment:
            image = data_augmentation(
                image,
                training=True
            )

        return image, label

    # APPLY PREPROCESSING
    ds = ds.map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # PREFETCH FOR PERFORMANCE
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, class_names