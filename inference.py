import numpy as np
import tensorflow as tf
from PIL import Image
from src.config import CLASS_NAMES

# PREDICT IMAGE
def predict_image(image: Image.Image, model: tf.keras.Model):

    # RESIZE
    image = image.resize((224, 224))

    # CONVERT TO ARRAY (No normalization applied)
    arr = np.expand_dims(
        np.array(image),
        axis=0
    )

    # PREDICT using the passed model
    pred = model.predict(
        arr,
        verbose=0
    )

    class_id = np.argmax(pred)

    confidence = float(
        np.max(pred)
    )

    return {
        "class": CLASS_NAMES[class_id],
        "confidence": confidence
    }