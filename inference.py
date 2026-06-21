import tensorflow as tf
import numpy as np
from PIL import Image
from src.config import CLASS_NAMES  # Removed BEST_MODEL_PATH since it's not needed here

# PREDICT IMAGE
def predict_image(image: Image.Image, model):  # <-- Added model parameter here
    # RESIZE
    image = image.resize((224, 224))

    # CONVERT TO ARRAY
    arr = np.expand_dims(
        np.array(image),
        axis=0
    )

    # PREDICT using the passed model instance
    pred = model.predict(
        arr,
        verbose=0
    )

    class_id = np.argmax(pred)
    confidence = float(np.max(pred))

    return {
        "class": CLASS_NAMES[class_id],
        "confidence": confidence
    }