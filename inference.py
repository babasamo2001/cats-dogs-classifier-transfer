import tensorflow as tf
import numpy as np
from PIL import Image
from src.config import CLASS_NAMES

# PREDICT IMAGE
def predict_image(image: Image.Image, model):
    # RESIZE
    image = image.resize((224, 224))

    # CONVERT TO ARRAY
    arr = np.expand_dims(
        np.array(image),
        axis=0
    ).astype(np.float32)  # Ensure data type matches input tensor requirements

    # TFSMLayer models are executed by calling them directly like a function
    raw_output = model(arr)

    # Handle dictionary output from TFSMLayer
    if isinstance(raw_output, dict):
        # Extract the first available tensor key from the dictionary dynamically
        first_key = list(raw_output.keys())[0]
        pred_tensor = raw_output[first_key]
        pred = pred_tensor.numpy()
    else:
        pred = raw_output.numpy() if hasattr(raw_output, 'numpy') else raw_output

    class_id = np.argmax(pred)
    confidence = float(np.max(pred))

    return {
        "class": CLASS_NAMES[class_id],
        "confidence": confidence
    }