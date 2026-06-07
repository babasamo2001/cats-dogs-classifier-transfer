import io

import tensorflow as tf

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import time
from PIL import Image

from src.config import BEST_MODEL_PATH
from inference import predict_image
from src.utils import setup_logger


request_count = 0
total_inference_time = 0.0

# APP INIT

app = FastAPI()

# STATIC FILES

app.mount("/static", StaticFiles(directory="static"), name="static")

# LOAD MODEL

model = tf.keras.models.load_model(BEST_MODEL_PATH)

# LOGGERS

logger = setup_logger("app", "logs/app.log")
infer_logger = setup_logger("inference", "logs/inference.log")

# FRONTEND ROUTE

@app.get("/", response_class=HTMLResponse)
def home():

    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

# HEALTH CHECK

@app.get("/health")
def health():
    try:
        return {
            "status": "ok",
            "service": "image-classifier-transfer",
            "model_loaded": model is not None,
            "model_path": BEST_MODEL_PATH,
            "timestamp": time.time()
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# PREDICT ENDPOINT

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        image = Image.open(
            io.BytesIO(await file.read())
        ).convert("RGB")

        result = predict_image(image)

        infer_logger.info(
            f"{result['class']} | {result['confidence']}"
        )

        return result

    except Exception as e:

        logger.error(str(e))

        return {"error": str(e)}


# MODEL INFO ENDPOINT

@app.get("/model-info")
def model_info():

    total_params = int(
        sum([tf.size(v).numpy() for v in model.trainable_weights])
    )

    return {
        "model_path": BEST_MODEL_PATH,
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "total_params": total_params
    }

# REQUEST COUNTER ENDPOINT

@app.get("/request-count")
def request_count_endpoint():
    global request_count
    request_count += 1

    return {
        "total_requests_to_this_endpoint": request_count
    }