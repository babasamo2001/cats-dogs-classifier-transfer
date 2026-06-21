import io
import os
import time
from contextlib import asynccontextmanager
import boto3
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from inference import predict_image
from src.utils import setup_logger

# --- S3 CONFIGURATION ---
S3_BUCKET_NAME = "cats-dogs-model-785534816302-eu-west-1-an"
S3_FOLDER_PREFIX = "best_model_savedmodel/"  # Trailing slash matches directory key
LOCAL_MODEL_DIR = "/tmp/best_model_savedmodel"

request_count = 0
total_inference_time = 0.0

# LOGGERS
logger = setup_logger("app", "logs/app.log")
infer_logger = setup_logger("inference", "logs/inference.log")

# Global placeholder for the model instance loaded into memory
model = None


def download_model_dir_from_s3():
    """Helper function to download all files from the S3 SavedModel directory prefix."""
    logger.info(f"Syncing directory from s3://{S3_BUCKET_NAME}/{S3_FOLDER_PREFIX}")
    try:
        s3 = boto3.client('s3')
        # List all objects matching the folder prefix path
        paginator = s3.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=S3_FOLDER_PREFIX):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                s3_key = obj["Key"]
                # Skip directory placeholders themselves
                if s3_key.endswith('/'):
                    continue

                # Compute local transient file pathing matching the nested directory tree
                relative_path = os.path.relpath(s3_key, S3_FOLDER_PREFIX)
                local_file_path = os.path.join(LOCAL_MODEL_DIR, relative_path)

                # Make nested structures like variables/ directory on the fly
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download individual asset fragments
                logger.info(f"Downloading slice: {relative_path}")
                s3.download_file(S3_BUCKET_NAME, s3_key, local_file_path)

        logger.info("SavedModel architecture directory completely downloaded.")
    except Exception as e:
        logger.error(f"S3 Directory Sync failed: {str(e)}")
        raise e


# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # 1. Fetch directory structure from cloud storage
    download_model_dir_from_s3()

    # 2. Clear old graphs and optimize memory allocations
    logger.info("Optimizing TensorFlow environment variables for low-RAM environment...")
    tf.keras.backend.clear_session()

    logger.info(f"Loading native SavedModel directory structural mapping from: {LOCAL_MODEL_DIR}")

    # 3. Load model directly using directory target mapping
    model = tf.keras.models.load_model(LOCAL_MODEL_DIR, compile=False)
    logger.info("TensorFlow SavedModel successfully assigned to RAM.")

    yield
    logger.info("Cleaning up resource configurations during shutdown...")


# --- APP INIT ---
app = FastAPI(lifespan=lifespan)

# STATIC FILES
app.mount("/static", StaticFiles(directory="static"), name="static")


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
            "transient_path": LOCAL_MODEL_DIR,
            "s3_source": f"s3://{S3_BUCKET_NAME}/{S3_FOLDER_PREFIX}",
            "timestamp": time.time()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# PREDICT ENDPOINT
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        result = predict_image(image, model)
        infer_logger.info(f"{result['class']} | {result['confidence']}")
        return result
    except Exception as e:
        logger.error(str(e))
        return {"error": str(e)}


# MODEL INFO ENDPOINT
@app.get("/model-info")
def model_info():
    try:
        total_params = int(sum([tf.size(v).numpy() for v in model.trainable_weights]))
    except Exception:
        total_params = "Unavailable"

    return {
        "model_path": LOCAL_MODEL_DIR,
        "input_shape": str(model.input_shape) if hasattr(model, 'input_shape') else "Dynamic",
        "output_shape": str(model.output_shape) if hasattr(model, 'output_shape') else "Dynamic",
        "total_params": total_params
    }


# REQUEST COUNTER ENDPOINT
@app.get("/request-count")
def request_count_endpoint():
    global request_count
    request_count += 1
    return {"total_requests_to_this_endpoint": request_count}