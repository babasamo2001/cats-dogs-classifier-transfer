import io
import os
import time
import boto3
import tensorflow as tf
from botocore.exceptions import NoCredentialsError, ClientError
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from inference import predict_image
from src.utils import setup_logger

# --- S3 CONFIGURATION ---
S3_BUCKET_NAME = "cats-dogs-model-785534816302-eu-west-1-an"  # Replace with your actual bucket name
S3_MODEL_KEY = "models/best_model.keras"  # Replace with the path to the file inside your bucket

# Pointing this to /tmp guarantees Linux wipes it when the EC2 instance shuts down/reboots
LOCAL_MODEL_PATH = "/tmp/best_model.keras"

request_count = 0
total_inference_time = 0.0

# LOGGERS
logger = setup_logger("app", "logs/app.log")
infer_logger = setup_logger("inference", "logs/inference.log")


# --- S3 DOWNLOAD FUNCTION ---
def download_model_from_s3(bucket_name: str, s3_key: str, local_path: str):
    """
    Downloads the model file from S3 to the /tmp folder.
    Because it checks os.path.exists, it only downloads ONCE per startup.
    When the server shuts down, Linux purges /tmp, forcing a fresh download next time.
    """
    if os.path.exists(local_path):
        logger.info(f"Model already found in temp storage at {local_path}. Skipping S3 sync.")
        return

    logger.info(f"Fresh startup detected. Downloading model from S3://{bucket_name}/{s3_key} to {local_path}...")

    try:
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket_name, s3_key, local_path)
        logger.info(f"Model successfully cached in temporary storage: {local_path}")
    except NoCredentialsError:
        logger.error("AWS credentials not found. Ensure your EC2 instance has an IAM role attached.")
        raise RuntimeError("AWS credentials missing.")
    except ClientError as e:
        logger.error(f"Failed to download model from S3: {e}")
        raise e


# --- APP INIT & MODEL LOADING ---
app = FastAPI()

# This runs during FastAPI initialization on startup
download_model_from_s3(S3_BUCKET_NAME, S3_MODEL_KEY, LOCAL_MODEL_PATH)

logger.info(f"Loading TensorFlow model from transient path: {LOCAL_MODEL_PATH}")
model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
logger.info("TensorFlow model loaded successfully into RAM.")

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
            "transient_path": LOCAL_MODEL_PATH,
            "s3_source": f"s3://{S3_BUCKET_NAME}/{S3_MODEL_KEY}",
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
        "model_path": LOCAL_MODEL_PATH,
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