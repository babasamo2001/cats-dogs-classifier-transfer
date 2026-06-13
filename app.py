import io
import time
import zipfile
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from inference import predict_image
from s3_downloader import download_model_from_s3
from src.utils import setup_logger

# --- S3 CONFIGURATION ---
S3_BUCKET_NAME = "cats-dogs-model-785534816302-eu-west-1-an"
S3_MODEL_KEY = "best_best_savedmodel.zip"
LOCAL_ZIP_PATH = "/tmp/best_best_savedmodel.zip"
EXTRACT_DIR = "/tmp/best_best_savedmodel"

request_count = 0
total_inference_time = 0.0

# LOGGERS
logger = setup_logger("app", "logs/app.log")
infer_logger = setup_logger("inference", "logs/inference.log")


# --- APP INIT & MODEL LOADING ---
app = FastAPI()

# 1. Pull down the compressed SavedModel bundle from S3
download_model_from_s3(S3_BUCKET_NAME, S3_MODEL_KEY, LOCAL_ZIP_PATH, logger)

# 2. Extract zip archive contents to /tmp
logger.info(f"Extracting zipped model directory to: {EXTRACT_DIR}")
with zipfile.ZipFile(LOCAL_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

# 3. Load the model from the unzipped directory track
logger.info(f"Loading TensorFlow model graph from directory: {EXTRACT_DIR}")
model = tf.keras.models.load_model(EXTRACT_DIR)
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
            "transient_path": EXTRACT_DIR,
            "s3_source": f"s3://{S3_BUCKET_NAME}/{S3_MODEL_KEY}",
            "timestamp": time.time()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# PREDICT ENDPOINT
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # Pass image and global model to inference
        result = predict_image(image, model)

        infer_logger.info(f"{result['class']} | {result['confidence']}")
        return result

    except Exception as e:
        logger.error(str(e))
        return {"error": str(e)}


# MODEL INFO ENDPOINT
@app.get("/model-info")
def model_info():
    total_params = int(sum([tf.size(v).numpy() for v in model.trainable_weights]))
    return {
        "model_path": EXTRACT_DIR,
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "total_params": total_params
    }


# REQUEST COUNTER ENDPOINT
@app.get("/request-count")
def request_count_endpoint():
    global request_count
    request_count += 1
    return {"total_requests_to_this_endpoint": request_count}