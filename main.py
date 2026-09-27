from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cat_dog_model.keras"

app = FastAPI(
    title="PawVision AI — Cats vs Dogs",
    description="Web API for the existing Cats vs Dogs CNN model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# IMPORTANT: existing CNN model is loaded exactly as provided. No model logic is changed.
model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_bytes: bytes):
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    img = img.resize((256, 256))
    # The trained CNN expects image values on the same 0..1 scale used during
    # standard image-generator training. This does NOT change the model; it
    # only supplies the input in the correct format.
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(BASE_DIR / "index.html", media_type="text/html")


@app.get("/model-info")
def model_info():
    return {
        "name": "PawVision CNN",
        "type": "Convolutional Neural Network",
        "input": "256 × 256 × 3",
        "classes": ["Cat", "Dog"],
        "output": "Single sigmoid probability",
        "threshold": 0.5,
        "preprocessing": "RGB resize to 256×256 + pixel scaling to 0..1",
        "weights": "Existing cat_dog_model.keras",
    }


@app.get("/health")
def health():
    return {
        "status": "online",
        "service": "PawVision AI",
        "model_loaded": True,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum size is 10 MB.")

    try:
        img_array = preprocess_image(image_bytes)
        predictions = model(img_array, training=False)
        score = float(np.asarray(predictions).reshape(-1)[0])

        # The saved model has a single sigmoid output: probability of the
        # positive class (Dog). Confidence is the probability of the class
        # that was actually selected, not a guarantee that the prediction is
        # correct. A wrong prediction can still have high confidence.
        label = "Dog" if score >= 0.5 else "Cat"
        confidence = score if label == "Dog" else (1.0 - score)

        if confidence >= 0.90:
            interpretation = "Very strong model signal"
        elif confidence >= 0.70:
            interpretation = "Strong model signal"
        elif confidence >= 0.55:
            interpretation = "Moderate model signal"
        else:
            interpretation = "Low-confidence / uncertain signal"

        return {
            "class": label,
            "confidence": round(confidence * 100, 2),
            "dog_probability": round(score * 100, 2),
            "interpretation": interpretation,
            "input_size": "256x256",
            "model_output": "sigmoid",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
