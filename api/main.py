from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Mount static files (CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates (for HTML rendering)
templates = Jinja2Templates(directory="templates")

# Allow CORS for frontend
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL_PATH = "../models/potato_disease_model_1.keras"
MODEL = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    """Convert uploaded file data into a preprocessed image tensor."""
    image = Image.open(BytesIO(data)).convert("RGB")  # Ensure it's RGB
    image = image.resize((256, 256))  # Resize to model's expected input size
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict disease from uploaded image file."""
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)  # Add batch dimension

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        "class": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
