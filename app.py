from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image as PILImage
from transformers import AutoImageProcessor, SiglipForImageClassification
import torch
import warnings
import io

MODEL_IDENTIFIER = "Ateeqq/ai-vs-human-image-detector"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore", message="Possibly corrupt EXIF data.")
warnings.filterwarnings("ignore", message=".*You are using the default legacy behaviour.*")

processor = AutoImageProcessor.from_pretrained(MODEL_IDENTIFIER)
model = SiglipForImageClassification.from_pretrained(MODEL_IDENTIFIER).to(DEVICE)
model.eval()

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AI vs Human Detector API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        results = {
            model.config.id2label[i]: round(prob.item(), 4)
            for i, prob in enumerate(probabilities)
        }
        return JSONResponse(content={"prediction": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
