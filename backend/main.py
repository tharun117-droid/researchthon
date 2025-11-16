from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import logging
from fastapi.middleware.cors import CORSMiddleware  # <-- IMPORT THIS

from processors.image_processor import image_processor
from processors.text_processor import text_processor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Explainable Deepfake Detection API",
    description="Multi-modal AI-generated content detection with explainability",
    version="1.0.0"
)

# --- ADD THIS MIDDLEWARE SECTION ---
origins = [
    "http://localhost",
    "http://localhost:5173", # Assuming your React frontend runs here
    "http://127.0.0.1:5173",
    "http://192.168.0.101:8088/",
    # Add any other origins you need to support
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------------------------

class TextRequest(BaseModel):
    text: str

class DetectionResponse(BaseModel):
    prediction: str
    confidence: float
    is_fake: bool
    explanations: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("🚀 Initializing Deepfake Detection API...")
    # Load placeholder models
    # NOTE: See section 2 about this path!
    image_processor.load_model("models/image_detector.pth")
    text_processor.load_model("models/text_detector.pth")
    logger.info("✅ All models initialized (placeholder)")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}

@app.post("/detect/image", response_model=DetectionResponse)
async def detect_image(file: UploadFile = File(...)):
    """Detect AI-generated images"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image (JPEG, PNG)")
        
        # Read and process image
        image_data = await file.read()
        image_tensor = image_processor.preprocess_image(image_data)
        
        # Get prediction and explanations
        prediction = image_processor.predict(image_tensor)
        explanations = image_processor.generate_explanations(image_tensor)
        
        return DetectionResponse(
            prediction=prediction['prediction'],
            confidence=prediction['confidence'],
            is_fake=prediction['is_fake'],
            explanations=explanations
        )
        
    except Exception as e:
        logger.error(f"Image detection error: {str(e)}")
        raise HTTPException(500, f"Image processing error: {str(e)}")

@app.post("/detect/text", response_model=DetectionResponse)
async def detect_text(request: TextRequest):
    """Detect AI-generated text"""
    try:
        text = request.text.strip()
        if len(text) < 10:
            raise HTTPException(400, "Text must be at least 10 characters")
        
        # Preprocess and predict
        text_tensor = text_processor.preprocess_text(text)
        prediction = text_processor.predict(text_tensor)
        explanations = text_processor.generate_explanations(text, text_tensor)
        
        return DetectionResponse(
            prediction=prediction['prediction'],
            confidence=prediction['confidence'],
            is_fake=prediction['is_fake'],
            explanations=explanations
        )
        
    except Exception as e:
        logger.error(f"Text detection error: {str(e)}")
        raise HTTPException(500, f"Text processing error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)