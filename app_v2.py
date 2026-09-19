# Save this file as: app_v2.py
import re
import torch
import uvicorn
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import pypdf
import pytesseract

# Initialize FastAPI App
app = FastAPI(
    title="AI Job Posting Fraud Verification Engine",
    description="Production API with PyPDF2 & Pytesseract OCR extraction layers.",
    version="2.0.0"
)

# System Initialization (Load Model & Tokenizer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./prod_job_classifier" # Ensure this folder is in your project directory

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    print(f"✅ Model successfully loaded on: {device}")
except Exception as e:
    print(f"❌ Failed to load model weights: {str(e)}")
    raise SystemExit(1)

class PredictionResponse(BaseModel):
    final_prediction: str
    confidence_real: float
    confidence_fake: float
    safety_override_triggered: bool
    model_raw_prediction: str
    extracted_text_snippet: str

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")

def extract_text_from_image(file_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR Extraction failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def analyze_job_posting(file: UploadFile = File(...)):
    filename = file.filename.lower()
    file_bytes = await file.read()
    
    # 1. Route to correct text extraction engine
    if filename.endswith('.pdf'):
        extracted_text = extract_text_from_pdf(file_bytes)
    elif filename.endswith(('.png', '.jpg', '.jpeg', '.jpe')):
        extracted_text = extract_text_from_image(file_bytes)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Upload PDF, PNG, or JPG.")
        
    if not extracted_text or len(extracted_text.strip()) < 5:
        raise HTTPException(status_code=400, detail="Could not extract readable text from the document.")

    try:
        lower_text = extracted_text.lower()
        
        # --- LAYER A: REAL-TIME KEYWORD SAFETY FILTER ---
        scam_patterns = [
            r"earn \$\d{3,}", r"\$\d{3,4} a week", r"\$\d{2,4} daily",
            r"wire app", r"telegram chat", r"no experience required"
        ]
        rule_triggered = any(re.search(pattern, lower_text) for pattern in scam_patterns)
        
        # --- LAYER B: TRANSFORMER MODEL INFERENCE ---
        inputs = tokenizer(
            extracted_text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
            prediction_id = torch.argmax(outputs.logits, dim=-1).item()
            
        # FIXED: Correctly index elements out of the probabilities numpy array
        prob_real = float(probabilities[0] * 100)
        prob_fake = float(probabilities[1] * 100)
        
        labels_map = {0: "Real Job Posting", 1: "FRAUDULENT/FAKE Job Posting"}
        model_raw_pred = labels_map[prediction_id]
        
        # --- LAYER C: HYBRID OVERRIDE LOGIC ---
        safety_override = False
        final_result = model_raw_pred
        
        # FIXED: Explicitly mapping to index 1 string value inside dictionary
        if rule_triggered and prediction_id == 0:
            safety_override = True
            final_result = labels_map[1] 
            
        return PredictionResponse(
            final_prediction=final_result,
            confidence_real=prob_real,
            confidence_fake=prob_fake,
            safety_override_triggered=safety_override,
            model_raw_prediction=model_raw_pred,
            extracted_text_snippet=extracted_text[:300] + "..." if len(extracted_text) > 300 else extracted_text
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app_v2:app", host="0.0.0.0", port=8000, reload=False)
