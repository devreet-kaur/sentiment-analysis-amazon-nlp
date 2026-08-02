"""
src/app.py - MAI203 Amazon Fine Food Reviews Sentiment API
FastAPI REST endpoint for serving sentiment predictions.

Run (macOS):   uvicorn src.app:app --reload --port 5001
Run (Windows): uvicorn src.app:app --reload --port 8000
Docs:          http://localhost:5001/docs  or  http://localhost:8000/docs
"""

import csv
import pickle
import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import nltk
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

nltk.download('punkt',        quiet=True)
nltk.download('punkt_tab',    quiet=True)
nltk.download('stopwords',    quiet=True)
nltk.download('wordnet',      quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Load config from params.yaml only
PARAMS   = yaml.safe_load(open('params.yaml'))
API_CFG  = PARAMS['api']
MON_CFG  = PARAMS['monitor']
PRE      = PARAMS['preprocessing']
LOG_PATH = Path(MON_CFG['current_data_path'])

STOP_WORDS = set(stopwords.words(PARAMS['preprocessing']['stopwords_lang']))
LEMMATIZER = WordNetLemmatizer()

# Module-level model state - loaded once at startup
_MODEL      = None
_VECTORIZER = None
_MODEL_NAME = "not loaded"


# ── Preprocessing ────────────────────────────────────────────────────────────

def get_wordnet_pos(tag: str):
    if tag.startswith('J'): return wordnet.ADJ
    if tag.startswith('V'): return wordnet.VERB
    if tag.startswith('N'): return wordnet.NOUN
    if tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN

def preprocess(text: str) -> str:
    text = str(text)[:PRE['max_text_length']].lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens
              if t not in STOP_WORDS and len(t) >= PRE['min_token_length']]
    tagged = pos_tag(tokens)
    return ' '.join([LEMMATIZER.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged])


# ── Model loader ─────────────────────────────────────────────────────────────

def load_model():
    global _MODEL, _VECTORIZER, _MODEL_NAME
    model_path = Path(API_CFG['model_path'])
    if not model_path.exists():
        raise RuntimeError(
            f"Model not found at {model_path}. Run: dvc repro train"
        )
    with open(model_path, 'rb') as f_in:
        checkpoint = pickle.load(f_in)
    _MODEL      = checkpoint['model']
    _VECTORIZER = checkpoint['vectorizer']
    _MODEL_NAME = checkpoint.get('model_name', 'Unknown')
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Model loaded: {_MODEL_NAME}")


# ── Lifespan (replaces deprecated on_event) ──────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(
    title="Amazon Reviews Sentiment API",
    description="MAI203 NLP project - sentiment analysis on food product reviews.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str
    review_id: Optional[str] = None

class PredictResponse(BaseModel):
    label:      str
    confidence: float
    all_scores: dict
    review_id:  Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get('/health')
def health():
    """Server health check."""
    return {"status": "ok", "model": _MODEL_NAME, "device": "cpu"}


@app.get('/classes')
def classes():
    """Return the list of output sentiment classes."""
    return {"classes": ["negative", "positive"]}


@app.post('/predict', response_model=PredictResponse)
def predict(body: PredictRequest):
    """
    Predict sentiment for a single food review text.

    Example:
        curl -X POST http://localhost:5001/predict \\
          -H "Content-Type: application/json" \\
          -d '{"text": "This coffee is absolutely amazing!"}'
    """
    if _MODEL is None or _VECTORIZER is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    if not body.text or not body.text.strip():
        raise HTTPException(status_code=422, detail="text field cannot be empty.")

    try:
        processed  = preprocess(body.text)
        features   = _VECTORIZER.transform([processed])
        label      = _MODEL.predict(features)[0]
        proba      = _MODEL.predict_proba(features)[0]
        cls_names  = _MODEL.classes_.tolist()
        all_scores = {c: round(float(p), 4) for c, p in zip(cls_names, proba)}
        confidence = round(float(max(proba)), 4)

        _log_prediction(body.text, label, confidence)

        return PredictResponse(
            label=label,
            confidence=confidence,
            all_scores=all_scores,
            review_id=body.review_id
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")


def _log_prediction(text: str, label: str, confidence: float):
    """Append each prediction to logs/inference_log.csv for drift monitoring."""
    is_new = not LOG_PATH.exists()
    with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        if is_new:
            writer.writerow(['timestamp', 'text_length', 'label', 'confidence'])
        writer.writerow([datetime.utcnow().isoformat(), len(text), label, confidence])


if __name__ == '__main__':
    uvicorn.run("src.app:app", host=API_CFG['host'], port=API_CFG['port'], reload=True)