# MAI203 - Amazon Fine Food Reviews NLP Pipeline

End-to-end NLP pipeline: preprocessing, sentiment analysis, text classification, and BM25 information retrieval on 50,000 Amazon food reviews.

**Course:** MAI203 Introduction to Natural Language Processing, Seneca Polytechnic  
**Instructor:** Prof. Junwei Huang  
**Team:** Devreet (macOS), Arushi (Windows)

---

## Setup

```bash
git clone <repo-url>
cd amazon-nlp-project
pip install -r requirements.txt
```

Download the dataset from https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews  
Place `Reviews.csv` in `data/raw/Reviews.csv`.

---

## DVC Pipeline

```bash
# Run full pipeline (prepare -> train -> evaluate -> monitor)
dvc repro

# Run single stage
dvc repro prepare
dvc repro train
dvc repro evaluate

# Push data and artifacts to remote
dvc push

# After every git pull, always run:
dvc pull
```

---

## Run notebooks in order

```
notebooks/notebook_A_preprocessing.ipynb
notebooks/notebook_B_sentiment.ipynb
notebooks/notebook_C_classification.ipynb
notebooks/notebook_D_bm25.ipynb
notebooks/notebook_E_report.ipynb
```

---

## API

**macOS (Devreet) - port 5001 because port 5000 is blocked by AirPlay:**
```bash
uvicorn src.app:app --reload --port 5001
# Docs: http://localhost:5001/docs
```

**Windows (Arushi):**
```powershell
uvicorn src.app:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

**Predict (macOS):**
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This coffee is absolutely amazing!"}'
```

**Predict (Windows PowerShell):**
```powershell
Invoke-RestMethod -Uri http://localhost:8000/predict `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text": "This coffee is absolutely amazing!"}'
```

**Response:**
```json
{"label": "positive", "confidence": 0.97, "all_scores": {"negative": 0.03, "positive": 0.97}}
```

---

## Docker

```bash
# Build
docker build -t amazon-nlp-api .

# Run (macOS - port 5001)
docker compose up

# Run (Windows - port 8000, edit docker-compose.yml first)
docker compose up
```

---

## MLflow

```bash
# View experiment results (macOS)
mlflow ui --port 5001

# View experiment results (Windows)
mlflow ui --port 8080
```

---

## Tests

```bash
pytest tests/test_api.py -v
```

---

## Git Rules

- Never `git add .` - always stage specific files
- Never commit data files, model files, mlruns, or API keys
- Always commit `dvc.lock` after every `dvc repro`
- Always run `dvc push` after `dvc repro`
- Write tests on the same branch as the feature
- 2 approvals required before merging any PR
- `dvc pull` after every `git pull`, always

---

## Repo Structure

```
amazon-nlp-project/
  data/
    raw/               (DVC-tracked, not in Git)
    processed/         (DVC-tracked, not in Git)
  notebooks/
    notebook_A_preprocessing.ipynb
    notebook_B_sentiment.ipynb
    notebook_C_classification.ipynb
    notebook_D_bm25.ipynb
    notebook_E_report.ipynb
  src/
    prepare.py
    train.py
    evaluate.py
    app.py
    monitor.py
  tests/
    test_api.py
  results/
    models/            (not in Git)
    metrics/
    plots/
  reports/
    drift/
  logs/                (not in Git)
  .github/workflows/ci.yml
  dvc.yaml
  dvc.lock
  params.yaml
  requirements.txt
  Dockerfile
  docker-compose.yml
  model_card.md
  README.md
```
