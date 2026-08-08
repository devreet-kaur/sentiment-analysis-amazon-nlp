# MAI203: Amazon Fine Food Reviews NLP Pipeline

End to end NLP pipeline: preprocessing, sentiment analysis, text classification, and BM25 information retrieval on 50,000 Amazon food reviews.

**Course:** MAI203 Introduction to Natural Language Processing, Seneca Polytechnic
**Instructor:** Prof. Junwei Huang
**Team:** Devreet Kaur and Arushi Anand

---

## For the Grader: Fastest Path to a Working Demo

This section exists so you can test the live API and read the notebooks without training anything yourself. If any step below fails, skip to the "If dvc pull fails" section further down, it gives a fallback that works with zero setup.

```bash
git clone https://github.com/devreet-kaur/sentiment-analysis-amazon-nlp.git
cd sentiment-analysis-amazon-nlp
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
dvc pull
```

`dvc pull` downloads the trained model, the cleaned dataset, and all evaluation plots from our DagsHub remote, without needing a Kaggle account or retraining anything. If this succeeds, `results/models/best_model.pkl` will exist and the API below will work immediately.

Then start the API:
```bash
uvicorn src.app:app --reload --port 5001
```

Open a second terminal and test it:
```bash
curl http://localhost:5001/health
curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d '{"text": "This coffee is amazing!"}'
```

Or open `http://localhost:5001/docs` in a browser for an interactive Swagger UI, no curl needed.

### If `dvc pull` fails or asks for credentials

Our DagsHub remote may require authentication depending on how it is configured. If `dvc pull` fails:

1. Email us and we will grant read access, or
2. Run the full pipeline from raw data instead (takes 5 to 10 minutes):
   ```bash
   # Download Reviews.csv from https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
   # Place it at data/raw/Reviews.csv
   dvc repro
   ```
   This runs all four pipeline stages (prepare, train, evaluate, monitor) and produces a working model.

---

## What This Project Does

Five notebooks, each mapped to a rubric section:

| Notebook | Rubric Part | What it does |
|---|---|---|
| `notebook_A_preprocessing.ipynb` | Part A (20%) | Cleans 568,454 raw reviews to a 50,000 row sample, 7 step text cleaning pipeline |
| `notebook_B_sentiment.ipynb` | Part B (20%) | Sentiment analysis. Compares VADER, Logistic Regression, DistilBERT |
| `notebook_C_classification.ipynb` | Part C (30%) | 3 class text classification (positive, negative, neutral). Naive Bayes vs Logistic Regression |
| `notebook_D_bm25.ipynb` | Part D (20%) | BM25 search engine over all 50,000 reviews, full IR evaluation |
| `notebook_E_report.ipynb` | Part E (10%) | Final report pulling results from B, C, D. Ethical considerations included |

Run them in order, A through E. Each depends on `data/processed/cleaned_reviews.csv`, produced by Notebook A (or by `dvc repro prepare`).

---

## Full DVC Pipeline

```bash
dvc repro              # runs all 4 stages: prepare -> train -> evaluate -> monitor
dvc repro prepare      # single stage
dvc repro train
dvc repro evaluate
dvc repro monitor
dvc push               # push new artifacts to the DagsHub remote
dvc pull               # after every git pull, always run this
```

---

## Live API

```bash
uvicorn src.app:app --reload --port 5001
```

The port is set in `params.yaml` under `api.port`, defaulting to 8000, but our examples below use 5001. If port 5001 is already in use on your machine, edit `params.yaml` and restart.

**Endpoints:**

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Confirms server and model are loaded |
| `/classes` | GET | Lists the possible output labels |
| `/predict` | POST | Returns sentiment label and confidence for a review |
| `/docs` | GET | Interactive Swagger UI, test everything in a browser |

**Example request:**
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This coffee is absolutely amazing!"}'
```

**Example response:**
```json
{"label": "positive", "confidence": 0.9979, "all_scores": {"negative": 0.0021, "positive": 0.9979}, "review_id": null}
```

**Windows PowerShell equivalent:**
```powershell
Invoke-RestMethod -Uri http://localhost:5001/predict `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text": "This coffee is absolutely amazing!"}'
```

---

## Docker

```bash
docker build -t amazon-nlp-api .
docker compose up -d
```

The API is available at `http://localhost:5001` by default. `docker-compose.yml` maps host port 5001 to the container's internal port 8000. If port 5001 is taken on your machine, edit the `ports` line in `docker-compose.yml` before running, no other changes needed on either macOS or Windows.

**Verify it's running:**
```bash
curl http://localhost:5001/health
docker compose logs -f api
```

**Stop it:**
```bash
docker compose down
```

---

## Testing

```bash
pytest tests/test_api.py -v
```

13 tests covering health, classes, and predict endpoints (valid inputs, missing fields, empty text, HTML in review text). All should pass once `results/models/best_model.pkl` exists (via `dvc pull` or `dvc repro train`).

---

## Model Monitoring (Evidently Drift Detection)

Our pipeline tracks whether live prediction traffic starts looking different from the training data, using Evidently.

**To generate drift data, make some API calls first:**
```bash
for i in {1..15}; do
  curl -s -X POST http://localhost:5001/predict \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"Sample review number $i, this product is decent\"}" > /dev/null
done
```

Each call appends a row to `logs/inference_log.csv`. You need at least 10 rows before drift detection runs meaningfully.

**Then run the monitor stage:**
```bash
dvc repro monitor
```

**Open the generated report:**
```bash
open reports/drift/drift_report.html        # macOS
start reports/drift/drift_report.html       # Windows
```

The report compares the distribution of review length and predicted label between training data and live traffic, flagging drift if it exceeds the threshold set in `params.yaml` (`monitor.drift_threshold`, default 0.15).

---

## Experiment Tracking (MLflow)

```bash
mlflow ui --port 5001
```
Open `http://localhost:5001` to see training runs, hyperparameters, and metrics for every model trained during `dvc repro train`.

If port 5001 is already used by the API, run MLflow on a different port:
```bash
mlflow ui --port 5002
```

---

## Repo Structure

```
sentiment-analysis-amazon-nlp/
  data/
    raw/                 (DVC tracked, not in Git. Reviews.csv lives here)
    processed/           (DVC tracked, not in Git. cleaned_reviews.csv output)
  notebooks/
    notebook_A_preprocessing.ipynb
    notebook_B_sentiment.ipynb
    notebook_C_classification.ipynb
    notebook_D_bm25.ipynb
    notebook_E_report.ipynb
  src/
    prepare.py            DVC stage: data cleaning
    train.py               DVC stage: model training + MLflow logging
    evaluate.py            DVC stage: metrics + plots
    app.py                 FastAPI prediction endpoint
    monitor.py              Evidently drift detection
  tests/
    test_api.py            13 pytest tests
  results/
    models/                (not in Git, best_model.pkl lives here after training)
    metrics/                train_metrics.json, eval_metrics.json
    plots/                  confusion_matrix.png, pr_curves.png, bm25_metrics.png
  reports/
    drift/                  drift_report.html output
  logs/                     (not in Git, inference_log.csv for monitoring)
  .github/workflows/ci.yml  lint + test + Docker smoke test, runs on every push
  dvc.yaml                  4 stage DVC pipeline definition
  dvc.lock                  pinned pipeline state
  params.yaml               all hyperparameters, never hardcoded in source
  requirements.txt
  Dockerfile
  docker-compose.yml
  model_card.md             filled with real evaluation metrics
  README.md
```

---

## Working Rules (for team reference)

- Never `git add .`, always stage specific files
- Never commit data files, model files, `mlruns/`, or API keys
- Always commit `dvc.lock` after every `dvc repro`
- Always run `dvc push` after `dvc repro`
- Run `dvc pull` after every `git pull`, always, no exceptions
- 2 approvals required before merging any PR, no exceptions