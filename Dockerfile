# Dockerfile - MAI203 Amazon Fine Food Reviews NLP Pipeline
FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data at build time so runtime has no network dependency
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); \
    nltk.download('stopwords'); nltk.download('wordnet'); \
    nltk.download('averaged_perceptron_tagger')"

# Copy source and config
COPY src/ ./src/
COPY params.yaml .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
