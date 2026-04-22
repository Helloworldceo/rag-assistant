FROM python:3.11-slim

WORKDIR /app

# System deps needed by sentence-transformers / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Pre-download the HuggingFace embedding model so the first request is fast.
# This bakes the ~90 MB model into the image layer.
RUN python -c "\
from langchain_community.embeddings import HuggingFaceEmbeddings; \
HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
