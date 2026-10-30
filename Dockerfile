FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/root/.cache
ENV HF_HOME=/root/.cache/huggingface
ENV WHISPER_MODEL=tiny

# Install system dependencies including ffmpeg for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install lightweight CPU-only PyTorch first
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper and Sentence-BERT models at image build time
# This eliminates runtime network downloads and prevents out-of-memory crashes on Render free tier
RUN python -c "import whisper; whisper.load_model('tiny')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY . .

# Set default PORT if not set by host
ENV PORT=8501
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

# Start Streamlit dynamically binding to Render's $PORT
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false"]
