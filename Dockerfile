FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Minimal system deps for OpenCV headless video + display
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Point to a cached model directory baked into the image or mounted at runtime
ENV MODEL_PATH=/models/fruits-and-vegetables-detector-36 \
    CAMERA_INDEX=0 \
    DECISION_SECONDS=7.0 \
    CONFIDENCE_THRESHOLD=0.95 \
    HISTORY_SIZE=30

# Expect the cached model to be available at MODEL_PATH. Mount or COPY it in your build.
CMD ["python", "scripts/run_webcam.py", "--model-path", "/models/fruits-and-vegetables-detector-36"]
