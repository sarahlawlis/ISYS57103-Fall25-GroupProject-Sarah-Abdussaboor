# Computer Vision for Self-Checkout Item Prediction and Fraud Detection
## ISYS 57103 - Sarah Lawlis and Abdussaboor Muhammad
---
### Project Proposal

As retailers adopt self-checkout systems to streamline operations, a new challenge emerges: detecting and preventing fraud when produce and other untagged items are weighed and entered manually. Customers may intentionally or unintentionally mislabel items, leading to revenue loss and inventory inaccuracies. Inspired by Walmart’s use of AI (pictured in Figure 1) to suggest likely produce matches, this project will explore how computer vision models can identify produce at checkout and cross-verify it with customer selections. By creating a labeled item dataset and training a vision model to recognize common fruits and vegetables, we aim to replicate and extend existing self-checkout fraud detection systems. This project will evaluate the accuracy of classification across similar-looking items (e.g., russet vs. sweet potatoes) and consider tradeoffs between real-time performance and reliability. Ultimately, our work will demystify how AI-powered computer vision is being deployed in everyday retail environments and highlight both its promise and limitations in reducing fraud while maintaining customer convenience.

![](assets/self_checkout_potatoes.jpeg)
*Figure 1: Self-Checkout Item Detection Powered by AI at Walmart*

## Running locally

Use the project venv (`./env`) to ensure the correct deps:

```bash
source env/bin/activate
python -m pytest
```

Run the webcam demo with optional env overrides:

```bash
MODEL_PATH=/models/fruits-and-vegetables-detector-36 \
CAMERA_INDEX=0 \
python -m scripts.run_webcam
```

## Caching the model for offline/edge use

Download the Hugging Face model once, then bake or mount it:

```bash
python scripts/cache_model.py --dest /models/fruits-and-vegetables-detector-36 \
  --model-id jazzmacedo/fruits-and-vegetables-detector-36
```

Point the app at the cached copy via `MODEL_PATH` (or `--model-path`).

## Make targets

Convenience shortcuts (override variables as needed):

```bash
make cache-model MODEL_DIR=./models/fruits-and-vegetables-detector-36 MODEL_ID=jazzmacedo/fruits-and-vegetables-detector-36
make test
make build-image DOCKER_IMAGE=webcam-classifier
make run-container MODEL_DIR=./models/fruits-and-vegetables-detector-36 DOCKER_IMAGE=webcam-classifier
```

## Container build

Build a runtime image (expects the cached model at `/models/...`—copy or mount it):

```bash
docker build -t webcam-classifier .
docker run --rm \ 
  -e MODEL_PATH=/models/fruits-and-vegetables-detector-36 \ 
  -v /host/models/fruits-and-vegetables-detector-36:/models/fruits-and-vegetables-detector-36:ro \ 
  --device /dev/video0 \ 
  webcam-classifier
```
