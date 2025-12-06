import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

DEFAULT_MODEL_ID = "jazzmacedo/fruits-and-vegetables-detector-36"
MODEL_PATH_ENV = "MODEL_PATH"
MODEL_ID_ENV = "MODEL_ID"


def load_model_and_processor(model_id: str | None = None, model_path: str | None = None, device=None):
    """
    Returns (model, processor, device).

    Prefers a locally cached model directory when provided (via argument or
    the MODEL_PATH env var). Falls back to the HF model id (argument or
    MODEL_ID env var) if no path is given.
    """

    resolved_model_path = model_path or os.getenv(MODEL_PATH_ENV)
    resolved_model_id = model_id or os.getenv(MODEL_ID_ENV, DEFAULT_MODEL_ID)

    if resolved_model_path:
        path = Path(resolved_model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model path '{path}' not found. Provide a valid cached model directory via argument or {MODEL_PATH_ENV}."
            )
        source = str(path)
        local_files_only = True
    else:
        source = resolved_model_id
        local_files_only = False

    processor = AutoImageProcessor.from_pretrained(source, local_files_only=local_files_only)
    model = AutoModelForImageClassification.from_pretrained(source, local_files_only=local_files_only)

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)
    model.eval()
    return model, processor, device


def predict(image: Image.Image, model, processor, device, top_k: int = 3):
    image = image.resize((224, 224)).convert("RGB")

    # Disable auto-tensor-conversion (this is the key fix)
    inputs = processor(images=image, return_tensors=None, padding=True)

    # Normalize pixel_values regardless of source (webcam or file)
    pixel_values = inputs["pixel_values"]

    # If pixel_values is list[list], stack
    if isinstance(pixel_values, list):
        pixel_values = np.stack(pixel_values)  # shape (1,3,224,224)

    # Convert to tensor
    pixel_values = torch.tensor(pixel_values, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        probs = torch.softmax(outputs.logits, dim=1)
        top_probs, top_idxs = torch.topk(probs, k=top_k, dim=1)

    label_map = model.config.id2label
    names = [label_map[idx.item()] for idx in top_idxs[0]]
    probs = top_probs[0].tolist()

    return list(zip(names, probs))
