from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

def load_model_and_processor(model_id: str = "jazzmacedo/fruits-and-vegetables-detector-36", device=None):
    """
    Returns (model, processor, device)
    """
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
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