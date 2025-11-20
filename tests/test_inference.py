import pytest
import torch
from PIL import Image
from src.inference import load_model_and_processor, predict

def test_load_model_and_processor():
    model, processor, device = load_model_and_processor()
    assert model is not None
    assert processor is not None

def test_predict_on_sample_image():
    model, processor, device = load_model_and_processor()
    # Use a dedicated test image from the test_assets directory
    img = Image.open("test_assets/banana.jpeg").convert("RGB")
    preds = predict(img, model, processor, device, top_k=2)
    assert isinstance(preds, list)
    assert len(preds) == 2
    assert all(isinstance(name, str) and isinstance(prob, float) for name, prob in preds)

# def predict(image: Image.Image, model, processor, device, top_k: int = 3):
#     """
#     Returns list of (label, prob) for top_k predictions.
#     """
#     # Ensure the image is resized and converted to RGB
#     image = image.resize((224, 224)).convert("RGB")

#     # Enable padding to handle tensor creation issues
#     inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.nn.functional.softmax(outputs.logits, dim=1)
#         topk_probs, topk_idxs = torch.topk(probs, k=top_k, dim=1)
#         topk_probs = topk_probs[0].tolist()
#         idxs = [int(i) for i in topk_idxs[0]]
#         label_map = model.config.id2label
#         names = [label_map[i] for i in idxs]
#     return list(zip(names, topk_probs))