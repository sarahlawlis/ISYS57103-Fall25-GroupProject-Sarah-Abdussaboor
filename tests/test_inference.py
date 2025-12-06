from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.inference import load_model_and_processor, predict


class DummyModel:
    def __init__(self):
        self.config = SimpleNamespace(id2label={0: "apple", 1: "banana"})

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, pixel_values):
        logits = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
        return SimpleNamespace(logits=logits)


class DummyProcessor:
    def __call__(self, images, return_tensors=None, padding=True):
        return {"pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32)}


@patch("src.inference.AutoModelForImageClassification")
@patch("src.inference.AutoImageProcessor")
def test_load_model_and_processor_prefers_local_cache(mock_image_processor, mock_model_cls, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    mock_image_processor.from_pretrained.return_value = DummyProcessor()
    dummy_model = DummyModel()
    mock_model_cls.from_pretrained.return_value = dummy_model

    model, processor, device = load_model_and_processor(model_path=str(model_dir))

    mock_image_processor.from_pretrained.assert_called_once_with(str(model_dir), local_files_only=True)
    mock_model_cls.from_pretrained.assert_called_once_with(str(model_dir), local_files_only=True)
    assert model is dummy_model
    assert isinstance(processor, DummyProcessor)
    assert device is not None


def test_predict_on_sample_image_without_network():
    processor = DummyProcessor()
    model = DummyModel()
    device = torch.device("cpu")

    img = Image.new("RGB", (224, 224), (0, 0, 0))
    preds = predict(img, model, processor, device, top_k=2)

    assert isinstance(preds, list)
    assert len(preds) == 2
    assert preds[0][0] == "apple"
    assert all(isinstance(name, str) and isinstance(prob, float) for name, prob in preds)
