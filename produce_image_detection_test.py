import torch
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np

# Load model from hugging face and processor
model_id = "jazzmacedo/fruits-and-vegetables-detector-36"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Get label mapping from model
label_map = model.config.id2label
print("Loaded labels:", label_map)

def real_time_fruit_classification():
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        # Preprocess for the model
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted_class_idx = torch.max(probs, dim=1)
            confidence = confidence.item()
            predicted_class = label_map[predicted_class_idx.item()]

        # Display result
        if confidence < 0.5:
            text = "No item detected"
            color = (0, 0, 255)
        else:
            text = f"{predicted_class}: {confidence:.2f}"
            color = (0, 255, 0)

        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Fruit & Vegetable Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run classification
real_time_fruit_classification()