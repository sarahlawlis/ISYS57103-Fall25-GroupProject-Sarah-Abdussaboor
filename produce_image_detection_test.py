import torch
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np

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
            top_k_probabilities, top_k_indices = torch.topk(probs, k=3, dim=1)
            top_k_probabilities = top_k_probabilities[0].tolist()
            confidence, predicted_class_idx = torch.max(probs, dim=1)
            confidence = confidence.item()
            # predicted_class = label_map[predicted_class_idx.item()]
            predicted_class_names = [label_map[idx.item()] for idx in top_k_indices[0]]

        # Display result
        if confidence < 0.5:
            text = "No item detected"
            color = (0, 0, 255)
        else:
            # text = f"{predicted_class}: {confidence:.2f}"
            text = ", ".join(
                f"{name}: {prob:.2%}"
                for name, prob in zip(predicted_class_names, top_k_probabilities)
)

            color = (0, 255, 0)

        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Fruit & Vegetable Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Load model from hugging face and processor
    model_id = "jazzmacedo/fruits-and-vegetables-detector-36"
    # model_id = "Adriana213/vgg16-fruit-classifier"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Get label mapping from model
    label_map = model.config.id2label
    # print("Loaded labels:", label_map)

    # Run classification
    real_time_fruit_classification()

# What will it take to deploy this? Start building unit tests (ask ChatGPT), regression tests, build dev stage and prod environment, for each one
# Learnings, wwho did what, diagrams from draw.io (input data, architecture, output data)
