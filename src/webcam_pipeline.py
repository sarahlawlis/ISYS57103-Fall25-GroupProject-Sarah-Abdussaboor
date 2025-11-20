from src.inference import load_model_and_processor, predict
import cv2
from PIL import Image

def run_webcam_pipeline():
    # Load the model and processor
    model, processor, device = load_model_and_processor()

    # Start webcam feed
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        # Convert frame to PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        # Predict using the model
        predictions = predict(image, model, processor, device, top_k=3)

        # Display predictions on the frame
        text = ", ".join([f"{label}: {prob:.2%}" for label, prob in predictions])
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Fruit & Vegetable Classification", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()