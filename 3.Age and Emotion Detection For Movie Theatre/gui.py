import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime

# Load the trained models
age_model = load_model("age_detection_model.h5")
emotion_model = load_model("emotion_detection_model.h5")

# Define emotion labels
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Initialize data storage
data = []

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to preprocess the input image for age prediction
def preprocess_image(image, target_size=(48, 48)):
    img = cv2.resize(image, target_size)
    img = img.astype('float32') / 255.0  # Normalize the image
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Function to preprocess the input image for emotion prediction
def preprocess_emotion_image(image, target_size=(48, 48)):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for emotion detection
    img = cv2.resize(img, target_size)  # Resize to model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Function to predict age
def predict_age(image):
    processed_image = preprocess_image(image)
    predicted_age = age_model.predict(processed_image)[0][0] * 100  # Get the predicted age and scale it
    return round(predicted_age, 2)

# Function to predict emotion
def predict_emotion(image):
    processed_image = preprocess_emotion_image(image)
    predictions = emotion_model.predict(processed_image)[0]
    predicted_label = EMOTION_LABELS[np.argmax(predictions)]  # Get label with highest probability
    return predicted_label

# Function to save data to CSV
def save_data_to_csv(age, emotion, timestamp):
    data.append([age, emotion, timestamp])
    df = pd.DataFrame(data, columns=["Age", "Emotion", "Entry Time"])
    df.to_csv("age_emotion_data.csv", index=False)

# Function to process the webcam feed for real-time detection
def process_video():
    cap = cv2.VideoCapture(0)  # Start webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            predicted_age = predict_age(face)

            # If the predicted age is valid (13 <= age <= 60)
            if 13 <= predicted_age <= 60:
                predicted_emotion = predict_emotion(face)
                cv2.putText(frame, f"Age: {predicted_age} Emotion: {predicted_emotion}", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Save data to CSV
                timestamp = datetime.now()
                save_data_to_csv(predicted_age, predicted_emotion, timestamp)
            else:
                # If the age is invalid (less than 13 or greater than 60), mark with red rectangle
                cv2.putText(frame, "Not allowed", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Age and Emotion Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the video processing function
process_video()
