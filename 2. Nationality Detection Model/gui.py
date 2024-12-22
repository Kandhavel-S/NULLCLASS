import os
import cv2
import numpy as np
from tkinter import Tk, Label, Button, filedialog, Canvas, messagebox
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import webcolors

# Load the trained models
age_model = load_model("age_detection_model.h5")
emotion_model = load_model("emotion_detection_model.h5")
nationality_model = load_model("nationality_detection_model.h5")

# Define emotion labels
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['US', 'African', 'Asian', 'Indian', 'Others'])  # Update with your classes

# Function to preprocess the input image for age prediction
def preprocess_image(image_path, target_size=(48, 48)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0  # Normalize the image
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Function to preprocess the input image for emotion prediction
def preprocess_emotion_image(image_path, target_size=(48, 48)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    img = cv2.resize(img, target_size)  # Resize to model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Function to preprocess the input image for nationality prediction
def preprocess_nationality_image(image_path, target_size=(48, 48)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to predict age
def predict_age(image_path):
    processed_image = preprocess_image(image_path)
    predicted_age = age_model.predict(processed_image)[0][0] * 100  # Get the predicted age
    return round(predicted_age, 2)

# Function to predict emotion
def predict_emotion(image_path):
    processed_image = preprocess_emotion_image(image_path)
    predictions = emotion_model.predict(processed_image)[0]
    predicted_label = EMOTION_LABELS[np.argmax(predictions)]  # Get label with highest probability
    return predicted_label

# Function to predict nationality
def predict_nationality(image_path):
    processed_image = preprocess_nationality_image(image_path)
    predictions = nationality_model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    nationality = label_encoder.inverse_transform([predicted_class])[0]
    return nationality

# Function to predict dress color (using dominant color)
def predict_dress_color(image_path):
    img = cv2.imread(image_path)
    # Resize image to speed up processing
    img = cv2.resize(img, (100, 100))
    # Convert image to RGB and reshape it to a list of pixels
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape((-1, 3))

    # Calculate the most common color (dominant color)
    counter = Counter(map(tuple, pixels))
    most_common = counter.most_common(1)[0][0]  # Get the most common color
    return most_common

# Function to convert RGB to color name
def rgb_to_name(rgb):
    try:
        return webcolors.rgb_to_name(rgb)
    except ValueError:
        return "Unknown Color"

# Function to handle file upload and prediction
def upload_and_predict():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*")],
    )
    if file_path:
        try:
            # Display the uploaded image in the GUI
            img = Image.open(file_path)
            img.thumbnail((200, 200))  # Resize for GUI display
            img = ImageTk.PhotoImage(img)
            canvas.delete("all")  # Clear any previous image
            canvas.create_image(0, 0, anchor="nw", image=img)
            canvas.image = img  # Keep a reference to avoid garbage collection

            # Predict nationality
            nationality = predict_nationality(file_path)
            nationality_label.config(text=f"Predicted Nationality: {nationality}")

            # Predict emotion
            predicted_emotion = predict_emotion(file_path)

            # Handle nationality-specific predictions
            if nationality == "Indian":
                predicted_age = predict_age(file_path)
                predicted_dress_color = predict_dress_color(file_path)  # Get dress color for Indian
                dress_color_name = rgb_to_name(predicted_dress_color)  # Convert RGB to color name
                result_label.config(text=f"Age: {predicted_age} years\nDress Color: {dress_color_name}\nEmotion: {predicted_emotion}")
            elif nationality == "US":
                predicted_age = predict_age(file_path)
                result_label.config(text=f"Age: {predicted_age} years\nEmotion: {predicted_emotion}")
            elif nationality == "African":
                predicted_dress_color = predict_dress_color(file_path)  # Get dress color for African
                dress_color_name = rgb_to_name(predicted_dress_color)  # Convert RGB to color name
                result_label.config(text=f"Dress Color: {dress_color_name}\nEmotion: {predicted_emotion}")
            else:
                result_label.config(text=f"Nationality: {nationality}\nEmotion: {predicted_emotion}")

        except Exception as e:
            result_label.config(text=f"Error: {e}")

# Initialize the GUI window
root = Tk()
root.title("Image Prediction System")
root.geometry("500x600")
root.resizable(False, False)

# GUI elements
title_label = Label(root, text="Image Prediction System", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

canvas = Canvas(root, width=200, height=200, bg="gray")
canvas.pack(pady=10)

upload_button = Button(root, text="Upload Image", command=upload_and_predict, font=("Helvetica", 12))
upload_button.pack(pady=10)

# Result labels
nationality_label = Label(root, text="Predicted Nationality: ", font=("Helvetica", 14))
nationality_label.pack(pady=5)

result_label = Label(root, text="Result: ", font=("Helvetica", 14))
result_label.pack(pady=10)

exit_button = Button(root, text="Exit", command=root.quit, font=("Helvetica", 12))
exit_button.pack(pady=20)

# Run the GUI application
root.mainloop()
