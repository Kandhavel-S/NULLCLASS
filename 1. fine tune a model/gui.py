import os
import cv2
import numpy as np
from tkinter import Tk, Label, Button, filedialog, Canvas, PhotoImage
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk

# Load the trained model
model_path = "age_detection_model.h5"
model = load_model(model_path)

# Function to preprocess the input image
def preprocess_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize the image
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Function to predict age
def predict_age(image_path):
    processed_image = preprocess_image(image_path)
    predicted_age = model.predict(processed_image)[0][0]  # Get the first value
    return round(predicted_age, 2)

# Function to handle file upload and prediction
def upload_and_predict():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")),
    )
    if file_path:
        # Display the uploaded image in the GUI
        img = Image.open(file_path)
        img.thumbnail((200, 200))  # Resize for GUI display
        img = ImageTk.PhotoImage(img)
        canvas.delete("all")  # Clear any previous image
        canvas.create_image(0, 0, anchor="nw", image=img)
        canvas.image = img  # Keep a reference to avoid garbage collection

        # Predict the age
        predicted_age = predict_age(file_path)
        result_label.config(text=f"Predicted Age: {predicted_age} years")

# Initialize the GUI window
root = Tk()
root.title("Age Detection GUI")
root.geometry("400x400")
root.resizable(False, False)

# GUI elements
title_label = Label(root, text="Age Detection System", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

canvas = Canvas(root, width=200, height=200, bg="gray")
canvas.pack(pady=10)

upload_button = Button(root, text="Upload Image", command=upload_and_predict, font=("Helvetica", 12))
upload_button.pack(pady=10)

result_label = Label(root, text="Predicted Age: ", font=("Helvetica", 14))
result_label.pack(pady=10)

exit_button = Button(root, text="Exit", command=root.quit, font=("Helvetica", 12))
exit_button.pack(pady=20)

# Run the GUI application
root.mainloop()
