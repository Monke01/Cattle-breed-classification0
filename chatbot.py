import os
import numpy as np
import cv2
import tensorflow as tf
import warnings
from tkinter import Tk, Label, Button, filedialog, messagebox
from PIL import Image, ImageTk

warnings.filterwarnings("ignore")

# ------------ CONSTANTS -----------------
MODEL_PATH = "Best_Cattle_Breed.h5"   # Your pretrained model
DATA_DIR = "data/"                     # Folder with class names

# -------------------- LOAD MODEL --------------------------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    messagebox.showerror("Error", f"Error loading model: {e}")
    exit()

CLASS_NAMES = sorted(os.listdir(DATA_DIR))
print(f"Loaded {len(CLASS_NAMES)} classes: {CLASS_NAMES}")

# -------------------- IMAGE PREPROCESS --------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid path.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# ---------------- PREDICTION FUNCTION ------------------------
def predict_image(image_path):
    try:
        img = preprocess_image(image_path)
        preds = model.predict(img)
        class_id = np.argmax(preds[0])
        confidence = preds[0][class_id] * 100
        return CLASS_NAMES[class_id], confidence
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None, None

# ------------------ GUI CALLBACKS ----------------------------
def browse_image():
    file_path = filedialog.askopenfilename(
        title="Select Cattle Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    ) 
    if file_path:
        image = Image.open(file_path)
        image = image.resize((300, 300))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo

        predicted_class, confidence = predict_image(file_path)
        if predicted_class:
            result_label.config(
                text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%"
            )

# --------------------- GUI SETUP -----------------------
root = Tk()
root.title("Cattle Breed Classifier")
root.geometry("400x500")

Label(root, text="Indian Cattle Breed Classifier", font=("Arial", 16)).pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

browse_btn = Button(root, text="Select Cattle Image", command=browse_image)
browse_btn.pack(pady=20)

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
