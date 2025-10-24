import os
import json
import datetime
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import requests
import markdown2

# ------------------ APP INIT ------------------
app = Flask(__name__)
MODEL_PATH = "Best_Cattle_Breed.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ------------------ CLASS NAMES & BREED INFO ------------------
CLASS_NAMES = sorted([name for name in os.listdir("data/") if os.path.isdir(os.path.join("data", name))])

with open("data/breed_info.json", "r") as f:
    BREED_INFO = json.load(f)

# ------------------ UPLOAD FOLDER ------------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ------------------ GROQ API ------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️ GROQ_API_KEY not found. Set it with export GROQ_API_KEY='your_key'")

def get_groq_advice(breed):
    url = "https://api.groq.com/openai/v1/responses"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    prompt = f"""
    You are an expert veterinarian and cattle specialist.
    Give detailed advice for {breed} breed including:
    - Regular feeding
    - Seasonal feeding tips
    - Medication & health care
    - Government schemes applicable in India
    - Other useful tips for farmers
    Keep it concise, structured, and clear.
    """

    data = {
        "model": "openai/gpt-oss-20b",
        "input": prompt
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        print("Groq raw status:", response.status_code)
        print("Groq raw response:", response.text)
        result = response.json()

        output_text = ""
        for item in result.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    output_text += content.get("text", "")

        if not output_text:
            output_text = "AI advice not found in the response."

        # Convert Markdown to HTML for rendering
        return markdown2.markdown(output_text)

    except Exception as e:
        print("Groq API error:", e)
        return f"<p>AI advice not available: {str(e)}</p>"

# ------------------ HELPER FUNCTIONS ------------------
def predict_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img).astype(np.float32)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    index = np.argmax(preds[0])
    breed_name = CLASS_NAMES[index].replace("_", " ").strip()
    return breed_name, float(np.max(preds))

def get_season():
    month = datetime.datetime.now().month
    return "summer" if 4 <= month <= 9 else "winter"

def get_breed_info(breed):
    default_info = {
        "description": "Information not available",
        "feeding": {"regular":"Information not available","summer":"Information not available","winter":"Information not available"},
        "medication": "Information not available",
        "government_schemes": [],
        "other_info": "Information not available"
    }
    return BREED_INFO.get(breed, default_info)

# ------------------ ROUTES ------------------
@app.route("/", methods=["GET","POST"])
def home():
    prediction = None
    confidence = None
    image_path = None
    info = {}
    seasonal_feed = ""
    season = get_season()
    advice = ""

    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # --- Prediction ---
            breed, conf = predict_image(filepath)
            prediction = breed
            confidence = conf
            image_path = filepath

            # --- Static Info ---
            info = get_breed_info(breed)
            seasonal_feed = info.get("feeding", {}).get(season, "Information not available")

            # --- AI Advice from Groq ---
            advice = get_groq_advice(breed)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence*100,2) if confidence else None,
        image_path=image_path,
        info=info,
        seasonal_feed=seasonal_feed,
        season=season.capitalize(),
        advice=advice
    )

# ------------------ RUN APP ------------------
if __name__ == "__main__":
    app.run(debug=True)
