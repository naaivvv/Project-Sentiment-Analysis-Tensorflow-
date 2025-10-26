import os
import re
import requests
import numpy as np
import tensorflow as tf
import pickle
from gensim.models import KeyedVectors
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
W2V_PATH = "GoogleNews-vectors-negative300.bin"
W2V_URL = "https://huggingface.co/stanfordnlp/word2vec-GoogleNews-vectors/resolve/main/GoogleNews-vectors-negative300.bin"

# --- DOWNLOAD WORD2VEC MODEL IF NOT PRESENT ---
def download_word2vec():
    if not os.path.exists(W2V_PATH):
        print("⏬ Downloading Google Word2Vec pretrained model (~1.5GB)...")
        with requests.get(W2V_URL, stream=True) as r:
            r.raise_for_status()
            with open(W2V_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("✅ Download complete!")

# --- LOAD MODELS ---
print("🚀 Starting up the app...")

# Download pretrained vectors if needed
download_word2vec()

# Load tokenizer and models
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
w2v_model = KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)
model = tf.keras.models.load_model("sentiment_model.keras")

# --- TEXT PREPROCESSING ---
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().split()
    vectors = [w2v_model[word] for word in text if word in w2v_model]
    return np.mean(vectors, axis=0).reshape(1, -1) if vectors else np.zeros((1, 300))

# --- ROUTES ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    processed = preprocess_text(text)
    prediction = model.predict(processed)
    sentiment = "Positive 😊" if prediction > 0.6 else "Negative 😞" if prediction < 0.4 else "Neutral 😐"
    return jsonify({"sentiment": sentiment, "score": float(prediction)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
