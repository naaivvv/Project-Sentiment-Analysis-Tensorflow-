import os, re, requests, numpy as np, tensorflow as tf, pickle
from gensim.models import KeyedVectors
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- SETTINGS ---
MODEL_PATH = "word2vec.model"
MODEL_URL = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
MODEL_BIN = "GoogleNews-vectors-negative300.bin.gz"

# --- DOWNLOAD PRETRAINED WORD2VEC IF NOT FOUND ---
def ensure_word2vec():
    if not os.path.exists(MODEL_BIN):
        print("⏬ Downloading Word2Vec pretrained vectors (GoogleNews)... This may take a while.")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_BIN, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("✅ Download complete!")

# --- PRELOAD MODELS ---
print("🚀 Initializing app...")

ensure_word2vec()
print("📦 Loading Word2Vec (this may take 1–2 minutes)...")
w2v_model = KeyedVectors.load_word2vec_format(MODEL_BIN, binary=True, limit=300000)  # limit for memory

print("✅ Loading Keras model and tokenizer...")
model = tf.keras.models.load_model("sentiment_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# --- PREPROCESSING ---
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    words = text.split()
    vectors = [w2v_model[word] for word in words if word in w2v_model]
    if not vectors:
        return np.zeros((1, 300))
    return np.mean(vectors, axis=0).reshape(1, -1)

# --- ROUTES ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    vec = preprocess_text(text)
    score = float(model.predict(vec)[0][0])
    if score > 0.65:
        sentiment = "Positive 😊"
    elif score < 0.35:
        sentiment = "Negative 😠"
    else:
        sentiment = "Neutral 😐"
    return jsonify({"sentiment": sentiment, "score": score})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
