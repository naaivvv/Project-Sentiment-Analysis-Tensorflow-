import os
import pickle
import logging
import numpy as np
from flask import Flask, render_template_string, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------------
# Setup logging (shows on Render console)
# -------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# -------------------------------------------------------
# Load model and tokenizer
# -------------------------------------------------------
MODEL_PATH = "sentiment_w2v_model.keras"
TOKENIZER_PATH = "tokenizer.pickle"

model = None
tokenizer = None

try:
    logger.info("🔍 Loading sentiment model and tokenizer...")
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as handle:
        tokenizer = pickle.load(handle)
    logger.info("✅ Model and tokenizer loaded successfully!")
except Exception as e:
    logger.error(f"❌ Error loading model or tokenizer: {e}")

# -------------------------------------------------------
# Helper function for prediction
# -------------------------------------------------------
def predict_sentiment(text):
    if not model or not tokenizer:
        return "Error: Model or tokenizer not loaded"

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    pred = model.predict(padded)
    label = "Positive 😀" if pred[0][0] > 0.5 else "Negative 😞"
    return label

# -------------------------------------------------------
# HTML Template (Tailwind + Live updates)
# -------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Sentiment Analysis</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
    body { background: #f9fafb; }
    .fade-in { animation: fadeIn 0.4s ease-in-out; }
    @keyframes fadeIn { from {opacity: 0;} to {opacity: 1;} }
</style>
</head>
<body class="flex items-center justify-center min-h-screen">
  <div class="bg-white shadow-xl rounded-2xl p-8 w-full max-w-lg text-center">
    <h1 class="text-3xl font-bold text-gray-800 mb-4">🧠 Sentiment Analysis</h1>
    <p class="text-gray-500 mb-6">Type your text below to analyze its sentiment</p>
    <textarea id="inputText" rows="4" class="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-blue-500" placeholder="Enter your sentence..."></textarea>
    <button onclick="analyzeSentiment()" class="mt-4 bg-blue-600 hover:bg-blue-700 text-white py-2 px-6 rounded-lg transition">Analyze</button>
    <div id="result" class="mt-6 text-xl font-semibold text-gray-800 fade-in"></div>
  </div>

<script>
async function analyzeSentiment() {
  const text = document.getElementById("inputText").value;
  const resultDiv = document.getElementById("result");
  resultDiv.innerHTML = "⏳ Analyzing...";
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    resultDiv.innerHTML = data.sentiment;
  } catch (err) {
    resultDiv.innerHTML = "⚠️ Error connecting to server.";
  }
}

// Auto-update if new text entered (every 2s)
setInterval(() => {
  const text = document.getElementById("inputText").value.trim();
  if (text.length > 0) analyzeSentiment();
}, 2000);
</script>
</body>
</html>
"""

# -------------------------------------------------------
# Flask Routes
# -------------------------------------------------------
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        sentiment = predict_sentiment(text)
        return jsonify({"sentiment": sentiment})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"sentiment": "Error during prediction"}), 500

# -------------------------------------------------------
# Start Flask App
# -------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 Starting Flask server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)
