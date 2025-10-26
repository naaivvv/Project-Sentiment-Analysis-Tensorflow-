import os
import pickle
import logging
import numpy as np
from flask import Flask, render_template_string, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------------
# Logging setup (appears in Render console)
# -------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# -------------------------------------------------------
# Load pre-trained model and tokenizer
# -------------------------------------------------------
MODEL_PATH = "sentiment_w2v_model.keras"
TOKENIZER_PATH = "tokenizer.pickle"

model = None
tokenizer = None

def load_resources():
    global model, tokenizer
    if model is None or tokenizer is None:
        import gc
        gc.collect()
        logger.info("🧠 Loading Keras model and tokenizer (lazy load)...")
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as handle:
            tokenizer = pickle.load(handle)
        logger.info("✅ Model ready.")

@app.before_request
def before_request():
    load_resources()


# -------------------------------------------------------
# Prediction Function
# -------------------------------------------------------
def predict_sentiment(text):
    if not model or not tokenizer:
        return {"label": "Error", "probability": 0.0}
    try:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=100)
        prob = float(model.predict(padded, verbose=0)[0][0])
        label = (
            "Positive" if prob >= 0.65 else
            "Negative" if prob <= 0.35 else
            "Neutral"
        )
        return {"label": label, "probability": prob}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"label": "Error", "probability": 0.0}

# -------------------------------------------------------
# Tailwind Chatbox HTML Template
# -------------------------------------------------------
HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Sentiment Chatbox</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb { background-color: rgba(0,0,0,0.2); border-radius: 4px; }
</style>
</head>
<body class="bg-gradient-to-br from-blue-100 to-indigo-100 min-h-screen flex flex-col justify-between items-center p-4">
  <div class="w-full max-w-2xl bg-white rounded-2xl shadow-2xl p-6 flex flex-col flex-grow">
    <h2 class="text-3xl font-semibold text-center mb-5 text-indigo-700">💬 Sentiment Chatbox</h2>
    <div id="chat" class="h-96 overflow-y-auto border border-gray-200 rounded-xl p-4 bg-gray-50 space-y-3 shadow-inner flex-grow"></div>
    <div class="flex gap-2 mt-4">
      <input id="msg" class="flex-1 border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-indigo-400 outline-none" placeholder="Type a message..." onkeydown="if(event.key==='Enter'){send();}" />
      <button onclick="send()" class="bg-indigo-600 hover:bg-indigo-700 text-white px-5 py-2 rounded-lg font-medium shadow transition transform hover:scale-105">Send</button>
    </div>
  </div>

  <footer class="mt-6 text-center text-sm text-gray-600">
    <p>Created by <strong>Edwin P. Bayog Jr.</strong> — BSCpE 4-A | Endterm Exam 2025</p>
  </footer>

<script>
async function send(){
  const msgInput = document.getElementById('msg');
  const msg = msgInput.value.trim();
  if(!msg) return;

  const chat = document.getElementById('chat');

  // user bubble
  const userDiv = document.createElement('div');
  userDiv.innerHTML = '<div class="text-right"><span class="inline-block bg-indigo-100 text-indigo-900 px-4 py-2 rounded-xl">'+msg+'</span></div>';
  chat.appendChild(userDiv);
  chat.scrollTop = chat.scrollHeight;

  msgInput.value = '';

  // loading text
  const loader = document.createElement('div');
  loader.id = "loader";
  loader.innerHTML = '<div class="text-left text-gray-500 italic">Analyzing...</div>';
  chat.appendChild(loader);
  chat.scrollTop = chat.scrollHeight;

  // send request
  const resp = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: msg})
  });

  const data = await resp.json();
  chat.removeChild(loader);

  const score = parseFloat(data.probability);
  let label = data.label;
  let emoji = '';
  if (label === 'Positive') emoji = '😄';
  else if (label === 'Negative') emoji = '😡';
  else emoji = '😐';

  const botDiv = document.createElement('div');
  botDiv.innerHTML = '<div class="text-left"><span class="inline-block bg-gray-200 text-gray-900 px-4 py-2 rounded-xl">Sentiment: <strong>'+label+'</strong> '+emoji+' ('+score.toFixed(3)+')</span></div>';
  chat.appendChild(botDiv);
  chat.scrollTop = chat.scrollHeight;
}
</script>
</body>
</html>"""

# -------------------------------------------------------
# Flask Routes
# -------------------------------------------------------
@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.json.get("text", "")
        result = predict_sentiment(text)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction route error: {e}")
        return jsonify({"label": "Error", "probability": 0.0}), 500

# -------------------------------------------------------
# Entry Point
# -------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 Starting Flask server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)
