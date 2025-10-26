import tensorflow as tf
import pickle, re
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model & tokenizer
model = tf.keras.models.load_model('sentiment_w2v_model.keras')
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

MAXLEN = 100

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data.get('text', '')
    t = clean_text(text)
    seq = pad_sequences(tokenizer.texts_to_sequences([t]), maxlen=MAXLEN, padding='post')
    prob = float(model.predict(seq)[0][0])
    return jsonify({'probability': prob})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
