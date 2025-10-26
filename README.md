# 💬 Sentiment Analysis using TensorFlow (Word2Vec + Flask + Ngrok)

### 🧠 Endterm Exam Project — NLP Transfer Learning  
**Name:** Edwin P. Bayog Jr.  
**Section:** BSCpE 4-A  
📂 **GitHub Repository:** [https://github.com/naaivvv/Project-Sentiment-Analysis-Tensorflow-](https://github.com/naaivvv/Project-Sentiment-Analysis-Tensorflow-)

🚀 **Google Colab Notebook:** [Open in Colab](https://colab.research.google.com/drive/1FfCgNrhpYva99ilcC11OUt_5ORAjbbZp?usp=sharing)


---

## 📘 Project Overview

This project demonstrates a **Sentiment Analysis System** built using **TensorFlow**, **Word2Vec embeddings**, and a **Flask web app** served through **Ngrok** for online access.  

The model was trained to classify text inputs into three categories:
- 😊 **Positive**
- 😐 **Neutral**
- 😡 **Negative**

It uses a **Google News Word2Vec pre-trained model** for better natural language understanding and includes a **TailwindCSS-powered chat interface** for real-time predictions.

---

## ⚙️ Technologies Used

- 🧠 **TensorFlow / Keras** — for deep learning model
- 💬 **Word2Vec (Google Pretrained Model)** — for word embeddings
- 🧾 **Flask** — to serve the model via a REST API
- 🌐 **Ngrok** — to expose the local Flask server online
- 🎨 **TailwindCSS** — for a modern chat UI
- 💾 **Pickle** — for saving and loading the tokenizer
- 📊 **Matplotlib** — for visualizing training accuracy and loss

---

## 🚀 Running the Project on Google Colab

Follow these steps to fully run and access the chatbox web app through **Ngrok**:

### 1️⃣ Upload or Clone the Notebook
- Open the notebook: `sentiment_word2vec_ngrok_v3_final.ipynb`
- Make sure your model (`sentiment_w2v_model.keras`) and tokenizer (`tokenizer.pickle`) are uploaded in Colab.

### 2️⃣ Install Required Libraries
In a new Colab cell, run:
```python
!pip install tensorflow flask flask-ngrok gensim nltk matplotlib
