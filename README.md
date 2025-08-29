# 🛡️ Toxic Comment Classification (LSTM + GloVe)

This project implements a **Toxic Comment Detector** using:
- Pretrained **GloVe word embeddings**
- **LSTM neural network** for text classification
- **Multilabel classification** (toxic, severe_toxic, obscene, threat, insult, identity_hate)
- **Streamlit app** for interactive testing

The model predicts whether a given comment contains toxic or harmful language.

---

## 📌 Features
- Train an **LSTM with GloVe embeddings**
- Multilabel classification (6 toxicity categories)
- Save & reuse the **Tokenizer** (`tokenizer.pkl`)
- Streamlit app for real-time comment analysis
- Adjustable toxicity threshold

---

## 📂 Project Structure
```
toxic_comment_classifier/
├── app/                  # Application code (if you have a front-end or API)
├── data/                 # Dataset files (e.g., raw data, processed data)
├── glove/                # GloVe embedding files and related scripts
│   └── glove.6B.100d.txt
├── glove_lstm_pipeline/  # Scripts for the GloVe + LSTM model
├── models/               # Saved trained models
├── notebooks/            # Jupyter notebooks for exploration and prototyping
├── Reports/              # Reports, visualizations, and a project summary
├── transformer_pipeline/ # Scripts for the Transformer-based model
├── utils/                # Utility scripts (e.g., Model_evaluation)
├── .gitignore            # Files and folders to ignore in Git
├── check_len.py          # A script for checking text length
├── requirements.txt      # Project dependencies
├── README.md             # This filets.txt
├── main_glove_lstm.py
```

---
🏋️ Training
Run the training pipeline:

bash
Copy code
python main_glove_lstm.py
This will:

Train the LSTM with GloVe embeddings

Save the best model → models/glove_lstm_best.keras

Save the final model → models/glove_lstm_final.keras

Save the tokenizer → models/tokenizer.pkl

🔮 Inference
Use the saved model & tokenizer to predict toxicity:

python
Copy code
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model + tokenizer
model = tf.keras.models.load_model("models/glove_lstm_final.keras")
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def predict_toxicity(text, model, tokenizer, max_len=320):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")
    prediction = model.predict(padded, verbose=0)
    return prediction[0]  # [toxic, severe_toxic, obscene, threat, insult, identity_hate]

print(predict_toxicity("You are an idiot!", model, tokenizer))
🎛️ Streamlit App
Run the app:

bash
Copy code
streamlit run app/streamlit_app.py
Features:

Enter a comment → classify as Toxic / Non-Toxic

Shows toxicity probability score

Adjustable threshold

Sidebar with sample comments

📦 Requirements
requirements.txt

nginx
Copy code
tensorflow
numpy
pandas
streamlit
scikit-learn
plotly
🚀 Deployment Notes
Always ship glove_lstm_final.keras and tokenizer.pkl together.

Keep the preprocessing consistent between training and inference.

Model outputs 6 probabilities (0–1), one for each toxicity label.

📌 Credits
Dataset: Jigsaw Toxic Comment Classification

Embeddings: GloVe (Stanford NLP)

Frameworks: TensorFlow / Keras, Streamlit
