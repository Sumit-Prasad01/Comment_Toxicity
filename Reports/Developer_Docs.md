# Toxic Comment Classification (LSTM + GloVe) – Developer Documentation
## 📌 Overview

### This project builds and deploys a Toxic Comment Classification system using:

- GloVe pretrained word embeddings

- LSTM neural network

- Multilabel classification (toxic, severe_toxic, obscene, threat, insult, identity_hate)

- Streamlit for interactive model inference

### The model predicts whether a given comment contains toxic or harmful language.

## It was trained on the Jigsaw Toxic Comment Classification dataset.

```
📂 Project Structure
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
## ⚙️ Data Preprocessing
### Dataset

#### Each row in the dataset contains:

- comment_text → input text

- toxic, severe_toxic, obscene, threat, insult, identity_hate → multilabel outputs

#### Preprocessing Steps

- Remove id column

- Clean text (lowercase, remove URLs, HTML tags, special characters)

- Tokenize comments using Keras Tokenizer

- Convert tokens to sequences (texts_to_sequences)

- Pad/truncate sequences to fixed length (max_len=320)

- Save the tokenizer for inference (tokenizer.pkl)

## 🧠 Model Architecture

### The model is an LSTM neural network with GloVe embeddings:

```
Embedding (pretrained GloVe, frozen)
    ↓
LSTM(128, dropout=0.3, recurrent_dropout=0.3)
    ↓
Dense(64, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(6, Sigmoid) → multilabel outputs
```

- Loss: binary_crossentropy

- Optimizer: Adam(learning_rate=1e-3)

- Metrics: accuracy

## 🏋️ Training Pipeline

### Training is handled in train_pipeline():

```
model, history, (X_test, y_test, tokenizer) = train_pipeline(
    train_path="data/train.csv",
    test_path="data/test.csv",
    glove_path="embeddings/glove.6B.100d.txt",
    label_cols=["toxic","severe_toxic","obscene","threat","insult","identity_hate"],
    max_words=50000,
    max_len=320,
    embedding_dim=100,
    batch_size=128,
    epochs=5
)
```


- Uses EarlyStopping to avoid overfitting

- Saves best model as glove_lstm_best.keras

- Saves final model as glove_lstm_final.keras

- Saves tokenizer as tokenizer.pkl

## 🔮 Inference (Prediction)

### During inference, we must:

- Load the trained model (.keras)

- Load the tokenizer (tokenizer.pkl)

- Preprocess new text into padded sequences

- Predict toxicity scores

### Example (inference.py):

```
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model + tokenizer
model = tf.keras.models.load_model("models/glove_lstm_final.keras")
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Prediction function
def predict_toxicity(text, model, tokenizer, max_len=320, threshold=0.5):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    prediction = model.predict(padded, verbose=0)
    return prediction[0]  # array of probabilities for 6 labels

# Example
print(predict_toxicity("You are an idiot!", model, tokenizer))

```

## 🎛️ Streamlit App

### The app (streamlit_app.py) provides an interactive UI:

### Features:

Enter a single comment → model classifies it

- Adjustable toxicity threshold slider

- Displays probability score + prediction (Toxic / Non-Toxic)

- Run with:

```
streamlit run app/streamlit_app.py
```

## 📦 Requirements

### requirements.txt

- tensorflow
- numpy
- pandas
- streamlit
- scikit-learn
- plotly

## 🚀 Deployment Notes

- Always ship model + tokenizer together (.keras + .pkl).

- Use the same max sequence length during training and inference.

- For large-scale inference, batch process comments instead of single queries.

## 📌 Key Takeaways

- The tokenizer is as critical as the model — without it, predictions will be invalid.

- Preprocessing must exactly match training-time steps.

- Model outputs 6 sigmoid scores, one for each toxic label.

- Streamlit app provides a simple way to demo the model for end users.



# Models
GloVe + LSTM
(Provide a brief description of the GloVe + LSTM model. Mention its architecture, e.g., "This model uses pre-trained GloVe word embeddings as the input layer, followed by an LSTM layer for sequence processing, and a dense output layer for binary classification.")

# Transformer (BERT)
(Describe the Transformer model approach. e.g., "We are using a pre-trained BERT model from the Hugging Face Transformers library for fine-tuning on our toxicity classification task.")