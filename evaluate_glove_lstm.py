import os
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------
# Config
# -------------------------
train_path = "data/processed/cleaned_train_data.csv"
test_path  = "data/processed/cleaned_test_data.csv"
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

max_words = 50000
max_len = 256
model_path = "models/glove_lstm_model.keras"

# -------------------------
# Load Data
# -------------------------
print("Loading train data for evaluation...")
df = pd.read_csv(train_path)

# Split into train/test with labels
X_train, X_test, y_train, y_test = train_test_split(
    df["comment_text"], df[label_cols],
    test_size=0.2, random_state=42
)

# Tokenize text using same tokenizer settings
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")

# -------------------------
# Load Model
# -------------------------
print("Loading model...")
model = keras.saving.load_model(model_path)

# -------------------------
# Predict
# -------------------------
print("Running predictions...")
preds = model.predict(X_test_pad, batch_size=128)
preds_binary = (preds > 0.5).astype(int)

# -------------------------
# Handle Evaluation
# -------------------------
if y_test is not None and len(y_test) > 0:
    print("\n✅ Classification Report:")
    print(classification_report(y_test.values, preds_binary, target_names=label_cols))
else:
    print("⚠️ No labels found in test set → saving predictions instead.")
    test_df = pd.read_csv(test_path)
    test_seq = tokenizer.texts_to_sequences(test_df["comment_text"])
    test_pad = pad_sequences(test_seq, maxlen=max_len, padding="post", truncating="post")

    preds = model.predict(test_pad, batch_size=128)
    preds_binary = (preds > 0.5).astype(int)

    for i, col in enumerate(label_cols):
        test_df[f"pred_{col}"] = preds_binary[:, i]

    os.makedirs("reports/predictions", exist_ok=True)
    save_path = "reports/predictions/test_predictions.csv"
    test_df.to_csv(save_path, index=False)
    print(f"✅ Predictions saved at {save_path}")
