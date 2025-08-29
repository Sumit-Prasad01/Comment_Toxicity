import os
import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from glove_lstm_pipeline.glove_lstm_pipeline import load_data, load_glove_embeddings, build_lstm_model

# -------------------------
# Config
# -------------------------
train_path = "data/processed/cleaned_train_data.csv"
test_path  = "data/processed/cleaned_test_data.csv"
glove_path = "glove.6B.100d.txt"
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

max_words = 50000
max_len = 320
embedding_dim = 100
batch_size = 128
epochs = 5

# -------------------------
# Pipeline
# -------------------------
print("Loading data...")
X_train, y_train, X_test, y_test, tokenizer = load_data(
    train_path, test_path, label_cols, max_words, max_len
)

print("Loading GloVe embeddings...")
embedding_matrix = load_glove_embeddings(glove_path, tokenizer, max_words, embedding_dim)

print("Building model...")
model = build_lstm_model(
    max_words, max_len, embedding_dim, embedding_matrix, num_labels=len(label_cols)
)

print("Training...")
history = model.fit(
    X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
)

# -------------------------
# Save model
# -------------------------
os.makedirs("models", exist_ok=True)

keras_path = "models/glove_lstm_model.keras"
h5_path = "models/glove_lstm_model.h5"

keras.saving.save_model(model, keras_path)
model.save(h5_path)

print(f"Model saved at:\n  {keras_path}\n  {h5_path}")

# -------------------------
# Reload & Evaluate
# -------------------------
print("\nReloading model for evaluation...")

try:
    loaded_model = keras.saving.load_model(keras_path)
    print("Loaded .keras model")
except Exception as e:
    print(f"Failed to load .keras model → trying .h5 instead. Error: {e}")
    import tensorflow as tf
    loaded_model = tf.keras.models.load_model(h5_path)
    print("Loaded .h5 model")

# -------------------------
# Evaluation
# -------------------------
if y_test is not None:
    print("\nRunning evaluation on test set...")
    preds = loaded_model.predict(X_test, batch_size=batch_size)
    preds_binary = (preds > 0.5).astype(int)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, preds_binary, target_names=label_cols, zero_division=0))

    # Confusion Matrices
    output_dir = "reports/visuals"
    os.makedirs(output_dir, exist_ok=True)

    for i, label in enumerate(label_cols):
        cm = confusion_matrix(y_test[:, i], preds_binary[:, i])
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix - {label}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        save_path = os.path.join(output_dir, f"confusion_matrix_{label}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Saved confusion matrix for '{label}' at {save_path}")

else:
    print("No test labels found → skipping evaluation. Only predictions will be generated.")
    preds = loaded_model.predict(X_test, batch_size=batch_size)
    preds_binary = (preds > 0.5).astype(int)

    # Save predictions
    import pandas as pd
    test_df = pd.read_csv(test_path)
    for i, col in enumerate(label_cols):
        test_df[f"pred_{col}"] = preds_binary[:, i]

    os.makedirs("reports/predictions", exist_ok=True)
    save_path = "reports/predictions/test_predictions.csv"
    test_df.to_csv(save_path, index=False)
    print(f"Predictions saved at {save_path}")
