import os
from src.glove_lstm_pipeline import load_data, load_glove_embeddings, build_lstm_model

# -------------------------
# Config
# -------------------------
train_path = "data/processed/cleaned_train_data.csv"
test_path  = "data/processed/cleaned_test_data.csv"
glove_path = "glove.6B.100d.txt"   
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

max_words = 50000
max_len = 256
embedding_dim = 100

# -------------------------
# Pipeline
# -------------------------
print("Loading data...")
X_train, y_train, X_test, y_test, tokenizer = load_data(train_path, test_path, label_cols, max_words, max_len)

print("Loading GloVe embeddings...")
embedding_matrix = load_glove_embeddings(glove_path, tokenizer, max_words, embedding_dim)

print("Building model...")
model = build_lstm_model(max_words, max_len, embedding_dim, embedding_matrix, num_labels=len(label_cols))

print("Training...")
history = model.fit(X_train, y_train, batch_size=128, epochs=5, validation_split=0.1)

print("Evaluating...")
if y_test is not None:
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {acc}")
else:
    print("Test labels not found. Skipping evaluation.")

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/glove_lstm_model.keras")
print("Model saved at models/glove_lstm_model.keras")
