import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# -------------------------
# Load Data
# -------------------------
def load_data(train_path, test_path, label_cols, max_words=50000, max_len=200):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train_df["comment_text"].values)

    # Train
    X_train = tokenizer.texts_to_sequences(train_df["comment_text"].values)
    X_train = pad_sequences(X_train, maxlen=max_len)
    y_train = train_df[label_cols].values

    # Test (only text, no labels)
    X_test = tokenizer.texts_to_sequences(test_df["comment_text"].values)
    X_test = pad_sequences(X_test, maxlen=max_len)

    # If labels exist in test set , else set None
    y_test = test_df[label_cols].values if set(label_cols).issubset(test_df.columns) else None

    return X_train, y_train, X_test, y_test, tokenizer



# -------------------------
# Load Pretrained GloVe
# -------------------------
def load_glove_embeddings(glove_path, tokenizer, max_words=50000, embedding_dim=100):
    embeddings_index = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector

    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


# -------------------------
# Build Model
# -------------------------
def build_lstm_model(max_words, max_len, embedding_dim, embedding_matrix, num_labels):
    model = Sequential([
        Embedding(input_dim=max_words,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  input_length=max_len,
                  trainable=False),
        LSTM(128, dropout=0.3, recurrent_dropout=0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_labels, activation="sigmoid")  # multilabel
    ])

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model
