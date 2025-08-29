import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# -------------------------
# Load Data
# -------------------------
def load_data(train_path, test_path, label_cols, max_words=50000, max_len=320):
    """
    Load training & test data, tokenize, pad sequences.
    Supports multilabel (sigmoid).
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Tokenizer
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["comment_text"].values)

    # Train
    X_train = tokenizer.texts_to_sequences(train_df["comment_text"].values)
    X_train = pad_sequences(X_train, maxlen=max_len, padding="post", truncating="post")
    y_train = train_df[label_cols].values.astype("float32")

    # Test (with or without labels)
    X_test = tokenizer.texts_to_sequences(test_df["comment_text"].values)
    X_test = pad_sequences(X_test, maxlen=max_len, padding="post", truncating="post")

    y_test = None
    if set(label_cols).issubset(test_df.columns):
        y_test = test_df[label_cols].values.astype("float32")

    return X_train, y_train, X_test, y_test, tokenizer


# -------------------------
# Load Pretrained GloVe
# -------------------------
def load_glove_embeddings(glove_path, tokenizer, max_words=50000, embedding_dim=100):
    """
    Load GloVe vectors and create embedding matrix for tokenizer vocab.
    """
    print("Loading GloVe embeddings...")
    embeddings_index = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector

    print(f"Loaded {len(embeddings_index)} word vectors from GloVe.")

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
    """
    LSTM model with pretrained embeddings (frozen).
    """
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

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"]
    )
    return model


# -------------------------
# Train Model (example usage)
# -------------------------
def train_pipeline(train_path, test_path, glove_path, label_cols,
                   max_words=50000, max_len=320, embedding_dim=100,
                   batch_size=128, epochs=5):

    # Load data
    X_train, y_train, X_test, y_test, tokenizer = load_data(
        train_path, test_path, label_cols, max_words=max_words, max_len=max_len
    )

    # Load embeddings
    embedding_matrix = load_glove_embeddings(glove_path, tokenizer, max_words, embedding_dim)

    # Build model
    model = build_lstm_model(max_words, max_len, embedding_dim, embedding_matrix, num_labels=len(label_cols))

    # Callbacks (early stopping + save best model)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ModelCheckpoint("models/glove_lstm_best.keras", save_best_only=True)
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks
    )

    # Save final model
    model.save("../models/glove_lstm_final.keras")
    print("Model training complete and saved.")

    return model, history, (X_test, y_test, tokenizer)
