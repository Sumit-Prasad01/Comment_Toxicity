import os
import tensorflow as tf
from src.data_prep import load_and_prepare
from src.model import create_model


def get_first_existing_path(possible_paths):
    """Return the first valid path from a list of possible paths."""
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found: {path}")
            return path
    raise FileNotFoundError(f"❌ None of the provided paths exist: {possible_paths}")


def train_model(
    max_len=256,
    epochs=2,
    batch_size=32
):
    # Define possible train/test paths
    possible_train_paths = [
        "data/processed/cleaned_train_data.csv",
        "data/cleaned_train_data.csv",
        "../data/processed/cleaned_train_data.csv"
    ]

    possible_test_paths = [
        "data/processed/cleaned_test_data.csv",
        "data/cleaned_test_data.csv",
        "../data/processed/cleaned_test_data.csv"
    ]

    # Pick the first valid file from each list
    train_path = get_first_existing_path(possible_train_paths)
    test_path  = get_first_existing_path(possible_test_paths)

    # Ensure models directory exists
    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    train_encodings, test_encodings, y, label_cols, tokenizer = load_and_prepare(
        train_path, test_path, max_len=max_len
    )

    # Convert to TensorFlow tensors
    input_ids = tf.convert_to_tensor(train_encodings["input_ids"])
    attention_masks = tf.convert_to_tensor(train_encodings["attention_mask"])
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    # Build dataset with both inputs
    dataset = tf.data.Dataset.from_tensor_slices((
        {"input_ids": input_ids, "attention_mask": attention_masks},
        y
    ))
    dataset = dataset.shuffle(1000).batch(batch_size)

    # Build model
    model = create_model(max_len=max_len, num_labels=len(label_cols))

    # Train
    history = model.fit(dataset, epochs=epochs)

    # Save model + tokenizer
    model_path = os.path.join(model_dir, "toxicity_model.keras")
    model.save(model_path)
    tokenizer.save_pretrained(os.path.join(model_dir, "tokenizer"))

    print(f"✅ Model saved at: {model_path}")
    print(f"✅ Tokenizer saved at: {os.path.join(model_dir, 'tokenizer')}")

    return history


if __name__ == "__main__":
    train_model()
