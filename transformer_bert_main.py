import argparse
import os
import tensorflow as tf
from transformer_pipeline.train import train_model
from transformer_pipeline.evaluate import evaluate_model
from transformer_pipeline.data_prep import load_and_prepare
from transformer_pipeline.model import create_model


def get_first_existing_path(possible_paths):
    """Return the first path that exists from the list"""
    for path in possible_paths:
        if os.path.exists(path):
            print(f" Found: {path}")
            return path
    raise FileNotFoundError(" No valid file found in possible paths.")


def main(args):
    max_len = 256

    # Define possible paths for train and test datasets
    possible_train_paths = [
        "data/processed/cleaned_train_data.csv",
        "../data/processed/cleaned_train_data.csv"
    ]

    possible_test_paths = [
        "data/processed/cleaned_test_data.csv",
        "../data/processed/cleaned_test_data.csv"
    ]

    # Pick the first valid path
    train_path = get_first_existing_path(possible_train_paths)
    test_path = get_first_existing_path(possible_test_paths)

    if args.mode == "train":
        print(" Starting Training...")

        # Load and prepare data
        train_encodings, test_encodings, y, label_cols, tokenizer = load_and_prepare(
            train_path, test_path, max_len=max_len
        )

        # Train model
        history = train_model()
        print(" Training Finished!")

    elif args.mode == "evaluate":
        print(" Loading Model for Evaluation...")

        # Load and prepare data
        train_encodings, test_encodings, y, label_cols, tokenizer = load_and_prepare(
            train_path, test_path, max_len=max_len
        )

        # Define possible model paths
        possible_model_paths = [
            "models/toxicity_model.keras",
            "models/toxicity_model.h5",
            "../models/toxicity_model.keras"
        ]
        model_path = get_first_existing_path(possible_model_paths)

        # Load model
        model = tf.keras.models.load_model(model_path)

        # Prepare evaluation data
        X_val = {
            "input_ids": tf.convert_to_tensor(test_encodings["input_ids"]),
            "attention_mask": tf.convert_to_tensor(test_encodings["attention_mask"])
        }

        # For now, reuse y since test labels may not exist
        y_val = y[:len(test_encodings["input_ids"])]

        evaluate_model(model, X_val, y_val, label_cols)
        print("Evaluation Completed!")

    else:
        print(" Invalid mode! Use --mode train or --mode evaluate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comment Toxicity Detection Pipeline")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "evaluate"],
                        help="Mode: train or evaluate")

    args = parser.parse_args()
    main(args)
