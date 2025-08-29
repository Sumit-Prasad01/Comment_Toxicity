import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

def create_and_save_tokenizer(data_path="../data/processed/cleaned_train_data.csv", text_col="comment_text",
                              max_words=50000, save_path="../models/tokenizer.pkl"):
    """
    Create a Keras Tokenizer from dataset and save it as tokenizer.pkl
    """
    # Load dataset
    df = pd.read_csv(data_path)
    print(f" Loaded dataset with {len(df)} rows")

    # Initialize tokenizer
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df[text_col].astype(str).values)

    # Save tokenizer
    import os
    os.makedirs("models", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(tokenizer, f)

    print(f" Tokenizer created and saved to {save_path}")
    print(f" Vocabulary size: {len(tokenizer.word_index)} words")

if __name__ == "__main__":
    create_and_save_tokenizer()
