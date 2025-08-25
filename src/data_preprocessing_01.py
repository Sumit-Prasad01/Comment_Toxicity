import re 
import string
import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Text Cleaning

def clean_text():
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", '', text)
    text = re.sub(r"\n", '', text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans('', '',string.punctuation))
    text = re.sub(r"\s+", "",text).strip()
    return text


# Load and Preprocess Data

def load_and_prepare(train_path, test_path, max_len = 128, test_size = 0.2):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df['clean_comment'] = train_df['comment_text'].apply(clean_text)
    test_df['clean_comment'] = test_df['comment_text'].apply(clean_text)

    label_cols = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
    y = train_df[label_cols].values

    tokenizer = BertTokenizer.from_pretrained('bert-based-uncased')

    train_encodings = tokenizer(
        list(train_df['clean_comment']), truncation = True, padding = True, max_length = max_len
    )
    test_encodings = tokenizer(
        list(test_df['clean_comment']), truncation = True, padding = True, max_length = max_len
    )

    X_train, X_val, y_train, y_val = train_test_split(train_encodings['input_ids'], y, test_size= test_size)

    return (train_encodings, test_encodings, y, label_cols, tokenizer)

