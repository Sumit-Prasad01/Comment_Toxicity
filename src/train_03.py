import tensorflow as tf
import numpy as np
from data_preprocessing_01 import load_and_prepare
from model_02 import create_model

# Training Script

def train_model():
    train_path = '../data/processed/cleaned_train_data.csv'
    test_path = '../data/processed/cleaned_test_data.csv'
    max_len = 128
    

    # Load Data

    train_encodings, test_encodings, y, label_cols, tokenizer =  load_and_prepare(train_path, test_path, max_len = max_len)

    x = tf.convert_to_tensor(train_encodings['input_ids'])
    y = tf.convert_to_tensor(y, dtype = tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices(({'input_ids' : x}, y))
    dataset = dataset.shuffle(1000).batch(32)
    
    model = create_model(max_len = max_len, num_labels = len(label_cols))

    history = model.fit(dataset, epochs = 5)

    model.save('../models/toxicity_model.keras')
    tokenizer.save_pretrained('../models/tokenizer')
