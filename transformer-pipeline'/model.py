import tensorflow as tf
from transformers import TFAutoModel

def create_model(max_len=256, num_labels=6, trainable=False):
    """
    Create a DistilBERT-based classification model.
    
    Args:
        max_len (int): maximum sequence length
        num_labels (int): number of output labels
        trainable (bool): whether to fine-tune DistilBERT (default: False for speed)
    """
    # Load DistilBERT
    bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")

    # Freeze encoder for speed (trainable=False)
    for layer in bert_model.layers:
        layer.trainable = trainable

    # Define inputs
    input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    # Get BERT embeddings (last_hidden_state, no pooler in DistilBERT)
    outputs = bert_model(input_ids, attention_mask=attention_mask)
    hidden_state = outputs.last_hidden_state       # shape: (batch, seq_len, hidden_size)
    cls_token = hidden_state[:, 0, :]              # take [CLS] token representation

    # Classification head
    x = tf.keras.layers.Dropout(0.3)(cls_token)
    output = tf.keras.layers.Dense(num_labels, activation="sigmoid")(x)  # multi-label

    # Build model
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")]
    )

    return model
