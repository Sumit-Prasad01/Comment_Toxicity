import tensorflow as tf
from transformers import TFBertModel

def create_model(max_len = 128, num_labels = 6):
    bert_model = TFBertModel.from_pretrained("bert-based-uncased")

    input_ids = tf.keras.layers.Inputs(shape = (max_len, ), dtype = tf.int32, name = 'input_ids')
    attention_mask = tf.keras.layers.Input(shape = (max_len, ), dtype = tf.int32, name = 'attention_mask')

    embeddings = bert_model(input_ids, attention_mask = attention_mask)[1]
    x = tf.keras.layers.Dropout(0.3)(embeddings)
    output = tf.keras.layers.Dense(num_labels, activation = 'sigmoid')(x)

    model = tf.keras.Model(inputs = [input_ids, attention_mask], output = output)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 2e - 5),
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
    )

    return model