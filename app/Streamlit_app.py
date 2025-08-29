import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------
# Load model + tokenizer
# -----------------------
@st.cache_resource
def load_model_and_tokenizer(model_path, tokenizer_path):
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# -----------------------
# Prediction function
# -----------------------
def predict_toxicity(text, model, tokenizer, max_len=320, threshold=0.5):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    prediction = model.predict(padded, verbose=0)
    prob = float(prediction[0][0])  # binary case
    return prob, prob > threshold

# -----------------------
# Streamlit app
# -----------------------
def main():
    st.title("🛡️ Toxic Comment Detector (LSTM + GloVe)")
    st.write("Enter a comment and the model will classify it.")

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(
        "../models/glove_lstm_model.keras", "../models/tokenizer.pkl"
    )

    user_input = st.text_area("💬 Enter a comment:")
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05)

    if st.button("🔍 Analyze"):
        if user_input.strip():
            prob, is_toxic = predict_toxicity(user_input, model, tokenizer, max_len=320, threshold=threshold)
            st.metric("Toxicity Score", f"{prob:.2%}")
            st.write(f"**Prediction:** {'❌ Toxic' if is_toxic else '✅ Non-Toxic'}")
            st.progress(prob, text=f"Toxicity Probability: {prob:.1%}")

if __name__ == "__main__":
    main()
