import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer
model = load_model('lstm_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Define maximum sequence length (should match the value used during training)
max_sequence_length = 100

# Streamlit app
st.title("📰 Fake News Detection")
st.write("Enter a news headline or article below to determine its authenticity.")

# Text input
user_input = st.text_area("News Text", height=150)

if st.button("Analyze"):
    if user_input:
        # Preprocess the input
        sequence = tokenizer.texts_to_sequences([user_input])
        padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
        
        # Make prediction
        prediction = model.predict(padded_sequence)[0][0]

        # Display result
        if prediction <= 0.6:
            st.error("🚨 The news is predicted to be **FAKE**.")
        else:
            st.success("✅ The news is predicted to be **REAL**.")

        # Show confidence for both classes
        st.markdown("### 🔍 Prediction Confidence")
        st.write(f"🧾 **REAL**: {prediction:.2%}")
        st.write(f"📄 **FAKE**: {(1 - prediction):.2%}")
    else:
        st.warning("Please enter some text to analyze.")
