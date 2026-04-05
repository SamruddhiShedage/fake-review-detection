import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB
import joblib
import re
import nltk

# Load your trained model, vectorizer, and scaler
vectorizer = joblib.load('vectorizer.pkl')
nb_model = joblib.load('nb_model.pkl')
scaler = joblib.load('scaler.pkl')

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Function to preprocess the review text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Function to predict if a review is real or fake
def predict_review(review, vectorizer, nb_model, scaler, threshold=0.6):
    # Step 1: Preprocess the review
    processed_review = preprocess_text(review)

    # Step 2: Vectorize the review using the pre-trained TF-IDF vectorizer
    new_review_vectorized = vectorizer.transform([processed_review])

    # Step 3: Extract custom features (length, exclamation marks, sentiment score, uppercase words)
    new_features = np.array([[len(review), review.count('!'), TextBlob(review).sentiment.polarity + 1, 
                               sum(1 for word in review.split() if word.isupper())]])

    # Step 4: Scale the custom features
    new_features_scaled = scaler.transform(new_features)

    # Step 5: Combine the vectorized text features with the scaled custom features
    new_review_features = hstack([new_review_vectorized, new_features_scaled])

    # Step 6: Make a prediction using the pre-trained model
    probas = nb_model.predict_proba(new_review_features)
    fake_prob = probas[0][0]  # Probability for "Fake"
    real_prob = probas[0][1]  # Probability for "Real"
    
    # Step 7: Apply threshold to determine the final classification
    if fake_prob > real_prob and fake_prob >= threshold:
        return "Fake", {"Fake": fake_prob, "Real": real_prob}
    elif real_prob > fake_prob and real_prob >= threshold:
        return "Real", {"Fake": fake_prob, "Real": real_prob}
    else:
        return "Uncertain", {"Fake": fake_prob, "Real": real_prob}

# Streamlit UI setup
st.title("Fake Review Detection")

st.write("Enter a product review to check if it's Real or Fake:")

# Text input for the user to provide a review
user_review = st.text_area("Enter Review", height=100)

# Button to make a prediction
if st.button('Predict'):
    if user_review.strip():  # Ensure the input is not empty
        label, probabilities = predict_review(user_review, vectorizer, nb_model, scaler)
        st.write(f"**Prediction:** {label}")
        st.write(f"**Probabilities:**")
        st.write(f"- Fake: {probabilities['Fake']:.2f}")
        st.write(f"- Real: {probabilities['Real']:.2f}")
    else:
        st.write("Please enter a review to predict.")
