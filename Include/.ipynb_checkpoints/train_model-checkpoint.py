#!/usr/bin/env python
# coding: utf-8

# Import libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
import joblib
from textblob import TextBlob

# Load the dataset
data = pd.read_csv('fake reviews dataset.csv')
print(data.columns)

# Specify the column containing the reviews
reviews_column = 'text_' 

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing function to clean the text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Apply preprocessing to the entire dataset
data['cleaned_text'] = data[reviews_column].apply(preprocess_text)

# Feature extraction
data['review_length'] = data['cleaned_text'].apply(len)
data['exclamation_marks'] = data['cleaned_text'].apply(lambda x: x.count('!'))
data['sentiment_score'] = data['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
data['uppercase_words'] = data['cleaned_text'].apply(lambda x: sum(1 for word in x.split() if word.isupper()))

# Split the data into train, test, and validation sets
print(data['label'].value_counts()) 

train_data, test_val_data = train_test_split(data, test_size=0.3, stratify=data['label'], random_state=42)
test_data, val_data = train_test_split(test_val_data, test_size=0.5, stratify=test_val_data['label'], random_state=42)

# Check the class distribution in each dataset
print("Train set class distribution:", train_data['label'].value_counts())
print("Test set class distribution:", test_data['label'].value_counts())
print("Validation set class distribution:", val_data['label'].value_counts())

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Unigrams + bigrams
vectorizer.fit(train_data['cleaned_text'])

train_data_vectorized = vectorizer.transform(train_data['cleaned_text'])
test_data_vectorized = vectorizer.transform(test_data['cleaned_text'])
val_data_vectorized = vectorizer.transform(val_data['cleaned_text'])

# Scale additional features
scaler = MinMaxScaler()
train_additional_features_scaled = scaler.fit_transform(train_data[['review_length', 'exclamation_marks', 'sentiment_score', 'uppercase_words']])
test_additional_features_scaled = scaler.transform(test_data[['review_length', 'exclamation_marks', 'sentiment_score', 'uppercase_words']])
val_additional_features_scaled = scaler.transform(val_data[['review_length', 'exclamation_marks', 'sentiment_score', 'uppercase_words']])

# Combine scaled features with the vectorized text features
train_features = hstack([train_data_vectorized, train_additional_features_scaled])
test_features = hstack([test_data_vectorized, test_additional_features_scaled])
val_features = hstack([val_data_vectorized, val_additional_features_scaled])

# Labels for training, validation, and testing
train_labels = train_data['label']
test_labels = test_data['label']
val_labels = val_data['label']  # Corrected this line

# GridSearchCV for hyperparameter tuning
param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100],
    'fit_prior': [True, False],
}

nb_model = MultinomialNB()

grid_search = GridSearchCV(estimator=nb_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(train_features, train_labels)

# Best model
best_nb_model = grid_search.best_estimator_
print("Class labels and their indexes:")
for index, class_label in enumerate(best_nb_model.classes_):
    print(f"Index {index} corresponds to class '{class_label}'")

# Save the trained model, vectorizer, and scaler
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(best_nb_model, 'nb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model, vectorizer, and scaler saved successfully.")
