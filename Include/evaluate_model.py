# Import libraries
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    classification_report
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
import re
from textblob import TextBlob

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load saved model, vectorizer, and scaler
vectorizer = joblib.load('vectorizer.pkl')
best_nb_model = joblib.load('nb_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load the dataset for evaluation
data = pd.read_csv('fake reviews dataset.csv')

# Specify the column containing the reviews
reviews_column = 'text_'  # Assuming this is the column name used in the training data

# Preprocessing function to clean the text (same as during training)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Apply preprocessing to the entire dataset
data['cleaned_text'] = data[reviews_column].apply(preprocess_text)

# Feature extraction (same as during training)
data['review_length'] = data['cleaned_text'].apply(len)
data['exclamation_marks'] = data['cleaned_text'].apply(lambda x: x.count('!'))
data['sentiment_score'] = data['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
data['uppercase_words'] = data['cleaned_text'].apply(lambda x: sum(1 for word in x.split() if word.isupper()))

# Split the data into train, test, and validation sets (same split as training)
train_data, test_val_data = train_test_split(data, test_size=0.30, stratify=data['label'], random_state=42)
test_data, val_data = train_test_split(test_val_data, test_size=0.50, stratify=test_val_data['label'], random_state=42)

# Define feature and label sets for training and testing
X_train = vectorizer.transform(train_data['cleaned_text'])
X_test = vectorizer.transform(test_data['cleaned_text'])
X_val = vectorizer.transform(val_data['cleaned_text'])

y_train = train_data['label']
y_test = test_data['label']
y_val = val_data['label']

# Scale additional features
train_additional_features = train_data[['review_length', 'exclamation_marks', 'sentiment_score', 'uppercase_words']]
test_additional_features = test_data[['review_length', 'exclamation_marks', 'sentiment_score', 'uppercase_words']]
val_additional_features = val_data[['review_length', 'exclamation_marks', 'sentiment_score', 'uppercase_words']]

train_additional_features_scaled = scaler.fit_transform(train_additional_features)
test_additional_features_scaled = scaler.transform(test_additional_features)
val_additional_features_scaled = scaler.transform(val_additional_features)

# Combine scaled features with the vectorized text features
X_train_final = hstack([X_train, train_additional_features_scaled])
X_test_final = hstack([X_test, test_additional_features_scaled])
X_val_final = hstack([X_val, val_additional_features_scaled])

# Evaluate the model on the test set
y_pred = best_nb_model.predict(X_test_final)

# Accuracy, Precision, Recall, F1 Score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None, labels=['CG', 'OR'])
recall = recall_score(y_test, y_pred, average=None, labels=['CG', 'OR'])
f1 = f1_score(y_test, y_pred, average=None, labels=['CG', 'OR'])

# Class-wise metrics
metrics_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1 Score'],
    'CG': [precision[0], recall[0], f1[0]],
    'OR': [precision[1], recall[1], f1[1]]
})

print("Class-wise Metrics:")
print(metrics_df.to_string(index=False))
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CG', 'OR'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Plot Precision, Recall, and F1 Score
plt.figure(figsize=(10, 6))
x_labels = ['Precision', 'Recall', 'F1 Score']
x = np.arange(len(x_labels))  # Label locations
width = 0.35  # Bar width

# Bars for CG and OR
plt.bar(x - width/2, metrics_df['CG'], width, label='CG', color='blue', alpha=0.7)
plt.bar(x + width/2, metrics_df['OR'], width, label='OR', color='orange', alpha=0.7)

# Add labels and title
plt.ylabel('Score')
plt.title('Class-wise Precision, Recall, and F1 Score')
plt.xticks(x, x_labels)
plt.ylim(0, 1.05)  # Ensure the y-axis covers the full range
plt.legend(loc='lower right')

# Annotate bars with values
for i in range(len(x_labels)):
    plt.text(i - width/2, metrics_df['CG'][i] + 0.02, f"{metrics_df['CG'][i]:.2f}", ha='center')
    plt.text(i + width/2, metrics_df['OR'][i] + 0.02, f"{metrics_df['OR'][i]:.2f}", ha='center')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


