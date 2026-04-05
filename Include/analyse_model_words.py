import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model and vectorizer
best_nb_model = joblib.load('nb_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Get the feature names (words)
feature_names = np.array(vectorizer.get_feature_names_out())

# Get the log-probabilities of each feature for both classes
class_log_probs = best_nb_model.feature_log_prob_

# Convert log-probabilities to probabilities
class_probs = np.exp(class_log_probs)

# Ensure top_n does not exceed the number of features
top_n = 30 # Define how many top words to inspect
num_features = len(feature_names)
if top_n > num_features:
    top_n = num_features  # Adjust top_n if it's larger than the number of features

# Get the indices of the top N words for both classes (checking bounds)
cg_top_indices = class_log_probs[0].argsort()[-top_n:][::-1]  # "CG" class (Fake)
or_top_indices = class_log_probs[1].argsort()[-top_n:][::-1]  # "OR" class (Genuine)

# Ensure the indices are within bounds
cg_top_indices = cg_top_indices[cg_top_indices < num_features]  # Filter out invalid indices
or_top_indices = or_top_indices[or_top_indices < num_features]  # Filter out invalid indices

# Extract the corresponding words and their probabilities
cg_top_words = feature_names[cg_top_indices]
or_top_words = feature_names[or_top_indices]

cg_top_probs = class_probs[0][cg_top_indices]
or_top_probs = class_probs[1][or_top_indices]

# Print the top words for each class
print(f"Top {top_n} words for CG (Fake) class and their probabilities:")
for word, prob in zip(cg_top_words, cg_top_probs):
    print(f"{word}: {prob:.4f}")

print(f"\nTop {top_n} words for OR (Genuine) class and their probabilities:")
for word, prob in zip(or_top_words, or_top_probs):
    print(f"{word}: {prob:.4f}")

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create word clouds for both classes
cg_wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(dict(zip(cg_top_words, cg_top_probs)))
or_wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(dict(zip(or_top_words, or_top_probs)))

# Plot the word clouds
plt.figure(figsize=(12, 6))

# Plot the CG (Fake) word cloud
plt.subplot(1, 2, 1)
plt.imshow(cg_wordcloud, interpolation="bilinear")
plt.title("Word Cloud for Fake Reviews (CG)")
plt.axis('off')

# Plot the OR (Genuine) word cloud
plt.subplot(1, 2, 2)
plt.imshow(or_wordcloud, interpolation="bilinear")
plt.title("Word Cloud for Genuine Reviews (OR)")
plt.axis('off')

plt.tight_layout()
plt.show()

# Optionally, plot the word importance (word cloud or bar chart)
cg_importances = class_log_probs[0][cg_top_indices]
or_importances = class_log_probs[1][or_top_indices]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(cg_top_words, cg_top_probs, color='skyblue')
plt.title("Top Words for CG (Fake)")

plt.subplot(1, 2, 2)
plt.barh(or_top_words, or_top_probs, color='lightgreen')
plt.title("Top Words for OR (Genuine)")

plt.tight_layout()
plt.show()
