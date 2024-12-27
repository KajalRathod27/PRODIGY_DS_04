import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

# Download NLTK stopwords
nltk.download('stopwords')

# ------------------------------
# 1. Load and Inspect the Dataset
# ------------------------------
df = pd.read_csv('Dataset/twitter_training.csv', encoding='latin1')

# Dynamically handle column names
df.columns = ['ID', 'Entity', 'Sentiment', 'Tweet'] if len(df.columns) == 4 else ['ID', 'Entity', 'Sentiment', 'Tweet'] + [f'Extra_{i}' for i in range(len(df.columns) - 4)]

# Drop unnecessary columns and null values
df = df[['Sentiment', 'Tweet']].dropna()

# ------------------------------
# 2. Data Preprocessing
# ------------------------------
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

# Preprocessing function (optimized)
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'http\S+|@\w+|#\w+|\d+|[^\w\s]', '', text)  # Remove URLs, mentions, hashtags, numbers, punctuation
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])  # Stopwords removal and stemming
    return text

# Apply preprocessing
df['Clean_Tweet'] = df['Tweet'].apply(preprocess_text)

# ------------------------------
# 3. Feature Extraction
# ------------------------------
# TF-IDF Vectorization (use a smaller number of features)
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X = tfidf_vectorizer.fit_transform(df['Clean_Tweet'])
y = df['Sentiment']

# ------------------------------
# 4. Sentiment Classification Model
# ------------------------------
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Accuracy and Classification Report
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ------------------------------
# 5. Model Evaluation with Validation Data
# ------------------------------
df_validation = pd.read_csv('Dataset/twitter_validation.csv', encoding='latin1')

# Dynamically handle validation dataset columns
df_validation.columns = ['ID', 'Entity', 'Sentiment', 'Tweet'] if len(df_validation.columns) == 4 else ['ID', 'Entity', 'Sentiment', 'Tweet'] + [f'Extra_{i}' for i in range(len(df_validation.columns) - 4)]

# Preprocess validation dataset
df_validation = df_validation[['Sentiment', 'Tweet']].dropna()
df_validation['Clean_Tweet'] = df_validation['Tweet'].apply(preprocess_text)

# Feature extraction using the same TF-IDF vectorizer
X_validation = tfidf_vectorizer.transform(df_validation['Clean_Tweet'])
y_validation = df_validation['Sentiment']

# Predict and evaluate validation dataset
y_val_pred = model.predict(X_validation)

# Validation Accuracy and Report
print("\nValidation Accuracy Score:", accuracy_score(y_validation, y_val_pred))
print("\nValidation Classification Report:\n", classification_report(y_validation, y_val_pred))

# Validation Confusion Matrix
conf_matrix_val = confusion_matrix(y_validation, y_val_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_val, annot=True, fmt='d', cmap='Greens', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
