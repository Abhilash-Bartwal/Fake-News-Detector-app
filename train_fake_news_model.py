# train_fake_news_model.py
import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# Load datasets
fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')

fake['label'] = 0
real['label'] = 1

df = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)

# Use only 'text' column for simplicity
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text.lower()

df['text'] = df['text'].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train, y_train)

# Save
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "fake_news_vectorizer.pkl")

print("Fake news model and vectorizer saved.")
