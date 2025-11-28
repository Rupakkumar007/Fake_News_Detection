# Fake News Detection Project by Rupak Kumar
import pandas as pd
import numpy as np
import re
import string

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = "FAKE"
true["label"] = "REAL"

# Combine both datasets
data = pd.concat([fake, true]).reset_index(drop=True)
data = data.sample(frac=1).reset_index(drop=True)  # shuffle data

# -------------------- NLP PREPROCESSING --------------------
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

data["text"] = data["text"].apply(clean_text)

# -------------------- TRAIN TEST SPLIT --------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# -------------------- FEATURE EXTRACTION --------------------
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# -------------------- MODEL TRAINING --------------------
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -------------------- EVALUATION --------------------
print("✅ Model Training Complete!")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------- SAVE MODEL --------------------
import pickle
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# -------------------- TEST FUNCTION --------------------
def predict_news(news):
    vec = vectorizer.transform([news])
    pred = model.predict(vec)
    return pred[0]

print("Example Prediction:")
print(predict_news("Government launches new policy for digital education."))
