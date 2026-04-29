import streamlit as st
import pandas as pd
import re
import nltk 

from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df = pd.read_csv("IMDB Dataset.csv")


# Label Encoding
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative':0})

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ',text) 
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["cleaned"] = df["review"].apply(clean_text)

# TFIDF
vectorizer = TfidfVectorizer(ngram_range = (1,2), max_features = 5000)
X = vectorizer.fit_transform(df['cleaned'])
y = df["sentiment"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2,random_state=42
)

model = LogisticRegression(max_iter = 1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

cv_scores = cross_val_score(model,X,y, cv=5)

st.title("🎬 IMDB Movie Review Sentiment")

st.subheader("📊 Model Performance")
st.write("Accuray:", round(accuracy, 2))
st.write("Cross Validation Score:", round(cv_scores.mean()))
st.write("Confusion Matrix")
st.write(cm)

st.divider()

review = st.text_area("✍️ Enter Movie Review")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)

        if prediction == 1:
            st.success("😁 Positive Review")
        else:
            st.error("😯 Negative Review")