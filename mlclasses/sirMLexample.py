import pandas as pd

import re 

import nltk

import streamlit as st 

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')

stop_words = set(stopwords.words ('english'))

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'subject:', '', text)

    text = re.sub(r'[^a-zA-Z]', ' ',text) 

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)

st.title("Email Spam Detection (NLP Project)")

st.write("Upload dataset, train model and test email messages")

st.sidebar.header("User Details")

name = st.sidebar.text_input("Enter Name")

email = st.sidebar.text_input("Enter Email")

if name and email:

    st.sidebar.success(f"Welcome {name}")

uploaded_file = st.file_uploader("Upload CSV File", type = ["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Dataset")

    st.dataframe(df.head())

    if "text" not in df.columns or "spam" not in df.columns:

        st.error("Dataset must contain columns: 'text' and 'spam'")

    else:

        df["clean_text"] = df["text"].apply(clean_text)

        st.subheader("Cleaned data")

        st.dataframe(df[["clean_text", "spam" ]].head())

        tfidf = TfidfVectorizer()

        X = tfidf.fit_transform(df["clean_text"])

        y = df["spam"]

        model = LogisticRegression()

        model.fit(X,y)

        st.success("Model Training Completed")

        st.subheader("Test Email / Message")

        user_input = st.text_area("Enter Email Content")

        if st.button("Predict Spam or Not"):

            if user_input.strip() == "":

                st.warning("Please Enter email content")

            else:

                cleaned = clean_text(user_input)

                vector = tfidf.transform([cleaned])

                prediction = model.predict(vector[0])

                st.write("Name:", name)

                st.write("Email:", email)

                if prediction == 1:

                    st.error("This is SPAM EMAIL")

                else:

                    st.success("This is NOT SPAM EMAIL")

else:

    st.info("Upload your dataset to start training")