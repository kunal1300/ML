import streamlit as st
import streamlit.components.v1 as components
from collections import Counter
import json
import pandas as pd
import re
import nltk
from pathlib import Path 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import altair as alt

st.set_page_config(page_title="Advanced Sentiment Dashboard", page_icon="🎬", layout="wide")

@st.cache_resource
def load_nltk_resources():
    nltk.download('stopwords', quiet=True)
    return set(stopwords.words('english'))

stop_words = load_nltk_resources()

@st.cache_data
def load_and_clean_data():
    dataset_path = Path(__file__).with_name("IMDB Dataset.csv")
    df = pd.read_csv(dataset_path)
    df['sentiment_encoded'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        words = text.split()
        words = [w for w in words if w not in stop_words]
        return " ".join(words)
    
    df["cleaned"] = df["review"].apply(clean_text)
    return df

@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform(df['cleaned'])
    y = df["sentiment_encoded"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return vectorizer, model, accuracy, cm

def clean_input_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# --- App Layout ---
st.sidebar.title("Navigation 🧭")
menu = st.sidebar.radio("Go to:", ["1. Dashboard & EDA 📊", "2. Model Performance 📈", "3. Sentiment Analyzer 🤖"])

st.sidebar.markdown("---")
st.sidebar.info("This is an advanced Sentiment Analysis Web App utilizing TF-IDF and Logistic Regression on IMDB Reviews.")

# --- Data Loading ---
with st.spinner("Loading dataset and training model... (This may take a minute on first run but will be cached after!)"):
    df = load_and_clean_data()
    vectorizer, model, accuracy, cm = train_model(df)


if menu == "1. Dashboard & EDA 📊":
    st.title("🎬 IMDB Sentiment Dashboard Overview")
    st.markdown("Explore the underlying **IMDB Dataset** and its distribution.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Data Preview")
        st.dataframe(df[['review', 'sentiment']].head(10), use_container_width=True)
    
    with col2:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        chart = alt.Chart(sentiment_counts).mark_bar().encode(
            x=alt.X('Sentiment', sort='-y'),
            y='Count',
            color=alt.Color('Sentiment', scale=alt.Scale(domain=['positive', 'negative'], range=['#00b0f0', '#ff4b4b']))
        ).properties(width=300, height=300)
        st.altair_chart(chart, use_container_width=True)
        
    st.divider()
    st.subheader("Dataset Statistics")
    st.write(f"- **Total Rows:** {df.shape[0]}")
    st.write(f"- **Total Vocabulary (Features):** {vectorizer.max_features}")


elif menu == "2. Model Performance 📈":
    st.title("📈 Model Performance Metrics")
    st.markdown("Detailed breakdown of our **Logistic Regression** classifier's ability to predict sentiment.")
    
    st.metric(label="Overall Accuracy on Test Set", value=f"{round(accuracy*100, 2)}%")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        # Format the confusion matrix into a stylized dataframe for display
        cm_df = pd.DataFrame(
            cm, 
            index=['Actual Negative (0)', 'Actual Positive (1)'], 
            columns=['Predicted Negative (0)', 'Predicted Positive (1)']
        )
        st.dataframe(cm_df, use_container_width=True)
        
    with col2:
        st.subheader("Why TF-IDF & Logistic Regression?")
        st.write("""
        - **TF-IDF** (Term Frequency-Inverse Document Frequency) normalizes word frequencies, making important but rare words stand out while penalizing common words.
        - **Logistic Regression** works exceptionally well on high-dimensional sparse data like text, providing fast inference and clear prediction probabilities.
        """)
        
elif menu == "3. Sentiment Analyzer 🤖":
    st.title("🤖 Predict Sentiment")
    
    tab1, tab2 = st.tabs(["✍️ Single Review Prediction", "📁 Bulk CSV Prediction"])
    
    with tab1:
        st.subheader("Test customized text")
        review = st.text_area("Enter your movie review here:", height=150, placeholder="E.g., The movie was absolutely fantastic! The acting was superb.")
        if st.button("Predict Sentiment", use_container_width=True, type="primary"):
            if review.strip() == "":
                st.warning("Please enter a review to predict.")
            else:
                cleaned = clean_input_text(review)
                vectorized = vectorizer.transform([cleaned])
                prediction = model.predict(vectorized)[0]
                proba = model.predict_proba(vectorized)[0]
                
                pos_prob = proba[1]
                neg_prob = proba[0]
                
                st.divider()
                col_res, col_chart = st.columns([1, 1])
                
                with col_res:
                    if prediction == 1:
                        st.success("🌟 Positive Review")
                    else:
                        st.error("💔 Negative Review")
                    st.write(f"**Confidence:** Positive: {pos_prob*100:.1f}% | Negative: {neg_prob*100:.1f}%")
                    
                with col_chart:
                    probs_df = pd.DataFrame({
                        "Sentiment": ["Negative", "Positive"],
                        "Probability": [neg_prob, pos_prob]
                    })
                    prob_chart = alt.Chart(probs_df).mark_bar().encode(
                        x=alt.X("Probability:Q", axis=alt.Axis(format='%')),
                        y=alt.Y("Sentiment:N", sort=["Positive", "Negative"]),
                        color=alt.Color("Sentiment", scale=alt.Scale(domain=["Positive", "Negative"], range=["#ff4b4b", "#00b0f0"]))
                    ).properties(height=150)
                    st.altair_chart(prob_chart, use_container_width=True)

                # --- Word Frequency Cloud ---
                st.divider()
                st.subheader("🔤 Word Frequency Cloud")
                st.caption("Bigger = more frequently used in your review. Smaller = used less.")

                word_list = cleaned.split()
                word_freq = Counter(word_list)

                if word_freq:
                    # Sort by frequency descending, take top 60
                    top_words = word_freq.most_common(60)
                    max_freq = top_words[0][1]
                    min_freq = top_words[-1][1]

                    words_json = json.dumps(
                        [{"text": w, "count": c} for w, c in top_words]
                    )

                    html_code = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{
    background: #0e1117;
    margin: 0;
    padding: 10px;
    font-family: 'Segoe UI', sans-serif;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    gap: 10px 16px;
    min-height: 220px;
    box-sizing: border-box;
  }}
  .word {{
    display: inline-block;
    cursor: default;
    transition: transform 0.2s ease, opacity 0.2s ease;
    line-height: 1.2;
    border-radius: 4px;
    padding: 2px 4px;
  }}
  .word:hover {{
    transform: scale(1.18);
    opacity: 1 !important;
  }}
</style>
</head>
<body>
<script>
  const words = {words_json};
  const maxFreq = {max_freq};
  const minFreq = {min_freq};
  const colors = [
    '#00b0f0','#ff4b4b','#f5a623','#7ed321','#bd10e0',
    '#50e3c2','#ff6b6b','#4ecdc4','#ffe66d','#a78bfa'
  ];

  const body = document.body;
  words.forEach((item, i) => {{
    const span = document.createElement('span');
    span.classList.add('word');
    span.textContent = item.text;

    // Map frequency to font size: 14px (min) to 64px (max)
    const ratio = maxFreq === minFreq
      ? 1
      : (item.count - minFreq) / (maxFreq - minFreq);
    const fontSize = Math.round(14 + ratio * 50);
    span.style.fontSize = fontSize + 'px';

    // Opacity proportional to size
    span.style.opacity = (0.55 + ratio * 0.45).toFixed(2);

    // Cycle through palette
    span.style.color = colors[i % colors.length];
    span.title = `"${{item.text}}" used ${{item.count}} time${{item.count > 1 ? 's' : ''}}` ;

    body.appendChild(span);
  }});
</script>
</body>
</html>
"""
                    components.html(html_code, height=280, scrolling=False)
                else:
                    st.info("No significant words found after cleaning. Try a longer review.")

    with tab2:
        st.subheader("Batch Operations")
        st.markdown("Upload a file containing multiple reviews, and the model will predict sentiments for automatically.")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                st.write("Preview of Uploaded Data:")
                st.dataframe(user_df.head(3), use_container_width=True)
    
                text_col = st.selectbox("Select the column containing the reviews", user_df.columns)
    
                if st.button("Predict Sentiment for File", type="primary"):
                    with st.spinner("Processing..."):
                        cleaned_texts = user_df[text_col].apply(clean_input_text)
                        X_input = vectorizer.transform(cleaned_texts)
                        file_predictions = model.predict(X_input)
                        
                        # Collect probabilities
                        probas = model.predict_proba(X_input)
                        
                        user_df['Predicted_Sentiment'] = ["Positive" if p == 1 else "Negative" for p in file_predictions]
                        user_df['Positive_Confidence'] = [f"{round(p[1]*100, 2)}%" for p in probas]
                        user_df['Negative_Confidence'] = [f"{round(p[0]*100, 2)}%" for p in probas]
                        
                        st.success("Prediction complete!")
                        st.dataframe(user_df[[text_col, 'Predicted_Sentiment', 'Positive_Confidence', 'Negative_Confidence']].head(20), use_container_width=True)

                        # --- Word Frequency Cloud (Bulk) ---
                        st.divider()
                        st.subheader("🔤 Word Frequency Cloud — All Reviews")
                        st.caption("Bigger = more frequently used across all uploaded reviews. Hover to see exact count.")

                        all_words = " ".join(cleaned_texts).split()
                        bulk_word_freq = Counter(all_words)

                        if bulk_word_freq:
                            top_words_bulk = bulk_word_freq.most_common(60)
                            max_freq_b = top_words_bulk[0][1]
                            min_freq_b = top_words_bulk[-1][1]

                            words_json_bulk = json.dumps(
                                [{"text": w, "count": c} for w, c in top_words_bulk]
                            )

                            html_bulk = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{
    background: #0e1117;
    margin: 0;
    padding: 14px;
    font-family: 'Segoe UI', sans-serif;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    gap: 12px 18px;
    min-height: 240px;
    box-sizing: border-box;
  }}
  .word {{
    display: inline-block;
    cursor: default;
    transition: transform 0.2s ease, opacity 0.2s ease;
    line-height: 1.2;
    border-radius: 4px;
    padding: 2px 5px;
  }}
  .word:hover {{
    transform: scale(1.18);
    opacity: 1 !important;
  }}
</style>
</head>
<body>
<script>
  const words = {words_json_bulk};
  const maxFreq = {max_freq_b};
  const minFreq = {min_freq_b};
  const colors = [
    '#00b0f0','#ff4b4b','#f5a623','#7ed321','#bd10e0',
    '#50e3c2','#ff6b6b','#4ecdc4','#ffe66d','#a78bfa'
  ];

  const body = document.body;
  words.forEach((item, i) => {{
    const span = document.createElement('span');
    span.classList.add('word');
    span.textContent = item.text;

    const ratio = maxFreq === minFreq
      ? 1
      : (item.count - minFreq) / (maxFreq - minFreq);
    const fontSize = Math.round(14 + ratio * 54);
    span.style.fontSize = fontSize + 'px';
    span.style.opacity = (0.55 + ratio * 0.45).toFixed(2);
    span.style.color = colors[i % colors.length];
    span.title = `"${{item.text}}" used ${{item.count}} time${{item.count > 1 ? 's' : ''}}`;

    body.appendChild(span);
  }});
</script>
</body>
</html>
"""
                            components.html(html_bulk, height=300, scrolling=False)
                        else:
                            st.info("No words found after cleaning the uploaded reviews.")

                        st.divider()
                        csv = user_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download rich results as CSV",
                            data=csv,
                            file_name="predictions_advanced.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Error processing the file: {e}")
