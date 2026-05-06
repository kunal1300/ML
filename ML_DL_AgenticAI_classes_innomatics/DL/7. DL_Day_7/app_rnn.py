import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline

# ─── PAGE CONFIG ───
st.set_page_config(page_title="RNN vs Transformer", page_icon="🧠", layout="wide")

# ─── CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0e1117; }
    .stTextInput > div > div > input { background-color: #262730; color: white; }
    .card {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #464b5d;
        background-color: #161b22;
        margin-bottom: 20px;
    }
    .model-label { font-size: 0.9rem; color: #8b949e; font-weight: 600; text-transform: uppercase; margin-bottom: 5px; }
    .prediction { font-size: 2rem; font-weight: 800; color: #58a6ff; }
    .correct { color: #3fb950; }
    .wrong { color: #f85149; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD DATA & MODELS (CACHED) ───
@st.cache_resource
def init_models():
    # 1. Dataset Setup
    sentences = [
        "I grew up in France so I speak fluent french",
        "I grew up in India so I speak fluent hindi",
        "I grew up in Germany so I speak fluent german",
        "I grew up in Spain so I speak fluent spanish",
    ]
    languages = {"french", "hindi", "german", "spanish"}
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    vocab_size = len(tokenizer.word_index) + 1
    sequences = tokenizer.texts_to_sequences(sentences)
    
    X, y = [], []
    for seq in sequences:
        for i in range(1, len(seq)):
            X.append(seq[:i])
            y.append(seq[i])
    
    X = pad_sequences(X, maxlen=10)
    y = np.array(y)
    
    # 2. Train RNN
    rnn_model = Sequential([
        Embedding(vocab_size, 32),
        SimpleRNN(32),
        Dense(vocab_size, activation='softmax')
    ])
    rnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    rnn_model.fit(X, y, epochs=1000, verbose=0)
    
    # 3. Load Transformer
    fill_mask = pipeline("fill-mask", model="bert-base-uncased")
    
    return rnn_model, tokenizer, fill_mask, languages

rnn_model, tokenizer, fill_mask, languages = init_models()

# ─── PREDICTION LOGIC ───
def rnn_predict(text):
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=10)
    pred = rnn_model.predict(seq, verbose=0)[0]
    top_indices = np.argsort(pred)[-5:][::-1]
    for idx in top_indices:
        word = tokenizer.index_word.get(idx, "")
        if word in languages:
            return word
    return tokenizer.index_word[np.argmax(pred)]

def transformer_predict(text):
    prompt = text.strip()
    if not prompt.endswith("."):
        prompt += "."
    prompt += " The language I speak is [MASK]."
    
    results = fill_mask(prompt, top_k=50)
    for r in results:
        token = r["token_str"].strip().lower()
        if token in languages:
            return token
    return "unknown"

# ─── UI ───
st.title("🧠 RNN vs Transformer")
st.markdown("### Language Prediction & Long-Term Memory Test")
st.info("💡 **Lesson:** RNNs often forget the beginning of a sentence (Vanishing Gradient), while Transformers remember everything (Attention).")

user_input = st.text_input("Enter a sentence starting with 'I grew up in [Country]...'", 
                          value="I grew up in France many years ago and after living there for a long time I speak fluent",
                          help="Try a short sentence vs a very long one to see the difference.")

if user_input:
    col1, col2 = st.columns(2)
    
    with st.spinner("Analyzing..."):
        r_pred = rnn_predict(user_input)
        t_pred = transformer_predict(user_input)
    
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="model-label">Simple RNN (Short-Term Memory)</div>
            <div class="prediction">{(r_pred).capitalize()}</div>
            <p style="font-size:0.8rem; color:#8b949e; margin-top:10px;">
                Simple RNNs struggle with long-distance context.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="model-label">Transformer (BERT - Attention)</div>
            <div class="prediction" style="color: #3fb950;">{(t_pred).capitalize()}</div>
            <p style="font-size:0.8rem; color:#8b949e; margin-top:10px;">
                Transformers use Attention to look back at any word instantly.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Explanation
    if r_pred != t_pred and t_pred != "unknown":
        st.warning(f"⚠️ **Result:** The RNN predicted '{r_pred}' but the Transformer correctly identified '{t_pred}'. This is because the RNN 'forgot' the beginning of your sentence!")
    elif r_pred == t_pred and t_pred != "unknown":
        st.success(f"✅ **Result:** Both models agreed on '{t_pred}'. The sentence was likely short enough for the RNN to maintain its context.")

# ─── SIDEBAR ───
with st.sidebar:
    st.header("About")
    st.write("This app compares a manually trained **Simple RNN** (Keras) against a pre-trained **Transformer** (BERT via Hugging Face).")
    st.markdown("---")
    st.write("**Training Data:**")
    st.code("I grew up in France -> french\nI grew up in India -> hindi\nI grew up in Germany -> german\nI grew up in Spain -> spanish")
