import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline

# ─── DATASET ───
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

# ─── RNN MODEL (WEAK ON PURPOSE) ───
rnn_model = Sequential([
    Embedding(vocab_size, 32),
    SimpleRNN(32),
    Dense(vocab_size, activation='softmax')
])
rnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
print("\n[INFO] Training RNN ...")
rnn_model.fit(X, y, epochs=1000, verbose=0)

# ─── TRANSFORMER (BERT) ───
print("[INFO] Loading Transformer ...")
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

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
    # Appending a clarifying sentence is often more effective for BERT
    prompt = text.strip()
    if not prompt.endswith("."):
        prompt += "."
    prompt += " The language I speak is [MASK]."
    
    results = fill_mask(prompt, top_k=100)
    for r in results:
        token = r["token_str"].strip().lower()
        if token in languages:
            return token
    return "unknown"

def predict(text):
    print("\nInput:", text)
    print("-> RNN Prediction         :", rnn_predict(text))
    print("-> Transformer Prediction :", transformer_predict(text))
    print("-" * 60)

# ─── TEST CASES ───
predict("I grew up in France so I speak fluent")
predict("I grew up in France many years ago when my family moved there for work and after living there for a long time I speak fluent")