import streamlit as st
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import accuracy_score

# -----------------------------
# Freeze randomness ✅
# -----------------------------
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# -----------------------------
# Load MNIST Dataset
# -----------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_train = X_train / 255.0   # 60,000 training images
X_test = X_test / 255.0     # 10,000 testing images

# Reshape for CNN
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# -----------------------------
# Build Vanilla CNN
# -----------------------------
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=3, verbose=0)

# Predictions for accuracy
y_pred = np.argmax(model.predict(X_test), axis=1)
acc = accuracy_score(y_test, y_pred)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Vanilla CNN – MNIST Digit Classifier")

st.subheader("📊 Model Accuracy")
st.metric("Test Accuracy", f"{acc:.4f}")

# Select image
index = st.slider("Select Test Image", 0, 9999, 0)

image = X_test[index].reshape(28,28)
true_label = y_test[index]

# Plot image
fig = plt.figure()
plt.imshow(image)
plt.title(f"Actual Label: {true_label}")
plt.axis('off')

st.pyplot(fig)

# Prediction
if st.button("Predict Digit"):

    prediction = model.predict(X_test[index].reshape(1,28,28,1))
    predicted_label = np.argmax(prediction)

    st.subheader("🎯 Prediction")
    st.success(f"Predicted Digit: {predicted_label}")