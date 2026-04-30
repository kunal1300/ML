import streamlit as st

import numpy as np

import random

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from keras.models import Sequential

from keras.layers import Dense

random.seed(1)

np.random.seed(1)

tf.random.set_seed(1)

# -----------------------------

# Dataset (STRONGLY NON-LINEAR)

# -----------------------------

data = {

    "Study Hours": [1, 2, 3, 4, 5, 6, 7],

    "Marks": [10, 20, 35, 80, 75, 90, 98]

}

# data = {

#     "Study Hours": [1, 2, 3, 4, 5, 6, 7],

#     "Marks": [10, 20, 30, 40, 50, 60, 70]

# }

df = pd.DataFrame(data)

X = np.array(df["Study Hours"]).reshape(-1, 1)

y = np.array(df["Marks"])

# -----------------------------

# Linear Regression

# -----------------------------

ml_model = LinearRegression()

ml_model.fit(X, y)

ml_predictions = ml_model.predict(X)

ml_r2 = r2_score(y, ml_predictions)

# -----------------------------

# ANN Model

# -----------------------------

ann_model = Sequential()

ann_model.add(Dense(100, activation='relu', input_dim=1))

ann_model.add(Dense(100, activation='relu'))

ann_model.add(Dense(100, activation='relu'))

ann_model.add(Dense(1))

ann_model.compile(optimizer='adam', loss='mse')

ann_model.fit(X, y, epochs=800, verbose=0)

ann_predictions = ann_model.predict(X).flatten()

ann_r2 = r2_score(y, ann_predictions)

# -----------------------------

# UI

# -----------------------------

st.title(" Machine Learning vs Deep Learning (ANN)")

st.subheader(" Training Data")

st.dataframe(df)

st.subheader(" Training Accuracy (R² Score)")

col1, col2 = st.columns(2)

with col1:

    st.metric("Linear Regression", f"{ml_r2:.4f}")

with col2:

    st.metric("ANN Model", f"{ann_r2:.4f}")

hours = st.number_input("Enter Study Hours:", min_value=0.0, step=0.5)

if st.button("Predict Marks"):

    ml_pred = ml_model.predict([[hours]])

    ann_pred = ann_model.predict(np.array([[hours]]))

    st.subheader(" Predictions")

    st.write(f"**Linear Regression:** {ml_pred[0]:.2f}")

    st.success(f" ANN Prediction: {ann_pred[0][0]:.2f}")

# -----------------------------

# Charts

# -----------------------------

st.subheader(" Model Performance")

# Linear Regression Plot

fig1 = plt.figure()

plt.scatter(df["Study Hours"], y, label="Actual Marks")

plt.scatter(df["Study Hours"], ml_predictions, label="Predicted Marks")

plt.xlabel("Study Hours")

plt.ylabel("Marks")

plt.title("Linear Regression")

plt.legend()

st.pyplot(fig1)

# ANN Plot

fig2 = plt.figure()

plt.scatter(df["Study Hours"], y, label="Actual Marks")

plt.scatter(df["Study Hours"], ann_predictions, label="Predicted Marks")

plt.xlabel("Study Hours")

plt.ylabel("Marks")

plt.title("ANN Model")

plt.legend()

st.pyplot(fig2)