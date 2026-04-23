import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("Student Marks Prediction using Linear Regression")

data = {
    "Study Hours" : [1,2,3,4,5,6],
    "Marks":[35,45,55,65,75,85]
}

df = pd.DataFrame(data)
st.write(df)

# Train Model
X = df[["Study Hours"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X,y)

# Prediction
st.subheader("Predict Marks")

hours = st.number_input("Enter Study Hours", min_value = 0.0, max_value = 12.0, value = 3.5)

if st.button("Predict"):
    prediction = model.predict([[hours]])
    st.success(f"Predicted Marks: {prediction[0]:.2f}")

# Graph
st.subheader("Graph")

plt.scatter(df["Study Hours"], df["Marks"])

x_range = np.linspace(0, 12, 100)
y_range = model.predict(x_range.reshape(-1,1))

plt.plot(x_range, y_range)

plt.xlabel("Study Hours")
plt.ylabel("Marks")

st.pyplot(plt)