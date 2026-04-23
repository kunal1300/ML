import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

st.title("Student Pass / Fail Prediction using Logistic Regression")

data = {
    "Marks":[10,20,30,34,35,40,50,60],
    "Result":[0,0,0,0,1,1,1,1]     # 0-Fail, 1-Pass
}

df = pd.DataFrame(data)
st.write(df)

# Train Model
X = df[["Marks"]]
y = df["Result"]

model = LogisticRegression()
model.fit(X,y)

# Prediction
st.subheader("Predict Result")

marks = st.number_input("Enter Student Marks",
                        min_value = 0.0,
                        max_value = 100.00,
                        value = 25.0)

if st.button("Predict"):
    input_data = np.array([[marks]])    # [[45]] --> 1 row 1 column
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success("Prediction: PASS")
    else:
        st.error("Prediction: FAIL")

    st.write(f"Probability of PASS: {probability[0][1]:.2f}")


# Graph
st.subheader("Graph")

plt.figure()

plt.scatter(df["Marks"], df["Result"], label = "Data")

x_range = np.linspace(0, 100, 100).reshape(-1,1)
y_prob = model.predict_proba(x_range)[:, 1]

plt.plot(x_range, y_prob, label = "Logistic Curve")

plt.xlabel("Marks")
plt.ylabel("Probability of Pass")
plt.legend()

st.pyplot(plt)