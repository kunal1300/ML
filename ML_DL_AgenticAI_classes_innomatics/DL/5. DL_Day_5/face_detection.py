import streamlit as st

import cv2 

st.title("Face Detection with OpenCV")

start = st.button("Start Camera")

face_cascade = cv2.CascadeClassifier(

    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

)

frame_placeholder = st.empty()

if start:

    cap= cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0),2)

        frame_placeholder.image(frame, channels="BGR")

    cap.release()