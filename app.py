import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Face Detection App", layout="wide")

st.title("🎬 Face Detection System")
st.write("Upload an image to detect faces")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file is not None:

    image = Image.open(file)
    img = np.array(image)

    st.image(img, caption="Original Image", use_container_width=True)

    # load model
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # draw boxes
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    st.subheader(f"Faces Detected: {len(faces)}")
    st.image(img, use_container_width=True)

    if len(faces) > 0:
        st.success("Face detection successful ✅")
    else:
        st.error("No faces detected ❌")