import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model


# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(
    page_title="Face Detection System",
    page_icon="😀",
    layout="wide",
)

# ==================================
# DICE FUNCTIONS
# ==================================
def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)

    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


# ==================================
# LOAD MODEL
# ==================================
@st.cache_resource
def load_my_model():
    model_path = Path("best_model.h5")
    if not model_path.exists():
        raise FileNotFoundError(
            "best_model.h5 not found. Put the model file in the same folder as this app."
        )

    return load_model(
        model_path,
        custom_objects={
            "dice_loss": dice_loss,
            "dice_coefficient": dice_coefficient,
        },
    )


# ==================================
# APP UI
# ==================================
st.title("😀 Face Detection System")
st.write("Upload an image and detect face using U-Net + MobileNetV2")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

predict_clicked = st.button("Predict")

if uploaded_file is None:
    st.info("Please upload an image to start.")
    st.stop()

if not predict_clicked:
    st.info("Click **Predict** after uploading an image.")
    st.image(Image.open(uploaded_file), caption="Uploaded Image", use_container_width=True)
    st.stop()

# ==================================
# PREPROCESS IMAGE
# ==================================
start_time = time.time()

image = Image.open(uploaded_file).convert("RGB")
original = np.array(image)

resized = cv2.resize(original, (256, 256))
input_image = resized.astype(np.float32) / 255.0
input_image = np.expand_dims(input_image, axis=0)

# ==================================
# PREDICTION
# ==================================
try:
    model = load_my_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

prediction = model.predict(input_image, verbose=0)
mask = prediction[0]

# Handle prediction shape safely
if mask.ndim == 3 and mask.shape[-1] == 1:
    mask_2d = mask[:, :, 0]
elif mask.ndim == 2:
    mask_2d = mask
else:
    mask_2d = np.squeeze(mask)

confidence_score = float(np.max(mask_2d))

binary_mask = (mask_2d > 0.5).astype(np.uint8)
mask_uint8 = (binary_mask * 255).astype(np.uint8)

# ==================================
# FIND FACE REGIONS
# ==================================
contours, _ = cv2.findContours(
    mask_uint8,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

result = original.copy()
face_count = 0

scale_x = original.shape[1] / 256
scale_y = original.shape[0] / 256

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area <= 100:
        continue

    x, y, w, h = cv2.boundingRect(cnt)
    x = int(x * scale_x)
    y = int(y * scale_y)
    w = int(w * scale_x)
    h = int(h * scale_y)

    face_count += 1

    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(
        result,
        f"Face {face_count}",
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

processing_time = round(time.time() - start_time, 3)

# ==================================
# SHOW IMAGES
# ==================================
st.subheader("Detection Result")
col1, col2 = st.columns(2)

with col1:
    st.image(original, caption="Original Image", use_container_width=True)

with col2:
    st.image(result, caption="Detected Face Bounding Box", use_container_width=True)

# ==================================
# DASHBOARD
# ==================================
st.subheader("📊 Performance Dashboard")
c1, c2, c3, c4 = st.columns(4)

c1.metric("Processing Time", f"{processing_time} sec")
c2.metric("Confidence Score", f"{confidence_score:.2f}")
c3.metric("Faces Detected", face_count)
c4.metric("Status", "Detected" if face_count > 0 else "No Face")

# ==================================
# STATUS
# ==================================
if face_count > 0:
    st.success(f"✅ {face_count} Face(s) Detected")
else:
    st.warning("⚠ No Face Detected")

# ==================================
# LOG TABLE
# ==================================
log_data = pd.DataFrame(
    {
        "Date Time": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Processing Time": [processing_time],
        "Confidence Score": [round(confidence_score, 3)],
        "Faces Detected": [face_count],
    }
)

st.subheader("Detection Log")
st.dataframe(log_data, use_container_width=True)

csv = log_data.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Detection Log",
    data=csv,
    file_name="detection_log.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption("Face Detection using U-Net + MobileNetV2")
