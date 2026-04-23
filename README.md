# 🎬 Real-Time Face Detection System

This project implements a real-time face detection system using OpenCV and Streamlit. It utilizes a dataset containing images with bounding box annotations to understand and visualize face locations, and performs detection on new images.

---

## 🚀 Features

* 📸 Upload image and detect faces
* 🟩 Visualize bounding boxes on detected faces
* 📊 Exploratory Data Analysis (EDA) on dataset
* ⚡ Fast face detection using OpenCV Haar Cascade
* 🌐 Interactive web app using Streamlit

---

## 🧠 Project Workflow

1. **Data Loading**

   * Loaded `.npy` dataset containing images and annotations

2. **Exploratory Data Analysis (EDA)**

   * Analyzed dataset structure
   * Visualized sample images
   * Studied face distribution and bounding boxes

3. **Visualization**

   * Drew ground truth bounding boxes from annotations

4. **Face Detection**

   * Used OpenCV Haar Cascade for detecting faces

5. **Deployment**

   * Built an interactive Streamlit web application

---

## 🛠 Tech Stack

* Python
* OpenCV
* Streamlit
* NumPy
* Matplotlib

---

## ▶️ How to Run the Project

### 🔹 Install dependencies

```bash
pip install -r requirements.txt
```

### 🔹 Run Streamlit app

```bash
python -m streamlit run app.py
```

---

## 📁 Project Structure

```
Face-Detection-Project/
│
├── app.py
├── requirements.txt
├── face_detection_notebook.ipynb
├── README.md
├── sample_output.png
```

---

## 📸 Output

The application detects faces in uploaded images and displays bounding boxes around them.

---

## 🎯 Future Improvements

* 🔥 Deep learning-based face detection (CNN/DNN)
* 🎥 Real-time webcam face detection
* 📈 Improve detection accuracy
* ☁️ Deploy app online

---

## 👨‍💻 Author

Shubham Rathore

---

## ⭐ Acknowledgment

This project was developed as part of a machine learning and computer vision learning program.
