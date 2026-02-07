# Mini-Project
# 🖐️ AI - Sign Language Recognition System

This project is a real-time sign language recognition system built with **Streamlit**, **TensorFlow**, and **WebRTC**.

## ✨ Features
- **Real-time Camera Capture**: Uses `streamlit-webrtc` for low-latency browser-based video.
- **Motion Recognition**: Integrates a pre-trained `.keras` model to classify frames.
- **TTS (Text-to-Speech)**: Automatically speaks the recognized sign using `pyttsx3`.
- **Live Captions**: Displays the recognized text as an overlay/caption on the UI.
- **Logging**: Track every recognized gesture in a CSV log file for later review.

## 🚀 Getting Started

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run the Application
Start the Streamlit server:
```bash
streamlit run app.py
```

### 3. Using Your Own Model
By default, the app looks for `sign_model.keras`. 
- Replace `sign_model.keras` with your actual trained model.
- Update the `LABELS` dictionary in `app.py` to match your model's classification indices.
- Ensure the image preprocessing in `SignLanguageProcessor.recv` (default 224x224) matches your model's input requirements.

## 📁 Project Structure
- `app.py`: Main application logic and UI.
- `create_model.py`: Script to generate a placeholder model for testing.
- `sign_model.keras`: The trained neural network model.
- `sign_recognition_log.csv`: Automatically generated session logs.
- `requirements.txt`: Python package dependencies.

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Deep Learning**: TensorFlow / Keras
- **Computer Vision**: OpenCV
- **Audio**: pyttsx3 (TTS)
- **WebRTC**: streamlit-webrtc
