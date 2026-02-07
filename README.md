# Mini-Project
# 🖐️ Real-Time Sign Language Recognition

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)

**SignSpeak AI** is a state-of-the-art web application that bridges the communication gap using computer vision and deep learning. It captures real-time video feed, translates sign language gestures into text, and provides an accessible interface for inclusive communication.

---

## 🌟 Key Features

- **🚀 Low-Latency Recognition**: Leverages `streamlit-webrtc` for high-performance, browser-based video streaming.
- **🧠 Deep Learning Engine**: Powered by a Convolutional Neural Network (CNN) trained on sign language datasets.
- **🎙️ Accessible Feedback**: (Upcoming/Beta) Integrated Text-to-Speech (TTS) for vocalizing recognized signs.
- **📊 Live Analytics**: Real-time probability tracking for each gesture class.
- **📝 Session Logging**: Automatic logging of recognized gestures to CSV for history and analysis.
- **🎨 Modern UI**: Sleek, dark-themed dashboard built with Streamlit's latest design system.

---

## 🛠️ Technical Architecture

The system follows a modular architecture:

1.  **Frontend**: Streamlit-based web interface with custom CSS for a premium feel.
2.  **Video Pipeline**: `streamlit-webrtc` handles the camera stream and frame synchronization.
3.  **Inference Engine**: A TensorFlow/Keras model processes frames in real-time.
4.  **Processor**: Custom `VideoProcessorBase` class handles image resizing, normalization, and model prediction.

---

## 🚀 Getting Started

### 📋 Prerequisites

- Python 3.9 or higher
- A working webcam

### 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Mini-Project.git
   cd mimi-project
   ```

2. **Set up a virtual environment (Recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Usage

1. **Initialize the Model**:
   If you don't have a trained model yet, run the dummy generator:
   ```bash
   python create_model.py
   ```

2. **Launch the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Interact**:
   - Grant camera permissions in your browser.
   - Click the **Start** button in the UI.
   - Perform gestures in front of the camera (e.g., "Hello", "Thank You").
   - View live predictions and raw data on the right sidebar.

---

## 📁 Project Structure

| File | Description |
| :--- | :--- |
| `app.py` | Core Streamlit application and UI logic. |
| `sign_model.keras` | Trained TensorFlow model file. |
| `create_model.py` | Utility to create a placeholder model for testing. |
| `check_model.py` | Diagnostic script for model verification. |
| `requirements.txt` | List of Python dependencies. |
| `sign_recognition_log.csv` | Auto-generated log of recognized signs. |

---

## 🧪 Model Customization

To use your own model:
1. Export your model in `.keras` or `.h5` format.
2. Place it in the root directory and name it `sign_model.keras`.
3. Update the `LABELS` dictionary in `app.py` to match your classes.
4. (Optional) Adjust the preprocessing logic in `SignLanguageProcessor.recv` if your model requires different input shapes.

---

## ⚠️ Troubleshooting

- **Camera not starting**: Ensure no other application is using your webcam and that you've granted browser permissions.
- **Low FPS**: Recognition performance depends on CPU/GPU power. Try closing other resource-intensive apps.
- **Model Load Error**: Ensure `tensorflow` version compatibility. Running `check_model.py` can help diagnose loading issues.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---


