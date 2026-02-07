import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase
import threading
import pyttsx3
import time
from datetime import datetime
import os
import pandas as pd

# --- Page Config ---
st.set_page_config(
    page_title="SignSpeak AI - Real-time Sign Language Translator",
    page_icon="🖐️",
    layout="wide"
)

# --- Design System (Custom CSS) ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stApp {
        background: radial-gradient(circle at top right, #1a1a2e, #0f0f1b);
    }
    .status-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    .prediction-text {
        font-size: 3rem;
        font-weight: 800;
        color: #00d2ff;
        text-shadow: 0 0 10px rgba(0, 210, 255, 0.5);
        text-align: center;
    }
    .caption-box {
        background: rgba(0, 0, 0, 0.7);
        color: #ffffff;
        padding: 10px 20px;
        border-radius: 10px;
        font-family: 'Courier New', Courier, monospace;
        text-align: center;
        margin-top: 10px;
    }
    .log-container {
        max-height: 300px;
        overflow-y: auto;
        font-family: inherit;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants & Initialization ---
MODEL_PATH = "sign_model.keras"
LOG_FILE = "sign_recognition_log.csv"

# Initialize TTS Engine
def init_tts():
    try:
        # On Windows, pyttsx3 can sometimes have permission issues with the SAPI5 driver
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        return engine
    except Exception as e:
        st.warning(f"⚠️ TTS (Text-to-Speech) is unavailable: {e}. The app will continue without voice.")
        return None

# Dictionary for mapping class indices to labels
LABELS = {0: "Hello", 1: "Thank You", 2: "I Love You", 3: "Yes", 4: "No", 5: "Please"}

# --- Model Loading ---
@st.cache_resource
def load_sign_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None
    else:
        st.info(f"💡 Info: '{MODEL_PATH}' not found. Using simulation mode.")
        return None

model = load_sign_model()

# --- Logging Utils ---
def log_recognition(text):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_data = {'Timestamp': timestamp, 'Recognized Sign': text}
        
        if not os.path.exists(LOG_FILE):
            pd.DataFrame([log_data]).to_csv(LOG_FILE, index=False)
        else:
            pd.DataFrame([log_data]).to_csv(LOG_FILE, mode='a', header=False, index=False)
    except PermissionError:
        st.error(f"❌ Permission Denied: Could not write to '{LOG_FILE}'. Please close the file if it is open in another program.")
    except Exception as e:
        st.error(f"❌ Logging error: {e}")

# --- State Management ---
if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = ""
if 'prediction_history' not in st.session_state:
    st.session_state['prediction_history'] = []

# --- Video Processing Component ---
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.labels = LABELS
        self.last_speech_time = 0
        self.speech_cooldown = 2.0  # Seconds
        self.current_prediction = ""
        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # --- Preprocessing ---
        # Resize to model input shape (Assuming 224x224 for common CNNs)
        try:
            processed_img = cv2.resize(img, (224, 224))
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            processed_img = np.expand_dims(processed_img, axis=0) / 255.0
            
            # --- Prediction ---
            if self.model:
                preds = self.model.predict(processed_img, verbose=0)
                class_idx = np.argmax(preds)
                confidence = np.max(preds)
                
                if confidence > 0.7:  # Confidence threshold
                    new_pred = self.labels.get(class_idx, "Unknown")
                else:
                    new_pred = "Scanning..."
            else:
                # Dummy prediction for demo if no model exists
                # In real use, this would be the actual prediction
                new_pred = "Awaiting Model"
            
            with self.lock:
                self.current_prediction = new_pred
                
            # Add text to frame for feedback
            cv2.putText(img, f"Predict: {new_pred}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Frame processing error: {e}")

        return frame.from_ndarray(img, format="bgr24")

# --- UI Layout ---
def main():
    st.title("🖐️ SignSpeak AI")
    st.subheader("Bridging Communication with Motion Recognition")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        ctx = webrtc_streamer(
            key="sign-recognition",
            video_processor_factory=SignLanguageProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False},
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        caption_placeholder = st.empty()

    with col2:
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.write("### Recognized Output")
        prediction_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.write("### 📜 Session Logs")
        log_placeholder = st.empty()
        
        if st.button("Clear Logs"):
            if os.path.exists(LOG_FILE):
                os.remove(LOG_FILE)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Background Update Loop ---
    # Since streamlit_webrtc runs in a separate thread, we need to pull 
    # the predictions to update the UI and handle TTS/Logging
    
    engine = init_tts()
    
    while ctx.state.playing:
        if ctx.video_processor:
            with ctx.video_processor.lock:
                pred = ctx.video_processor.current_prediction
            
            if pred and pred != st.session_state['last_prediction'] and pred not in ["Scanning...", "Awaiting Model", ""]:
                st.session_state['last_prediction'] = pred
                
                # Update Predict View
                prediction_placeholder.markdown(f'<div class="prediction-text">{pred}</div>', unsafe_allow_html=True)
                
                # Update Captions
                caption_placeholder.markdown(f'<div class="caption-box">CAPTIONS: {pred}</div>', unsafe_allow_html=True)
                
                # TTS (Non-blocking if possible, but pyttsx3 is mostly synchronous)
                # In Streamlit, this runs on the server.
                if engine:
                    engine.say(pred)
                    engine.runAndWait()
                
                # Logging
                log_recognition(pred)
                
                # Update UI logs
                if os.path.exists(LOG_FILE):
                    df_logs = pd.read_csv(LOG_FILE)
                    log_placeholder.dataframe(df_logs.tail(10), use_container_width=True)
            
            elif not pred:
                prediction_placeholder.markdown(f'<div class="prediction-text">---</div>', unsafe_allow_html=True)

        time.sleep(0.5)

if __name__ == "__main__":
    main()
