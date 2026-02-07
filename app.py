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
import av

# --- Page Config ---
st.set_page_config(
    page_title="Real-time Sign Language Translator",
    page_icon="🖐️",
    layout="wide"
)

# --- Design System ---
st.markdown("""
<style>
    .stApp { background: #0e1117; color: white; }
    .status-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 10px;
    }
    .prediction-text {
        font-size: 3.5rem;
        font-weight: 800;
        color: #00d2ff;
        text-align: center;
    }
    .debug-info { color: #888; font-family: monospace; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "sign_model.keras"
LOG_FILE = "sign_recognition_log.csv"
LABELS = {0: "Hello", 1: "Thank You", 2: "I Love You", 3: "Yes", 4: "No", 5: "Please"}

@st.cache_resource
def load_sign_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Model Load Error: {e}")
    return None

model = load_sign_model()

class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.current_prediction = "Initializing..."
        self.raw_preds = []
        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # Preprocess
            processed = cv2.resize(img, (224, 224))
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            processed = np.expand_dims(processed, axis=0) / 255.0
            
            if self.model:
                preds = self.model.predict(processed, verbose=0)[0]
                idx = np.argmax(preds)
                conf = float(preds[idx])
                
                with self.lock:
                    self.raw_preds = preds.tolist()
                    if conf > 0.4: # Lowered threshold to 40% for easier testing
                        self.current_prediction = LABELS.get(idx, "Unknown")
                    else:
                        self.current_prediction = f"Scanning... ({conf:.0%})"
            else:
                with self.lock: self.current_prediction = "Model Missing"
                
            cv2.putText(img, f"UI State: {self.current_prediction}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        except Exception as e:
            with self.lock: self.current_prediction = f"Error: {str(e)[:15]}"

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("🖐️ SignSpeak AI")
    
    col1, col2 = st.columns([3, 2])

    with col1:
        ctx = webrtc_streamer(
            key="sign-recognition",
            video_processor_factory=SignLanguageProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False},
        )
        caption_placeholder = st.empty()

    with col2:
        st.write("### 📺 Prediction")
        predict_placeholder = st.empty()
        
        show_debug = st.checkbox("Show Debug Data", value=True)
        debug_placeholder = st.empty()
        
        st.write("### 📜 Session Logs")
        log_placeholder = st.empty()

    # Update Loop
    if ctx.state.playing:
        last_log_time = 0
        while ctx.state.playing:
            if ctx.video_processor:
                with ctx.video_processor.lock:
                    pred = ctx.video_processor.current_prediction
                    raw = ctx.video_processor.raw_preds
                
                predict_placeholder.markdown(f'<div class="prediction-text">{pred}</div>', unsafe_allow_html=True)
                
                if show_debug and raw:
                    debug_str = " | ".join([f"{LABELS[i]}: {p:.1%}" for i, p in enumerate(raw)])
                    debug_placeholder.markdown(f'<div class="debug-info">RAW: {debug_str}<br>Time: {datetime.now().strftime("%H:%M:%S.%f")[:-4]}</div>', unsafe_allow_html=True)

                # Log if it's a solid prediction and hasn't been logged in the last 2 seconds
                if "Scanning" not in pred and "..." not in pred and time.time() - last_log_time > 2:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    log_entry = pd.DataFrame([{"Time": timestamp, "Sign": pred}])
                    if not os.path.exists(LOG_FILE): log_entry.to_csv(LOG_FILE, index=False)
                    else: log_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)
                    last_log_time = time.time()
                    
                if os.path.exists(LOG_FILE):
                    log_placeholder.dataframe(pd.read_csv(LOG_FILE).tail(5), use_container_width=True)

            time.sleep(0.1)
    else:
        predict_placeholder.info("Click 'Start' to begin camera stream.")

if __name__ == "__main__":
    main()
