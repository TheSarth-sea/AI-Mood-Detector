import time
from collections import deque, Counter
import numpy as np
import av
import cv2
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
from facenet_pytorch import MTCNN

st.set_page_config(page_title="AI Mood Detector", layout="wide")
st.title("Live AI Mood Detector")
st.write("Real-time emotion tracking powered by PyTorch.")


# ---------------------------------------------------------------------------
# Model loading (cached so it only happens once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 7)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device)
    model.eval()

    face_detector = MTCNN(keep_all=True, device=device)

    return model, face_detector, device


model, face_detector, device = load_models()
idx_to_label = {0: 'surprise', 1: 'fear', 2: 'disgust', 3: 'happy', 4: 'sad', 5: 'anger', 6: 'neutral'}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ---------------------------------------------------------------------------
# Shared state between the video processor thread and the main Streamlit thread.
# streamlit-webrtc runs recv() in a separate thread, so we use a plain
# module-level object (safe here since there's only ever one active stream)
# for the log the processor writes to.
# ---------------------------------------------------------------------------
class MoodLog:
    def __init__(self, maxlen=300):
        self.timeline = deque(maxlen=maxlen)  # (timestamp, emotion) — dominant emotion per frame
        self.current_counts = Counter()       # emotion -> count, for "who's in frame right now"


mood_log = MoodLog()


# ---------------------------------------------------------------------------
# Video processor
# ---------------------------------------------------------------------------
class MoodProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        boxes, _ = face_detector.detect(pil_img)

        frame_counts = Counter()

        if boxes is not None:
            for box in boxes:
                x, y, x2, y2 = [int(b) for b in box]

                h_img, w_img, _ = img.shape
                x, y = max(0, x), max(0, y)
                x2, y2 = min(w_img, x2), min(h_img, y2)

                if x2 - x > 20 and y2 - y > 20:
                    face_roi = rgb_img[y:y2, x:x2]
                    face_pil = Image.fromarray(face_roi)

                    tensor_img = transform(face_pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(tensor_img)
                        probs = torch.softmax(outputs, dim=1)
                        conf, pred_idx = torch.max(probs, dim=1)

                    emotion = idx_to_label[pred_idx.item()]
                    confidence = conf.item() * 100
                    frame_counts[emotion] += 1

                    cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{emotion} ({confidence:.1f}%)"
                    cv2.putText(img, label_text, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Update shared log — timeline tracks the single dominant emotion this frame
        # (simplest useful signal for one person; still a meaningful "room mood"
        # pulse when several people are in frame)
        if frame_counts:
            dominant = frame_counts.most_common(1)[0][0]
            mood_log.timeline.append((time.time(), dominant))
        mood_log.current_counts = frame_counts

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------------------------------------------------------
# Layout: video on the left, dashboard on the right
# ---------------------------------------------------------------------------
col_video, col_dashboard = st.columns([2, 1])

with col_video:
    webrtc_streamer(
        key="mood-detector",
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_processor_factory=MoodProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

with col_dashboard:
    st.subheader("Right now")
    now_placeholder = st.empty()

    st.subheader("Mood over time")
    timeline_placeholder = st.empty()

    st.subheader("Session summary")
    summary_placeholder = st.empty()

    st.button("Refresh dashboard")  # forces a rerun so the placeholders above redraw with the latest log data

    # "Right now" — how many people, what emotions
    if mood_log.current_counts:
        now_placeholder.bar_chart(dict(mood_log.current_counts))
    else:
        now_placeholder.info("No face currently detected.")

    # Timeline — dominant emotion per frame, plotted as a simple event stream
    if mood_log.timeline:
        recent = list(mood_log.timeline)[-100:]  # last 100 logged frames
        emotion_order = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'anger', 'neutral']
        emotion_to_y = {e: i for i, e in enumerate(emotion_order)}
        chart_data = {
            "time": [t for t, _ in recent],
            "emotion_level": [emotion_to_y[e] for _, e in recent],
        }
        timeline_placeholder.line_chart(chart_data, x="time", y="emotion_level")
        st.caption("Y-axis: " + ", ".join(f"{i}={e}" for i, e in enumerate(emotion_order)))
    else:
        timeline_placeholder.info("Timeline will populate once a face is detected.")

    # Session summary — overall distribution across the whole session so far
    if mood_log.timeline:
        all_emotions = [e for _, e in mood_log.timeline]
        counts = Counter(all_emotions)
        dominant_overall = counts.most_common(1)[0][0]
        summary_placeholder.write(
            f"**Dominant mood this session:** {dominant_overall}\n\n"
            + "\n".join(f"- {e}: {c} frames" for e, c in counts.most_common())
        )
    else:
        summary_placeholder.info("No data yet.")