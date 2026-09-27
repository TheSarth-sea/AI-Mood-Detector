import av
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from facenet_pytorch import MTCNN

st.title("Live AI Mood Detector")
st.write("Real-time emotion tracking powered by PyTorch.")

# Cache models to prevent reloading on every frame(EG)
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Emotion Model
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 7)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    # 2. Load PyTorch Face Detector
    face_detector = MTCNN(keep_all=True, device=device)
    
    return model, face_detector, device

model, face_detector, device = load_models()
idx_to_label = {0: 'surprise', 1: 'fear', 2: 'disgust', 3: 'happy', 4: 'sad', 5: 'anger', 6: 'neutral'}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MoodProcessor:
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        # Detect faces
        boxes, _ = face_detector.detect(pil_img)
        
        if boxes is not None:
            for box in boxes:
                # MTCNN returns bounding boxes as [x1, y1, x2, y2]
                x, y, x2, y2 = [int(b) for b in box]
                
                # Enforce frame boundaries
                h_img, w_img, _ = img.shape
                x, y = max(0, x), max(0, y)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                # Ensure the bounding box is large enough to process
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
                    
                    # Draw UI overlay
                    cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{emotion} ({confidence:.1f}%)"
                    cv2.putText(img, label_text, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="mood-detector",
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_processor_factory=MoodProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
