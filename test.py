import os
import urllib.request
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# 1. Device and Label Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'anger', 'neutral']

# 2. Load Face Cascade with Auto-Download Fallback
cascade_path = "haarcascade_frontalface_default.xml"

if not os.path.exists(cascade_path):
    print("Downloading 'haarcascade_frontalface_default.xml'...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_path)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise IOError("Failed to load Cascade Classifier XML file.")

# 3. Reconstruct PyTorch Model Architecture
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))

try:
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    print("Successfully loaded 'best_model.pth'")
except FileNotFoundError:
    print("Error: 'best_model.pth' not found in current directory.")
    exit()

model = model.to(device)
model.eval()

# 4. Define Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Start Video Capture
cap = cv2.VideoCapture(0)

print("\nCamera feed started... Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_face)

        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)
            label = f"{class_names[pred_idx]}: {conf.item()*100:.1f}%"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('AI Mood Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()