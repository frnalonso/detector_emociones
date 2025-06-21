import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from streamlit_webrtc import VideoProcessorBase
from utils import detect_face, predict_emotion, EMOTION_LABELS
from torchvision.models import resnet18

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelo cargado una sola vez
def load_model():
    model = resnet18(num_classes=7)
    model.load_state_dict(torch.load("prod/modelo.pth", map_location=device))
    model.eval().to(device)
    return model

# Detector de rostros global
mtcnn = MTCNN(keep_all=False, device=device)

# Transformación uniforme
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Procesador de video para streamlit-webrtc
class EmotionDetector(VideoProcessorBase):
    def __init__(self, model):
        self.model = model
        self.frame_count = 0

    def recv(self, frame):
        
        img = frame.to_ndarray(format="bgr24")

        # Disminuimos resolución para mejorar performance
        img_small = cv2.resize(img, (640, 480))
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Procesar solo 1 de cada 5 frames
        self.frame_count += 1
        if self.frame_count % 5 != 0:
            return img_small

        face_tensor = detect_face(pil_img)
        if face_tensor is not None:
            
            emotion = predict_emotion(self.model, face_tensor)
            boxes, _ = mtcnn.detect(pil_img)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(img_small, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_small, emotion.upper(), (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return img_small
