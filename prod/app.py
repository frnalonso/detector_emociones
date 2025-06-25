import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
from PIL import Image
import torch
from torchvision.models import resnet18, resnet50
from utils import detect_face, predict_emotion, load_model, EMOTION_LABELS, show_tensor_image
from facenet_pytorch import MTCNN
from torchvision import transforms

# Configuración
st.title("Clasificador de Emociones por Imagen y Webcam")
st.write("Seleccioná 'Imagen y carga una imagen para detectar la emoción' o 'Webcam' para iniciar la detección de emociones.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(lambda: resnet50(num_classes=7))
mtcnn = MTCNN(keep_all=False, device=device)


# Variables de estado globales
last_emotion = "Detectando..."
last_confidence = 0.0
last_boxes = None
frame_count = 0
miss_counter = 0
MISS_LIMIT = 10  # N° de frames sin detección antes de borrar emoción

# Función callback para procesar cada frame
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global last_emotion, last_confidence, last_boxes, frame_count, miss_counter

    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    frame_count += 1

    # Solo procesamos cada 5 frames
    if frame_count % 5 == 0:
        face = detect_face(pil_img)
        if face is not None:
            boxes, _ = mtcnn.detect(pil_img)
            if boxes is not None:
                last_boxes = boxes
                last_emotion, last_confidence = predict_emotion(model, face)
                miss_counter = 0
        else:
            miss_counter += 1
            if miss_counter >= MISS_LIMIT:
                last_emotion = "Sin detección"
                last_confidence = 0.0
                last_boxes = None

    # Mostrar la última emoción detectada
    if last_boxes is not None:
        for box in last_boxes:
            (x1, y1, x2, y2) = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{last_emotion.upper()} ({last_confidence * 100:.1f}%)"
            cv2.putText(img, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# Interfaz
modo = st.selectbox("Elegí el modo de entrada:", ["Imagen", "Webcam"])

if modo == "Imagen":
    uploaded_file = st.file_uploader("Subí una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagen cargada", use_column_width=True)

        face = detect_face(image)
        if face is None:
            st.error("No se detectó ningún rostro.")
        else:
            emotion, confidence = predict_emotion(model, face)
            st.success(f"Emoción detectada: **{emotion.upper()} ({confidence*100:.1f}%)**")
        
        # Mostrar rostro procesado
        st.subheader("Rostro que se pasa al modelo")
        show_tensor_image(face)
        #img = transforms.ToPILImage()(face.squeeze(0).cpu())
        #st.image(img, caption="Rostro procesado (224x224)", width=200)

elif modo == "Webcam":
    webrtc_streamer(
        key="webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
