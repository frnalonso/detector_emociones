import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
from PIL import Image
import torch
from torchvision.models import resnet18
from utils import detect_face, predict_emotion, load_model, EMOTION_LABELS
from facenet_pytorch import MTCNN

# Configuración
st.title("Detector de Emociones en Tiempo Real")
st.write("Seleccioná 'Webcam' para iniciar la detección de emociones.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(lambda: resnet18(num_classes=7))
mtcnn = MTCNN(keep_all=False, device=device)

# Función callback para procesar cada frame
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    face = detect_face(pil_img)
    if face is not None:
        emotion = predict_emotion(model, face)
        boxes, _ = mtcnn.detect(pil_img)
        if boxes is not None:
            for box in boxes:
                (x1, y1, x2, y2) = box.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, emotion.upper(), (x1, y1 - 10),
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
            emotion = predict_emotion(model, face)
            st.success(f"Emoción detectada: **{emotion.upper()}**")

elif modo == "Webcam":
    st.warning("Esta opción solo funciona si ejecutás Streamlit localmente.")
    webrtc_streamer(
        key="webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
