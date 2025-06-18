import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
from torchvision.models import resnet18
from utils import load_model, detect_face, predict_emotion

from facenet_pytorch import MTCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instanciar el detector de rostros
mtcnn = MTCNN(keep_all=False, device=device)


st.title("Clasificador de Emociones Faciales")
st.write("Subí una imagen o activá la webcam para detectar emociones.")

modo = st.selectbox("Elegí el modo de entrada:", ["Imagen", "Webcam"])

model = load_model(lambda: resnet18(num_classes=7))

if modo == "Imagen":
    uploaded_file = st.file_uploader("Elegí una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Imagen cargada', use_column_width=True)

        with st.spinner('Detectando rostro y emoción...'):
            face_tensor = detect_face(image)

            if face_tensor is None:
                st.error("No se detectó ningún rostro en la imagen.")
            else:
                emocion = predict_emotion(model, face_tensor)
                st.success(f"Emoción detectada: **{emocion.upper()}**")

elif modo == "Webcam":
    st.warning("Esta opción solo funciona si ejecutás Streamlit localmente.")
    start_camera = st.button("Iniciar webcam")
    if start_camera:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face = detect_face(img)
            if face is not None:
                emocion = predict_emotion(model, face)
                boxes, _ = mtcnn.detect(img)
                if boxes is not None:
                    for box in boxes:
                        (x1, y1, x2, y2) = box.astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, emocion.upper(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, (0, 255, 0), 2)

            stframe.image(frame, channels="BGR")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()