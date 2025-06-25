import torch
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
import torch.nn.functional as F
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.pyplot as plt


EMOTION_LABELS = ['enojado', 'disgustado', 'temeroso', 'feliz', 'triste', 'sorprendido', 'neutral']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)

def load_model(model_class, path='prod/modelo.pth'):
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    model.to(device)
    return model

#def detect_face(image):
 #   face = mtcnn(image)
  #  if face is None:
   #     return None
    #resize = transforms.Resize((224, 224))
    #return resize(face).unsqueeze(0).to(device)




def detect_face(image):
    face = mtcnn(image)  # Devuelve un tensor [3, H, W]
    if face is None:
        return None
    face = face.unsqueeze(0).to(device)  # [1, 3, H, W]
    
    # Redimensionamos correctamente usando interpolate
    #face = F.interpolate(face, size=(224, 224), mode='bilinear', align_corners=False)
    return face


def show_tensor_image(tensor, title="Rostro procesado (224x224)"):
    tensor = tensor.squeeze(0).cpu().clamp(0, 1)  # (3, 224, 224)
    np_img = tensor.permute(1, 2, 0).numpy()      # Convertir a formato HWC
    fig, ax = plt.subplots()
    ax.imshow(np_img)
    ax.axis("off")
    ax.set_title(title)
    st.pyplot(fig)



def predict_emotion(model, face_tensor):
    with torch.no_grad():
        output = model(face_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        label = EMOTION_LABELS[predicted_class.item()]
        return label, confidence.item()