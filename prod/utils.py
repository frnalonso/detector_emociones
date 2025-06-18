import torch
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN

emotion_labels = ['enojado', 'disgustado', 'temeroso', 'feliz', 'triste', 'sorprendido', 'neutral']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)

def load_model(model_class, path='prod/modelo.pth'):
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    model.to(device)
    return model

def detect_face(image):
    face = mtcnn(image)
    if face is None:
        return None
    resize = transforms.Resize((128, 128))
    return resize(face).unsqueeze(0).to(device)

def predict_emotion(model, face_tensor):
    with torch.no_grad():
        output = model(face_tensor)
        _, predicted = torch.max(output, 1)
    return emotion_labels[predicted.item()]
