import cv2
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from prod.utils import emotion_labels




# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("dispositivo utilizado: ",device)

# Cargar modelo
model = resnet18(num_classes=7)
model.load_state_dict(torch.load("prod/modelo.pth", map_location=device))
model.eval().to(device)

# Transformación
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Detector de rostro
mtcnn = MTCNN(keep_all=False, device=device)

# Captura de webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face = mtcnn(img)

    if face is not None:
        face = transform(transforms.ToPILImage()(face)).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(face)
            _, predicted = torch.max(output, 1)
            emotion = emotion_labels[predicted.item()]

        boxes, _ = mtcnn.detect(img)
        if boxes is not None:
            for box in boxes:
                (x1, y1, x2, y2) = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, emotion.upper(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)

    cv2.imshow('Detector de emociones', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()