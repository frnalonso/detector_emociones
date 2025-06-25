#modelo pre-entrenado ResNET50
model = models.resnet50(pretrained=True)

#modelo pre-entrenado ResNET18
# Cargar ResNet18 con pesos preentrenados
model = models.resnet18(pretrained=True)

# Configuración
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 7

# Cargar modelo ResNet50
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Pérdida con pesos si los tenés definidos
class_weights = torch.tensor([1.0, 2.5, 2.0, 0.7, 1.0, 1.5, 1.2], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizador
optimizer = optim.Adam(model.parameters(), lr=0.0001)

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

class_weights = torch.tensor([1.0, 2.5, 2.0, 0.7, 1.0, 1.5, 1.2], device=device)
# Función de pérdida con pesos
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam( model.parameters(), lr=0.0001)

def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, num_epochs):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        # Entrenamiento
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validación
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Gráfico final
    epochs = np.arange(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Acc")
    plt.plot(epochs, val_accuracies, label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()

train_model(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, num_epochs=5)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Precisión en test: {100 * correct / total:.2f}%")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support

# Recolectamos predicciones y verdaderos
y_true = []
y_pred = []

model.eval()
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Etiquetas de emociones
labels_text = ['enojado', 'disgustado', 'temeroso', 'feliz', 'triste', 'sorprendido', 'neutral']

# Matriz de Confusión
cm = confusion_matrix(y_true, y_pred, labels=range(len(labels_text)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_text)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.show()

# Reporte detallado de métricas
print("\n📊 Reporte de Clasificación:\n")
print(classification_report(y_true, y_pred, target_names=labels_text, digits=3))

torch.save(model.state_dict(), '/content/mymodel/modeloresnet50v2.pth')




'''
Red entrenada desde cero
- 4 bloques convolucionales:
  - Conv2d + ReLU + MaxPool2d
  - Aumento la cantidad de canales: 3 -> 32 -> 64 -> 128.
  - MaxPool2D reducción de la resolución en cada bloque
- Capas (densas):
  - Se aplana el resultado convolucional (128x16x16).
  - 256 neuronas ocultas.
  - 7 neuronas de salida (para las 7 clases de emociones).
'''

'''
primer bloque convierte la imagen de entrada de 128x128x3 a 64x64x64.
segundo bloque reduce a 32x32x128.
tercer bloque tamaño 16x16x256
aplico flatten para transformar el volumen 3D de activaciones en un vector unidimensional de tamaño 256×16×16 = 65536
Dropout al 50% para regularización y evitar overfitting
'''

class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*8*8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
