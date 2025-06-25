class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        all_images = glob(os.path.join(img_dir, "*", "*.jpg"))
        self.img_paths = {os.path.basename(p): p for p in all_images}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = self.img_paths.get(img_name)

        if img_path is None:
            raise FileNotFoundError(f"Imagen {img_name} no encontrada en {self.img_dir}")

        image = Image.open(img_path).convert("RGB")  # Convertimos a RGB directamente
        label = self.img_labels.iloc[idx, 1] - 1 # Ajusta el label restando 1 (para que los labels estén entre 0 y 6).

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

])

training_data = CustomImageDataset(
    annotations_file = '/kaggle/input/raf-db-dataset/train_labels.csv',
    img_dir='/kaggle/input/raf-db-dataset/DATASET/train',
    transform=transform_train
)


test_data = CustomImageDataset(
    annotations_file = '/kaggle/input/raf-db-dataset/test_labels.csv',
    img_dir='/kaggle/input/raf-db-dataset/DATASET/test',
    transform=transform_test
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# Obtener un batch del dataloader
train_features, train_labels = next(iter(train_dataloader))

# Crear figura
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle("Muestras del dataset", fontsize=12)

# Recorrer 16 imágenes
for i in range(16):
    img = train_features[i].permute(1, 2, 0)  # (C, H, W) → (H, W, C)
    label = train_labels[i]

    ax = axes[i // 4, i % 4]  # Subgrilla
    ax.imshow(img)
    ax.set_title(f"Label: {label}")
    ax.axis("off")

plt.show()