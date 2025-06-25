# 🧠 Detección de Emociones Faciales con Deep Learning

Este proyecto utiliza modelos de deep learning preentrenados (ResNet18, ResNet50) para reconocer emociones humanas a partir de imágenes faciales. Se entrenó sobre la base de datos RAF-DB y permite detectar emociones en tiempo real a través de la cámara web utilizando Streamlit.

---

## 📁 Estructura del Proyecto

```
emotion-detector/
│
├── app.py                      # Aplicación principal con Streamlit
├── model/                      # Modelos entrenados y definición de arquitectura
├── data/                       # Scripts para preparar RAF-DB
├── prod/                      # Funciones auxiliares: procesamiento de rostros, métricas
├── requirements.txt            # Lista de dependencias
├── README.md                   # Este archivo
└── ...
```

---

## 🚀 Cómo Ejecutar el Proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/tuusuario/emotion-detector.git
cd emotion-detector
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar la aplicación web con Streamlit

```bash
streamlit run prod/app.py
```

---

## 📦 Requisitos

El proyecto utiliza:

- Python ≥ 3.8  
- PyTorch  
- torchvision  
- OpenCV  
- facenet-pytorch  
- PIL  
- Streamlit  
- scikit-learn  
- matplotlib  

Ver todos en `requirements.txt`.

---

## 💡 Funcionalidades

- Detección de rostro en vivo
- Clasificación de emociones (enojado, feliz, triste, neutral, etc.)
- Métricas detalladas y visualización
