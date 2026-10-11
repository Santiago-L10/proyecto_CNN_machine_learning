import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import json
import os

st.set_page_config(
    page_title="Clasificador Fashion-MNIST - CNN",
    page_icon="👕",
    layout="centered"
)

# 1. Carga en caché del modelo y clases
@st.cache_resource
def load_model_and_classes():
    model_path = os.path.join("models", "modelo_fashion_cnn.keras")
    classes_path = os.path.join("models", "class_indices.json")
    
    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        st.error("No se encontraron los artefactos en la carpeta 'models/'. Verifica que existan 'modelo_fashion_cnn.keras' y 'class_indices.json'.")
        st.stop()
        
    model = tf.keras.models.load_model(model_path)
    with open(classes_path, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
        
    classes = [class_indices[str(i)] for i in range(len(class_indices))]
    return model, classes

model, class_names = load_model_and_classes()

# Traducción de nombres para la interfaz de usuario
traducciones = {
    "T-shirt/top": "Camiseta / Top",
    "Trouser": "Pantalón",
    "Pullover": "Suéter / Jersey",
    "Dress": "Vestido",
    "Coat": "Abrigo",
    "Sandal": "Sandalia",
    "Shirt": "Camisa",
    "Sneaker": "Zapatilla deportiva",
    "Bag": "Bolso / Cartera",
    "Ankle boot": "Botín"
}

# 2. Función de preprocesamiento de imagen
def preprocess_image(image: Image.Image):
    # Convertir a escala de grises
    img_gray = image.convert("L")
    
    # Redimensionar a 28x28 píxeles
    img_resized = img_gray.resize((28, 28))
    
    # Convertir a arreglo numpy y normalizar a [0, 1]
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Invertir colores si el fondo es blanco (Fashion-MNIST tiene fondo negro y prenda blanca)
    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array
        
    # Dar forma esperada por la CNN: (1, 28, 28, 1)
    img_tensor = img_array.reshape(1, 28, 28, 1)
    return img_tensor, img_resized

# 3. Interfaz de Usuario
st.title("👕 Clasificador de Prendas con CNN")
st.markdown("""
Esta aplicación utiliza una **Red Neuronal Convolucional (CNN)** con arquitectura multicapa y 
regularización, entrenada sobre el dataset **Fashion-MNIST** (10 clases).
""")

st.write("---")

uploaded_file = st.file_uploader(
    "Sube una imagen de una prenda o calzado:", 
    type=["png", "jpg", "jpeg", "webp", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col_img, col_pred = st.columns([1, 1])
    
    with col_img:
        st.write("### Imagen Original")
        st.image(image, use_column_width=True)
        
    tensor, processed_preview = preprocess_image(image)
    
    with st.spinner("Procesando imagen con la Red Convolucional..."):
        predictions = model.predict(tensor, verbose=0)[0]
        pred_idx = int(np.argmax(predictions))
        confidence = float(predictions[pred_idx]) * 100
        predicted_class_en = class_names[pred_idx]
        predicted_class_es = traducciones.get(predicted_class_en, predicted_class_en)

    with col_pred:
        st.write("### Resultado")
        st.success(f"**Clase Predicha:**\n### {predicted_class_es}")
        st.caption(f"Etiqueta original: `{predicted_class_en}`")
        st.metric(label="Nivel de Confianza", value=f"{confidence:.2f}%")
        st.progress(confidence / 100.0)

    st.write("---")
    st.write("### Distribución de Probabilidades por Clase")
    
    # Gráfico de barras de probabilidades
    chart_data = {traducciones.get(c, c): float(p) for c, p in zip(class_names, predictions)}
    st.bar_chart(chart_data)

st.write("---")
st.markdown("""
**Detalles del Modelo:**
* **Arquitectura:** Conv2D (32) $\\rightarrow$ Conv2D (32) $\\rightarrow$ MaxPool $\\rightarrow$ Conv2D (64) $\\rightarrow$ Conv2D (64) $\\rightarrow$ MaxPool $\\rightarrow$ Conv2D (128) $\\rightarrow$ MaxPool $\\rightarrow$ Densa (128) $\\rightarrow$ Softmax (10).
* **Técnicas aplicadas:** Data Augmentation (Flip horizontal, rotación y zoom leve), Batch Normalization, Dropout y Callbacks adaptativos.
""")