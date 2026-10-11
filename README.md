# Clasificador de Prendas con Red Neuronal Convolucional (CNN) y Streamlit

Proyecto de Visión por Computadora y Deep Learning que implementa una Red Neuronal Convolucional (CNN) para clasificar prendas de vestir y calzado a partir del dataset **Fashion-MNIST**.

## Contenido del Proyecto
- `notebooks/fashion_mnist_cnn.ipynb`: Notebook de experimentación en Google Colab con análisis exploratorio, preprocesamiento, data augmentation, entrenamiento con callbacks, métricas de evaluación, matriz de confusión y visualización de predicciones.
- `models/`: Contiene el modelo serializado (`modelo_fashion_cnn.keras`) y el índice de clases (`class_indices.json`).
- `app.py`: Aplicación web interactiva desarrollada con Streamlit para realizar inferencia con nuevas imágenes.
- `requirements.txt`: Lista de dependencias del entorno.

---

## 📌 Requisitos Previos
* **Python 3.10 o 3.11** (Recomendado para compatibilidad con TensorFlow en Windows).
* Sistema operativo: Windows, Linux o macOS.

---

## 🚀 Instrucciones de Ejecución

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DE_TU_REPOSITORIO>
   cd <NOMBRE_DEL_REPOSITORIO>
   ```

2. **Crear y activar un entorno virtual (recomendado):**
    ```bash
    # Windows (PowerShell)
    py -3.11 -m venv env_cnn
    .\env_cnn\Scripts\activate

    # Linux / macOS
    python3.11 -m venv env_cnn
    source env_cnn/bin/activate
    ```

3. **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Ejecutar la aplicación Streamlit:**
    ```bash
    python -m streamlit run app.py
    ```

    La interfaz se abrirá automáticamente en tu navegador en http://localhost:8501.

---

## 🧠 Arquitectura de la Red Convolucional (CNN)

* **Entrada:** Imágenes en escala de grises de $28 \times 28 \times 1$.
* **Data Augmentation:** `RandomFlip("horizontal")`, `RandomRotation(0.08)`, `RandomZoom(0.08)`.
* **Bloques Convolucionales:**
  * **Bloque 1:** Conv2D (32 filtros) + BatchNorm + Conv2D (32) + BatchNorm + MaxPool (2x2) + Dropout (0.25).
  * **Bloque 2:** Conv2D (64 filtros) + BatchNorm + Conv2D (64) + BatchNorm + MaxPool (2x2) + Dropout (0.25).
  * **Bloque 3:** Conv2D (128 filtros) + BatchNorm + MaxPool (2x2) + Dropout (0.30).
* **Clasificador Denso:** Flatten + Densa (128 neuronas, ReLU) + BatchNorm + Dropout (0.40) + Salida Densa (10 clases, Softmax).

---

## 📊 Métricas y Resultados

* **Exactitud en Validación (Val Accuracy):** ~92.34%
* **Exactitud en Prueba (Test Accuracy):** ~92%
* **Optimizador:** Adam ($\text{lr} = 0.001$) con reducción automática ante estancamiento (`ReduceLROnPlateau`).
* **Función de Pérdida:** `sparse_categorical_crossentropy`.
* **Control de Sobreajuste:** `EarlyStopping` y regularización con `Dropout` y `BatchNormalization`.
