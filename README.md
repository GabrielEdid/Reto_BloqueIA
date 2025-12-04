# Reto_BloqueIA

Sistema completo de reconocimiento óptico de caracteres (OCR) y extracción de información de tickets/recibos utilizando modelos de Deep Learning (TrOCR y Donut) con PyTorch Lightning.

## 📋 Descripción General

Este proyecto implementa una solución end-to-end para la digitalización y extracción automática de información de tickets y recibos, incluyendo:

- **Modelos de OCR**: TrOCR y Donut para reconocimiento de texto
- **Extracción de totales**: Modelo especializado para detectar montos totales
- **API Server**: Servidor Flask para inferencia en tiempo real
- **App Móvil**: Aplicación React Native para captura de tickets
- **Pipeline completo**: Desde entrenamiento hasta deployment

## 🎯 Características Principales

- ✅ OCR de alta precisión con TrOCR y Donut
- ✅ Detección especializada de montos totales
- ✅ Integración con EasyOCR como fallback
- ✅ API REST para inferencia
- ✅ Aplicación móvil para captura de tickets
- ✅ Scripts de entrenamiento configurables
- ✅ Evaluación y testing exhaustivo

## 🗂️ Estructura del Proyecto

```
Reto_BloqueIA/
├── training/                    # Scripts y modelos de entrenamiento
│   ├── donut/                  # Entrenamiento del modelo Donut
│   │   ├── train_donut.py
│   │   ├── run_donut_a4500.sh
│   │   └── donut_logs/
│   ├── trocr/                  # Entrenamiento del modelo TrOCR
│   │   ├── train_trocr.py
│   │   ├── new_train_trocr.py
│   │   ├── run_trocr_rtx4070.sh
│   │   └── trocr_logs/
│   └── totals/                 # Modelo especializado en totales
│       ├── train_totals_trocr.py
│       ├── totals-epoch=02-val_loss=0.599.ckpt
│       └── trocr_logs/
│
├── evaluation/                  # Scripts de evaluación
│   ├── evaluate_donut.py
│   ├── evaluate_trocr.py
│   ├── evaluate_totals_trocr.py
│   ├── inspect_trocr_predictions.py
│   ├── preview_totals_predictions.py
│   ├── evaluation_results_frozen.txt
│   ├── evaluation_results_full.txt
│   └── evaluation_results_partial.txt
│
├── testing/                     # Scripts de testing y validación
│   ├── test_custom_images.py
│   ├── test_custom_with_ocr.py
│   ├── test_extraction.py
│   ├── visualize_crops.py
│   ├── check_crops/
│   ├── debug_crops/
│   └── test_crops/
│
├── server/                      # Servidor de inferencia
│   ├── server.py               # API Flask
│   ├── requirements_server.txt
│   └── tickets_de_prueba/      # Tickets para testing
│
├── TicketRecognition/          # App móvil React Native
│   ├── App.js
│   ├── app.json
│   ├── babel.config.js
│   ├── package.json
│   └── assets/
│
├── data/                        # Datos de desarrollo
│   ├── test_images/
│   ├── test_crops/
│   ├── debug_crops/
│   └── check_crops/
│
├── data_set/                    # Dataset principal CORD-v2
│   ├── images/
│   ├── boxes/
│   ├── annotations.xml
│   ├── receipts.csv
│   └── split/
│       ├── train/
│       └── test/
│
├── models/                      # Modelos entrenados
│   └── trocr/
│
├── doctr/                       # Integración DocTR
│
├── Notebooks/                   # Notebooks Jupyter
│   ├── Reto.ipynb
│   ├── Dataset_Study.ipynb
│   ├── Donut_Training.ipynb
│   ├── TrOCR_Training.ipynb
│   ├── 06_CV_Histograms.ipynb
│   └── Project_Traffic_sign_classifier.ipynb
│
├── requirements.txt             # Dependencias Python
├── setup.sh                     # Script de instalación
└── README.md                    # Este archivo
```

## 🚀 Instalación y Configuración

### Requisitos Previos

- Python 3.8 o superior
- CUDA 11.x o superior (para entrenamiento con GPU)
- Node.js 14+ (para la app móvil)

### Setup Rápido

```bash
# Clonar el repositorio
git clone https://github.com/GabrielEdid/Reto_BloqueIA.git
cd Reto_BloqueIA

# Ejecutar script de instalación
chmod +x setup.sh
./setup.sh
```

El script `setup.sh` automáticamente:

1. Crea un entorno virtual en `env/`
2. Actualiza pip
3. Instala todas las dependencias de `requirements.txt`

### Setup Manual

```bash
# Crear entorno virtual
python3 -m venv env

# Activar entorno virtual
source env/bin/activate  # Linux/Mac
# o
env\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## 🎓 Entrenamiento de Modelos

### TrOCR (Transformer-based OCR)

```bash
# Activar entorno
source env/bin/activate

# Entrenamiento básico
cd training/trocr
python train_trocr.py

# Entrenamiento con script optimizado
python new_train_trocr.py

# Ejecución en RTX 4070
bash run_trocr_rtx4070.sh
```

### Donut (Document Understanding Transformer)

```bash
cd training/donut
python train_donut.py

# Ejecución en A4500
bash run_donut_a4500.sh
```

### Modelo de Totales

```bash
cd training/totals
python train_totals_trocr.py
```

**Modelo pre-entrenado disponible**: `training/totals/totals-epoch=02-val_loss=0.599.ckpt`

## 🧪 Evaluación

### Evaluar TrOCR

```bash
cd evaluation
python evaluate_trocr.py
```

### Evaluar Donut

```bash
python evaluate_donut.py
```

### Evaluar Modelo de Totales

```bash
python evaluate_totals_trocr.py
```

### Ver Predicciones

```bash
# Previsualizar predicciones de totales
python preview_totals_predictions.py

# Inspeccionar predicciones de TrOCR
python inspect_trocr_predictions.py
```

## 🧩 Testing

### Test con Imágenes Personalizadas

```bash
cd testing

# Test básico
python test_custom_images.py

# Test con OCR combinado (TrOCR + EasyOCR)
python test_custom_with_ocr.py

# Test de extracción
python test_extraction.py

# Visualizar crops
python visualize_crops.py
```

## 🌐 Servidor de Inferencia

### Iniciar Servidor

```bash
cd server

# Iniciar servidor con configuración por defecto (puerto 8000)
python server.py

# Especificar checkpoint personalizado
python server.py --checkpoint ../training/totals/totals-epoch=02-val_loss=0.599.ckpt

# Cambiar puerto
python server.py --port 5000

# Guardar crops de las imágenes procesadas
python server.py --save-crops server_crops

# Ejecutar sin EasyOCR (solo TrOCR disponible)
python server.py --no-easyocr

# Configuración completa personalizada
python server.py \
    --checkpoint ../training/totals_trocr/totals-epoch=02-val_loss=0.599.ckpt \
    --save-crops server_crops \
    --port 8000 \
    --host 0.0.0.0
```

**Opciones disponibles:**

- `--checkpoint`: Ruta al checkpoint del modelo TrOCR (default: `../training/totals_trocr/totals-epoch=02-val_loss=0.599.ckpt`)
- `--port`: Puerto del servidor (default: `8000`)
- `--host`: Host del servidor (default: `0.0.0.0`)
- `--save-crops`: Directorio para guardar crops de imágenes procesadas (opcional)
- `--no-easyocr`: Desactivar EasyOCR, solo usar TrOCR

El servidor estará disponible en `http://localhost:8000`

### API Endpoints

```bash
# Health check
GET /health

# Procesar ticket - Método 1: Solo TrOCR (más rápido)
POST /process
Content-Type: multipart/form-data
Body:
  - image: archivo de imagen
  - method: "trocr" (opcional, default)

# Procesar ticket - Método 2: TrOCR + EasyOCR (más preciso)
POST /process
Content-Type: multipart/form-data
Body:
  - image: archivo de imagen
  - method: "trocr+easyocr"

# Ejemplos con curl
# Método básico (TrOCR solo)
curl -X POST -F "image=@ticket.jpg" http://localhost:8000/process

# Método combinado (TrOCR + EasyOCR)
curl -X POST \
  -F "image=@ticket.jpg" \
  -F "method=trocr+easyocr" \
  http://localhost:8000/process
```

**Respuesta del servidor:**

```json
{
  "success": true,
  "total": "1234.56",
  "confidence": 0.95,
  "method": "trocr",
  "processing_time": 0.234
}
```

## 📱 Aplicación Móvil

### Setup de la App

```bash
cd TicketRecognition

# Instalar dependencias
npm install

# Iniciar Expo
npm start
```

### Ejecutar en Dispositivo

- **iOS**: Escanea el QR con la app Expo Go o presiona `i` para simulador
- **Android**: Escanea el QR con la app Expo Go o presiona `a` para simulador

## 📊 Dataset

### CORD-v2 (Consolidated Receipt Dataset v2)

- **Fuente**: Naver Clova IX
- **Ubicación**: `data_set/`
- **Contenido**:
  - 1,000+ imágenes de recibos
  - Anotaciones OCR
  - Bounding boxes
  - Información estructurada clave-valor

### Estructura del Dataset

```
data_set/
├── images/          # Imágenes de recibos
├── boxes/           # Bounding boxes
├── split/
│   ├── train/      # Set de entrenamiento
│   └── test/       # Set de prueba
├── annotations.xml  # Anotaciones
└── receipts.csv    # Metadata
```

## 🔧 Dependencias Principales

### Machine Learning

- **PyTorch** 2.0+ - Framework de Deep Learning
- **PyTorch Lightning** - Wrapper de alto nivel
- **Transformers (HuggingFace)** - Modelos pre-entrenados
- **torchvision** - Utilidades de visión computacional

### OCR y Procesamiento

- **EasyOCR** - OCR de fallback
- **Pillow** - Procesamiento de imágenes
- **OpenCV** - Visión computacional

### Data Science

- **NumPy** - Computación numérica
- **Pandas** - Manipulación de datos
- **Matplotlib** - Visualización

### Server

- **Flask** - API REST
- **Flask-CORS** - Manejo de CORS

### Mobile

- **React Native** - Framework móvil
- **Expo** - Toolchain para React Native

## 📈 Resultados

Los resultados de evaluación están disponibles en `evaluation/`:

- `evaluation_results_frozen.txt` - Modelo con capas congeladas
- `evaluation_results_full.txt` - Modelo completamente entrenado
- `evaluation_results_partial.txt` - Entrenamiento parcial

## 🎯 Modelos Disponibles

### TrOCR

- **Estrategias**: frozen, full, partial
- **Logs**: `training/trocr/trocr_logs/`
- **Checkpoints**: `models/trocr/`

### Donut

- **Estrategias**: frozen, partial
- **Logs**: `training/donut/donut_logs/`

### Totals

- **Modelo optimizado** para extracción de totales
- **Checkpoint**: `training/totals/totals-epoch=02-val_loss=0.599.ckpt`

## 🛠️ Comandos Útiles

```bash
# Activar entorno
source env/bin/activate

# Desactivar entorno
deactivate

# Ver logs de entrenamiento
tensorboard --logdir training/trocr/trocr_logs

# Limpiar cache de Python
find . -type d -name __pycache__ -exec rm -r {} +

# Actualizar dependencias
pip install -r requirements.txt --upgrade
```

## 📝 Notas de Desarrollo

- Los notebooks en la raíz son para exploración y desarrollo
- Los logs de entrenamiento se guardan automáticamente
- El servidor usa el mejor checkpoint disponible
- La app móvil se conecta al servidor local por defecto

## 🤝 Contribuciones

Este proyecto fue desarrollado como parte del Reto de Bloque de IA.

**Equipo**:

- Gabriel Edid
- Paul Araque

## 📄 Licencia

Ver archivo LICENSE para detalles.

## 🔗 Enlaces Útiles

- [CORD-v2 Dataset](https://huggingface.co/datasets/naver-clova-ix/cord-v2)
- [TrOCR Documentation](https://huggingface.co/docs/transformers/model_doc/trocr)
- [Donut Documentation](https://huggingface.co/docs/transformers/model_doc/donut)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)

---

**Última actualización**: Diciembre 2025
