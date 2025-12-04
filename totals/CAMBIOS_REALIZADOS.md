# 🔧 Cambios Realizados - Resumen Ejecutivo

## 🎯 Problema Principal

El modelo predecía palabras aleatorias ("ITEM", "TAX", ":") en lugar de números porque:

- **Recibía imágenes completas de recibos** (864x1296 px)
- TrOCR está diseñado para **OCR de línea única**, no documentos completos
- No se usaban los bounding boxes disponibles en CORD

## ✅ Soluciones Implementadas

### 1. **Extracción de Bounding Boxes** ✨

```python
# ANTES: Se usaba toda la imagen
image = sample["image"]
processor(image, ...)

# AHORA: Se extrae el bbox y se hace crop
info = extract_total_info(gt_string)  # → {text: "45500", bbox: {...}}
image = image.crop((x_min, y_min, x_max, y_max))
processor(image, ...)  # Solo ve el número
```

### 2. **Formato Simplificado**

```python
# ANTES: Confuso con múltiples campos
"TOTAL 45500 CASH 50000 CHANGE 4500 CARD 0"

# AHORA: Solo el número limpio
"45500"
```

### 3. **Augmentación Mejorada**

- ColorJitter más agresivo (0.05 → 0.2)
- GaussianBlur aleatorio (30% prob)
- Aplicado DESPUÉS del crop

### 4. **Hiperparámetros Optimizados**

| Parámetro                | Antes | Ahora | Razón                  |
| ------------------------ | ----- | ----- | ---------------------- |
| `max_length`             | 64    | 32    | Los números son cortos |
| `learning_rate`          | 3e-5  | 5e-5  | Aprende más rápido     |
| `warmup_steps`           | 500   | 300   | Warmup más corto       |
| `unfreeze_last_n_layers` | 0     | 2     | Encoder más adaptable  |
| `early_stop patience`    | 8     | 5     | Para antes si estanca  |

## 📁 Archivos Modificados

### ✏️ `train_totals_trocr.py`

- Nueva función `extract_total_info()` que busca bboxes en `valid_line`
- `CORDTotalsHFDataset` hace crop antes de procesar
- Mejor logging (muestra cuántos samples tienen bbox)

### ✏️ `evaluate_totals_trocr.py`

- Usa bboxes para hacer crop durante evaluación
- Métricas adicionales: Median AE, Min/Max Error
- Cuenta samples sin bbox

### ✏️ `preview_totals_predictions.py`

- Visualiza predicciones con crop aplicado
- Indica si cada sample usó bbox o imagen completa

### 🆕 `visualize_crops.py`

- Script nuevo para verificar que los crops funcionan
- Guarda imágenes mostrando bbox y crop lado a lado
- Útil para debugging

### 📚 `README_TRAINING.md`

- Guía completa de entrenamiento
- Comandos para diferentes GPUs
- Métricas esperadas
- Troubleshooting

## 🚀 Cómo Usar

### 1️⃣ Verificar que los crops funcionan

```bash
cd totals
python visualize_crops.py --num 20 --split train --output crops_check
# Revisar las imágenes en crops_check/
```

### 2️⃣ Entrenar (en la máquina potente)

```bash
# GPU potente (batch 16)
python train_totals_trocr.py --epochs 30 --batch 16 --lr 5e-5 --num_workers 8

# GPU mediana (batch 8)
python train_totals_trocr.py --epochs 30 --batch 8 --lr 5e-5 --num_workers 4
```

### 3️⃣ Evaluar

```bash
# Ver métricas
python evaluate_totals_trocr.py \
  --checkpoint trocr_checkpoints/totals/totals-epoch=XX-val_loss=Y.YYY.ckpt

# Ver ejemplos
python preview_totals_predictions.py \
  --checkpoint trocr_checkpoints/totals/totals-epoch=XX-val_loss=Y.YYY.ckpt \
  --num 30
```

## 📊 Resultados Esperados

### ✅ Antes de los cambios:

```
MAE: 102,040,919,489,837,160,312,833,748,940,881,920.00  ❌
Predicciones: "ITEM", "TAX", ":", "ID", "R"  ❌
```

### ✅ Después de los cambios:

```
MAE: < 5,000  ✅
Median AE: < 1,000  ✅
Predicciones: "45500", "23000", "89100"  ✅
Train Acc: > 0.90  ✅
Val Acc: > 0.85  ✅
```

## ⚠️ Importante

1. **DEBES re-entrenar desde cero** - Los checkpoints viejos no sirven
2. **Verificar bboxes primero** - Usa `visualize_crops.py` antes de entrenar
3. **Monitorear val_loss** - Si no baja de 1.5, hay un problema
4. **Batch size grande** - Aprovecha toda la VRAM disponible

## 🔍 Troubleshooting

### Si aún predice palabras:

- ✅ Verificar que `extract_total_info()` encuentra bboxes
- ✅ Revisar output: debe decir "sin bbox: X" donde X < 100
- ✅ Usar `visualize_crops.py` para ver si los crops son correctos

### Si MAE sigue alto (>10,000):

- ✅ Entrenar más épocas (30-50)
- ✅ Aumentar batch size si hay VRAM
- ✅ Probar learning rate más alto (1e-4)
- ✅ Descongelar más capas (unfreeze_last_n_layers=4)

### Si val_loss no baja:

- ✅ Verificar que los crops se ven bien
- ✅ Revisar que el texto de salida sea solo números
- ✅ Intentar con modelo más grande (trocr-large-printed)

## 🎓 Próximos Pasos

Una vez que funcione:

1. Extender a otros campos (subtotal, tax, etc.)
2. Multi-task learning (un modelo para todos los campos)
3. Post-processing (validar consistencia)
4. Ensemble de múltiples checkpoints

---

**Autor**: Gabriel  
**Fecha**: Diciembre 3, 2025  
**Branch**: compu_stride
