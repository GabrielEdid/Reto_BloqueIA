# TrOCR Totals - Guía de Entrenamiento

## Cambios Principales Realizados

### ✅ **Problema identificado**: El modelo recibía imágenes completas de recibos

**Solución**: Ahora se extraen y croppean las regiones específicas donde está el total usando bounding boxes del dataset CORD.

### 🎯 Mejoras Implementadas

1. **Extracción de Bounding Boxes**

   - Nueva función `extract_total_info()` que busca el campo total en `valid_line` del JSON
   - Extrae coordenadas del quad y crea un bbox con margen de 10px
   - Fallback a `gt_parse.total.total_price` si no hay bbox disponible

2. **Crop de Imágenes**

   - Se recorta solo la región del total antes de pasarla al modelo
   - TrOCR ahora ve texto de línea única en lugar de documento completo
   - Validación robusta de coordenadas del bbox

3. **Formato de Salida Simplificado**

   - Antes: `"TOTAL 45500 CASH 50000 CHANGE 4500"` (confuso)
   - Ahora: `"45500"` (solo el número limpio)
   - Más fácil para el modelo aprender

4. **Augmentación Mejorada**

   - ColorJitter más agresivo (brightness=0.2, contrast=0.2)
   - GaussianBlur aleatorio (30% probabilidad)
   - Aplicado después del crop para mejor efecto

5. **Hiperparámetros Optimizados**
   - `max_length`: 64 → 32 (suficiente para números)
   - `learning_rate`: 3e-5 → 5e-5 (aprende más rápido)
   - `warmup_steps`: 500 → 300 (warmup más corto)
   - `unfreeze_last_n_layers`: 0 → 2 (encoder más adaptable)
   - `early_stopping patience`: 8 → 5 (detiene antes si no mejora)

## 📊 Comandos Recomendados

### Para Máquina Potente (RTX 4090/A100):

```bash
# Entrenamiento completo con batch grande
python train_totals_trocr.py \
  --epochs 30 \
  --batch 16 \
  --lr 5e-5 \
  --num_workers 8
```

### Para GPU Mediana (RTX 3080/4070):

```bash
python train_totals_trocr.py \
  --epochs 30 \
  --batch 8 \
  --lr 5e-5 \
  --num_workers 4
```

### Para GPU Pequeña (RTX 3060):

```bash
python train_totals_trocr.py \
  --epochs 30 \
  --batch 4 \
  --lr 4e-5 \
  --num_workers 2
```

## 🔍 Evaluación

```bash
# Evaluar el mejor checkpoint
python evaluate_totals_trocr.py \
  --checkpoint trocr_checkpoints/totals/totals-epoch=XX-val_loss=Y.YYY.ckpt

# Ver ejemplos de predicciones
python preview_totals_predictions.py \
  --checkpoint trocr_checkpoints/totals/totals-epoch=XX-val_loss=Y.YYY.ckpt \
  --num 50
```

## 📈 Métricas Esperadas

Con estos cambios, deberías ver:

| Métrica       | Valor Esperado | Explicación                                        |
| ------------- | -------------- | -------------------------------------------------- |
| **MAE**       | < 5,000        | Error absoluto promedio en la predicción del total |
| **Median AE** | < 1,000        | El 50% de predicciones con error menor a esto      |
| **Train Acc** | > 0.90         | Accuracy a nivel de token durante entrenamiento    |
| **Val Acc**   | > 0.85         | Accuracy en validación                             |

### ⚠️ Si los resultados aún son malos:

1. **Verificar que se usan bboxes**: En el output debería decir cuántos samples tienen bbox

   ```
   [CORDTotalsHFDataset] train: 800 válidos de 800 (sin bbox: 50, sin texto: 0)
   ```

2. **Revisar predicciones**: Deben ser números, no palabras:

   ```
   GT total_price: 45500
   Predicción completa: "45500"  ✅
   Predicción completa: "ITEM"   ❌
   ```

3. **Incrementar épocas**: Si val_loss sigue bajando al final, necesita más entrenamiento

4. **Probar descongelar más capas**: Cambiar `unfreeze_last_n_layers` de 2 a 4 en el código

5. **Usar modelo más grande**: Cambiar de `trocr-base-printed` a `trocr-large-printed`

## 🐛 Debugging

### Si el modelo predice palabras random:

- **Causa**: No se están usando los crops correctamente
- **Solución**: Verificar que `extract_total_info()` encuentra bboxes

### Si MAE es gigante (>100M):

- **Causa**: Overflow en conversión de strings a números
- **Solución**: Ya está arreglado con `clean_number()` más robusto

### Si val_loss no baja de ~1.5:

- **Causa**: Modelo no aprende la tarea
- **Solución**:
  - Verificar que los crops son correctos
  - Aumentar batch size si hay memoria
  - Probar learning rate más alto (1e-4)

## 📁 Estructura de Checkpoints

```
trocr_checkpoints/totals/
├── last.ckpt                              # Último checkpoint (para resumir)
├── totals-epoch=04-val_loss=0.123.ckpt   # Mejor modelo
├── totals-epoch=05-val_loss=0.145.ckpt   # Top 2
└── totals-epoch=03-val_loss=0.156.ckpt   # Top 3
```

## 🚀 Tips para Optimizar Velocidad

1. **Usar más workers**: `--num_workers 8` o más si tienes CPU potente
2. **Precision 16-mixed**: Ya está activado automáticamente en GPU
3. **Batch size grande**: Aprovecha toda la VRAM disponible
4. **Pin memory**: Ya está activado (`pin_memory=True`)
5. **Persistent workers**: Ya está activado si `num_workers > 0`

## 🎓 Próximos Pasos

Una vez que este modelo funcione bien:

1. **Extender a otros campos**: Modificar para predecir también `subtotal`, `tax`, `cashprice`, etc.
2. **Multi-task learning**: Un solo modelo que prediga múltiples campos
3. **Ensemble**: Combinar predicciones de múltiples checkpoints
4. **Post-processing**: Validar que el total sea consistente con suma de items
5. **Active learning**: Encontrar ejemplos difíciles y re-entrenar

## 📝 Notas Importantes

- Los cambios son **retrocompatibles**: Si ya tienes checkpoints viejos, funcionarán pero darán malos resultados porque fueron entrenados con imágenes completas
- **Necesitas re-entrenar desde cero** con estos cambios para ver mejoras
- El dataset CORD tiene ~800 imágenes de train, suficiente para fine-tuning
- Si algunos samples no tienen bbox, se usará la imagen completa (no ideal pero mejor que nada)
