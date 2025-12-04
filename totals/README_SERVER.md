# Servidor de Procesamiento de Tickets

Este servidor Flask procesa imágenes de tickets usando TrOCR con dos métodos disponibles:

- **EasyOCR + TrOCR**: Detecta automáticamente la región del total con EasyOCR y luego lee el número con TrOCR
- **Solo TrOCR**: Procesa la imagen completa directamente con TrOCR

## Instalación

1. Instalar dependencias:

```bash
pip install -r requirements_server.txt
```

## Uso

### Iniciar el servidor

**⚠️ Nota importante sobre el puerto:**
En macOS, el puerto 5000 está ocupado por AirPlay Receiver. **Usa el puerto 8000** en su lugar.

**Opción 1: Con EasyOCR y TrOCR (recomendado)**

```bash
python server.py --checkpoint totals-epoch=02-val_loss=0.599.ckpt --port 8000
```

**Opción 2: Solo TrOCR (más rápido, menos memoria)**

```bash
python server.py --checkpoint totals-epoch=02-val_loss=0.599.ckpt --port 8000 --no-easyocr
```

### Parámetros disponibles

- `--checkpoint`: Ruta al checkpoint del modelo TrOCR (requerido)
- `--port`: Puerto del servidor (default: 5000, recomendado: 8000 en macOS)
- `--host`: Host del servidor (default: 0.0.0.0)
- `--no-easyocr`: No cargar EasyOCR (solo modo TrOCR disponible)

## Endpoints

### GET /health

Verifica el estado del servidor.

**Respuesta:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "easyocr_loaded": true,
  "device": "cuda"
}
```

### POST /process

Procesa una imagen de ticket.

**Request:**

```json
{
  "image": "data:image/jpeg;base64,...",
  "method": "easyocr" // o "trocr"
}
```

**Respuesta exitosa:**

```json
{
  "success": true,
  "trocr_prediction": "123.45",
  "easyocr_detection": "123.45",
  "info": "'TOTAL' → detectado '123.45'",
  "method": "easyocr"
}
```

**Respuesta con error:**

```json
{
  "success": false,
  "error": "Descripción del error"
}
```

## Configuración para React Native

1. **Encontrar tu IP local:**

2. **Actualizar App.js:**
   Cambiar la línea:

   ```javascript
   const SERVER_URL = "http://192.168.1.100:8000"; // Usa puerto 8000
   ```

   Por tu IP local y asegúrate de usar el mismo puerto que el servidor. = "http://192.168.1.100:5000";

   ```

   Por tu IP local.

   ```

3. **Asegurar que el servidor y el dispositivo/emulador estén en la misma red.**

## Ejemplo de prueba con curl

```bash
# Codificar imagen a base64
base64_image=$(base64 -i test_ticket.jpg)

# Enviar petición
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$base64_image\", \"method\": \"easyocr\"}"
```

## Notas

- El servidor carga los modelos al iniciar, lo cual puede tardar unos minutos
- Se requiere GPU para mejor rendimiento (funciona en CPU pero es más lento)
- Las imágenes se procesan en base64 para compatibilidad con React Native
- EasyOCR detecta texto en español e inglés
