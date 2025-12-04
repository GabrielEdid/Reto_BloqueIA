#!/usr/bin/env python3
"""
Servidor Flask para procesar imágenes de tickets con TrOCR.
Recibe imágenes desde la app React Native y las procesa según el método seleccionado.
"""
import os
import argparse
from pathlib import Path
import base64
import io

import torch
import easyocr
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


app = Flask(__name__)
CORS(app)  # Permitir peticiones desde React Native

# Variables globales para el modelo
model = None
processor = None
reader = None
device = None


def find_total_region(image, reader_instance):
    """
    Usa EasyOCR para encontrar el número asociado al "TOTAL".
    Retorna el crop SOLO del número (sin la palabra TOTAL).
    """
    img_array = np.array(image)
    
    # Detectar texto
    results = reader_instance.readtext(img_array)
    
    # Buscar líneas que contengan "total"
    total_boxes = []
    for (bbox, text, conf) in results:
        text_lower = text.lower()
        if "total" in text_lower and "sub" not in text_lower:
            total_boxes.append((bbox, text, conf))
    
    if not total_boxes:
        return image, None, "No se encontró 'TOTAL' en la imagen", None
    
    # Tomar la última ocurrencia
    total_bbox, total_text, total_conf = total_boxes[-1]
    
    # Extraer coordenadas del bbox de "TOTAL"
    xs = [point[0] for point in total_bbox]
    ys = [point[1] for point in total_bbox]
    total_x_min, total_x_max = int(min(xs)), int(max(xs))
    total_y_min, total_y_max = int(min(ys)), int(max(ys))
    total_y_center = (total_y_min + total_y_max) / 2
    
    # Buscar números cercanos
    nearby_numbers = []
    for (bbox, text, conf) in results:
        digit_count = sum(c.isdigit() for c in text)
        if digit_count < len(text) * 0.3:
            continue
        
        bbox_ys = [point[1] for point in bbox]
        bbox_y_center = (min(bbox_ys) + max(bbox_ys)) / 2
        
        y_diff = bbox_y_center - total_y_center
        if -30 <= y_diff <= 60:
            bbox_xs = [point[0] for point in bbox]
            nearby_numbers.append({
                'bbox': bbox,
                'text': text,
                'conf': conf,
                'x_min': int(min(bbox_xs)),
                'x_max': int(max(bbox_xs)),
                'y_min': int(min(bbox_ys)),
                'y_max': int(max(bbox_ys)),
                'y_diff': abs(y_diff),
                'x_distance': int(min(bbox_xs)) - total_x_max
            })
    
    if not nearby_numbers:
        return image, None, f"'{total_text}' sin número visible", None
    
    nearby_numbers.sort(key=lambda x: (x['y_diff'], -x['x_distance']))
    best_number = nearby_numbers[0]
    
    # Crop del número
    x_min = best_number['x_min']
    x_max = best_number['x_max']
    y_min = best_number['y_min']
    y_max = best_number['y_max']
    
    # Añadir margen
    margin = 10
    width, height = image.size
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(width, x_max + margin)
    y_max = min(height, y_max + margin)
    
    crop = image.crop((x_min, y_min, x_max, y_max))
    info = f"'{total_text}' → detectado '{best_number['text']}'"
    
    return crop, (x_min, y_min, x_max, y_max), info, best_number['text']


def process_image(image, method):
    """
    Procesa la imagen según el método seleccionado.
    
    Args:
        image: PIL Image
        method: "easyocr" o "trocr"
    
    Returns:
        dict con los resultados
    """
    global model, processor, reader, device
    
    try:
        ocr_text = None
        info = ""
        
        if method == "easyocr":
            # Modo: EasyOCR detecta, TrOCR lee el crop
            if reader is None:
                return {
                    "success": False,
                    "error": "EasyOCR no está inicializado"
                }
            
            crop, bbox, detection_info, ocr_text = find_total_region(image, reader)
            info = detection_info
        else:
            # Modo: Solo TrOCR (imagen completa)
            crop = image
            info = "Procesando imagen completa con TrOCR"
        
        # Preprocesar para TrOCR
        pixel_values = processor(crop, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        
        # Generar predicción
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return {
            "success": True,
            "trocr_prediction": generated_text,
            "easyocr_detection": ocr_text,
            "info": info,
            "method": method
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar que el servidor está corriendo."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "easyocr_loaded": reader is not None,
        "device": str(device)
    })


@app.route('/process', methods=['POST'])
def process_ticket():
    """
    Endpoint principal para procesar imágenes de tickets.
    
    Espera:
        - image: imagen en base64
        - method: "easyocr" o "trocr"
    
    Retorna:
        JSON con los resultados del procesamiento
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "No se proporcionó la imagen"
            }), 400
        
        method = data.get('method', 'easyocr')
        if method not in ['easyocr', 'trocr']:
            return jsonify({
                "success": False,
                "error": "Método inválido. Use 'easyocr' o 'trocr'"
            }), 400
        
        # Decodificar imagen desde base64
        image_data = data['image']
        
        # Remover prefijo si existe (data:image/jpeg;base64,)
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Procesar imagen
        result = process_image(image, method)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error procesando la imagen: {str(e)}"
        }), 500


def initialize_models(checkpoint_path, use_easyocr=True):
    """
    Inicializa los modelos necesarios.
    """
    global model, processor, reader, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Usando device: {device}")
    
    # Inicializar EasyOCR si es necesario
    if use_easyocr:
        print("📚 Inicializando EasyOCR...")
        reader = easyocr.Reader(['es', 'en'], gpu=torch.cuda.is_available())
        print("✅ EasyOCR listo")
    else:
        print("⏭️  Saltando EasyOCR (solo modo TrOCR)")
    
    # Cargar processor
    print("📚 Cargando TrOCR processor...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    
    # Cargar modelo
    print(f"📚 Cargando modelo desde: {checkpoint_path}")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    model = model.to(device)
    
    # Cargar pesos del checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    
    # Quitar prefijo "model."
    new_state_dict = {}
    prefix = "model."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_k = k[len(prefix):]
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    print("✅ Modelo TrOCR listo")


def main():
    parser = argparse.ArgumentParser(description="Servidor para procesamiento de tickets")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="trocr_checkpoints/totals/totals-epoch=02-val_loss=0.599.ckpt",
        help="Ruta al checkpoint del modelo TrOCR"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Puerto para el servidor"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host para el servidor"
    )
    parser.add_argument(
        "--no-easyocr",
        action="store_true",
        help="No cargar EasyOCR (solo modo TrOCR disponible)"
    )
    args = parser.parse_args()
    
    # Verificar que existe el checkpoint
    if not Path(args.checkpoint).exists():
        print(f"❌ ERROR: No existe el checkpoint: {args.checkpoint}")
        return
    
    print("=" * 80)
    print("🚀 Iniciando servidor de procesamiento de tickets")
    print("=" * 80)
    
    # Inicializar modelos
    initialize_models(args.checkpoint, use_easyocr=not args.no_easyocr)
    
    print("\n" + "=" * 80)
    print(f"🌐 Servidor corriendo en http://{args.host}:{args.port}")
    print(f"📱 Endpoints disponibles:")
    print(f"   - GET  /health  - Verificar estado del servidor")
    print(f"   - POST /process - Procesar imagen de ticket")
    print("=" * 80 + "\n")
    
    # Iniciar servidor
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
