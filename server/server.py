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
        # Estrategia de respaldo: buscar TODOS los números en la imagen
        all_number_candidates = []
        for (bbox, text, conf) in results:
            # Más permisivo: cualquier texto con dígitos
            if any(c.isdigit() for c in text):
                bbox_xs = [point[0] for point in bbox]
                bbox_ys = [point[1] for point in bbox]
                bbox_y_center = (min(bbox_ys) + max(bbox_ys)) / 2
                y_diff = abs(bbox_y_center - total_y_center)
                
                all_number_candidates.append({
                    'x_min': int(min(bbox_xs)),
                    'x_max': int(max(bbox_xs)),
                    'y_min': int(min(bbox_ys)),
                    'y_max': int(max(bbox_ys)),
                    'y_diff': y_diff,
                    'text': text
                })
        
        if not all_number_candidates:
            # Último recurso: crop alrededor de "TOTAL" expandiendo mucho a la derecha
            width, height = image.size
            x_min = total_x_min
            x_max = min(width, total_x_max + 200)  # 200px a la derecha
            y_min = max(0, total_y_min - 20)
            y_max = min(height, total_y_max + 20)
            crop = image.crop((x_min, y_min, x_max, y_max))
            
            # Mejorar imagen
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(crop)
            crop = enhancer.enhance(2.0)  # Contraste más agresivo
            enhancer = ImageEnhance.Sharpness(crop)
            crop = enhancer.enhance(2.5)
            
            return crop, (x_min, y_min, x_max, y_max), f"'{total_text}' - respaldo: expansión derecha", None
        
        # Usar el número más cercano verticalmente
        all_number_candidates.sort(key=lambda x: x['y_diff'])
        nearby_numbers = all_number_candidates[:3]  # Top 3 más cercanos
    
    # Combinar TODOS los números en la misma línea horizontal
    # (no solo el primero, sino todos los fragmentos: "5", "180", ".", "00", etc.)
    same_line_numbers = [n for n in nearby_numbers if n['y_diff'] < 20]
    
    if not same_line_numbers:
        same_line_numbers = nearby_numbers[:1]  # Al menos uno
    
    # Calcular bbox que englobe TODOS los números detectados en esa línea
    x_min = min(n['x_min'] for n in same_line_numbers)
    x_max = max(n['x_max'] for n in same_line_numbers)
    y_min = min(n['y_min'] for n in same_line_numbers)
    y_max = max(n['y_max'] for n in same_line_numbers)
    
    # Información de todos los fragmentos detectados
    all_texts = " ".join([n['text'] for n in same_line_numbers])
    
    # Expandir horizontalmente para capturar números que puedan estar cortados
    # EasyOCR a veces no detecta todos los dígitos
    width, height = image.size
    bbox_width = x_max - x_min
    horizontal_expansion = int(bbox_width * 0.3)  # Expandir 30% a cada lado
    
    # Añadir margen generoso
    margin_x = max(15, horizontal_expansion)
    margin_y = 15
    x_min = max(0, x_min - margin_x)
    y_min = max(0, y_min - margin_y)
    x_max = min(width, x_max + margin_x)
    y_max = min(height, y_max + margin_y)
    
    # Hacer crop de toda la línea del número
    crop = image.crop((x_min, y_min, x_max, y_max))
    
    # Aplicar mejoras de imagen para mejorar legibilidad
    # 1. Aumentar contraste
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(crop)
    crop = enhancer.enhance(1.5)  # Aumentar contraste 50%
    
    # 2. Aumentar nitidez
    enhancer = ImageEnhance.Sharpness(crop)
    crop = enhancer.enhance(2.0)  # Duplicar nitidez
    
    info = f"'{total_text}' → detectado '{all_texts}'"
    
    return crop, (x_min, y_min, x_max, y_max), info, all_texts


def process_image(image, method, save_crops_dir=None):
    """
    Procesa la imagen según el método seleccionado.
    
    Args:
        image: PIL Image
        method: "easyocr" o "trocr"
        save_crops_dir: Directorio para guardar crops (opcional)
    
    Returns:
        dict con los resultados
    """
    global model, processor, reader, device
    
    from datetime import datetime
    
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
        
        # Guardar crop si está habilitado
        crop_filename = None
        if save_crops_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            crop_filename = f"crop_{method}_{timestamp}.jpg"
            crop_path = Path(save_crops_dir) / crop_filename
            crop.save(crop_path, quality=95)
        
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
            "method": method,
            "crop_saved": crop_filename
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
        
        # Procesar imagen (con save_crops_dir si está configurado globalmente)
        save_dir = getattr(app, 'save_crops_dir', None)
        result = process_image(image, method, save_crops_dir=save_dir)
        
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
    parser.add_argument(
        "--save-crops",
        type=str,
        default=None,
        help="Directorio donde guardar los crops (ej: server_crops)"
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
    
    # Configurar directorio de crops si está habilitado
    if args.save_crops:
        crops_path = Path(args.save_crops)
        crops_path.mkdir(parents=True, exist_ok=True)
        app.save_crops_dir = str(crops_path)
        print(f"💾 Guardando crops en: {crops_path}")
    else:
        app.save_crops_dir = None
    
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
