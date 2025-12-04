#!/usr/bin/env python3
"""
Script para probar el modelo TrOCR con detección automática de la región del total.
Usa EasyOCR para detectar texto y encontrar la línea del "TOTAL".
"""
import argparse
import re
from pathlib import Path

import torch
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def find_total_region(image_path, reader):
    """
    Usa EasyOCR para encontrar el número asociado al "TOTAL".
    Retorna el crop SOLO del número (sin la palabra TOTAL).
    """
    # Leer imagen
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)
    
    # Detectar texto
    results = reader.readtext(img_array)
    
    # Buscar líneas que contengan "total"
    total_boxes = []
    for (bbox, text, conf) in results:
        text_lower = text.lower()
        if "total" in text_lower and "sub" not in text_lower:
            total_boxes.append((bbox, text, conf))
    
    if not total_boxes:
        # No se encontró "total", usar imagen completa
        return image, None, "No se encontró 'TOTAL' en la imagen", None
    
    # Tomar la última ocurrencia (generalmente el total final está al final)
    total_bbox, total_text, total_conf = total_boxes[-1]
    
    # Extraer coordenadas del bbox de "TOTAL"
    xs = [point[0] for point in total_bbox]
    ys = [point[1] for point in total_bbox]
    total_x_min, total_x_max = int(min(xs)), int(max(xs))
    total_y_min, total_y_max = int(min(ys)), int(max(ys))
    total_y_center = (total_y_min + total_y_max) / 2
    
    # Buscar números cercanos (misma línea horizontal o justo debajo)
    nearby_numbers = []
    for (bbox, text, conf) in results:
        # Verificar si contiene dígitos (al menos 30% del texto)
        digit_count = sum(c.isdigit() for c in text)
        if digit_count < len(text) * 0.3:
            continue
        
        # Calcular centro vertical del bbox
        bbox_ys = [point[1] for point in bbox]
        bbox_y_center = (min(bbox_ys) + max(bbox_ys)) / 2
        
        # Verificar si está en la misma línea (±30px) o justo debajo (hasta +60px)
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
                'x_distance': int(min(bbox_xs)) - total_x_max  # Distancia a la derecha
            })
    
    # Si no hay números cercanos, usar imagen completa
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Ruta al checkpoint del modelo (.ckpt)"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="tickets_de_prueba",
        help="Directorio con las imágenes a probar"
    )
    parser.add_argument(
        "--save_crops",
        action="store_true",
        help="Guardar los crops generados"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="custom_crops",
        help="Directorio para guardar crops"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["trocr_only", "easyocr+trocr"],
        default="easyocr+trocr",
        help="Modo: 'trocr_only' (imagen completa a TrOCR) o 'easyocr+trocr' (EasyOCR detecta, TrOCR lee)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")
    print(f"Modo: {args.mode}")

    # Inicializar EasyOCR solo si es necesario
    reader = None
    if args.mode == "easyocr+trocr":
        print("Inicializando EasyOCR...")
        reader = easyocr.Reader(['es', 'en'], gpu=torch.cuda.is_available())

    # Cargar processor
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

    # Cargar modelo
    print(f"Cargando checkpoint: {args.checkpoint}")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    model = model.to(device)

    # Cargar pesos del checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Quitar prefijo "model."
    new_state_dict = {}
    prefix = "model."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_k = k[len(prefix) :]
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # Buscar imágenes
    images_path = Path(args.images_dir)
    if not images_path.exists():
        print(f"ERROR: No existe el directorio {images_path}")
        return

    # Crear directorio de output si se solicita
    if args.save_crops:
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        print(f"Guardando crops en: {output_path}")

    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_files = sorted([f for f in images_path.iterdir() if f.suffix in image_extensions])

    if not image_files:
        print(f"No se encontraron imágenes en {images_path}")
        return

    print(f"\nEncontradas {len(image_files)} imágenes\n")
    print("=" * 80)

    with torch.no_grad():
        for img_file in image_files:
            try:
                print(f"\n📄 {img_file.name}")
                
                if args.mode == "easyocr+trocr":
                    # Modo: EasyOCR detecta, TrOCR lee el crop
                    crop, bbox, info, ocr_text = find_total_region(img_file, reader)
                    print(f"   {info}")
                    
                    # Guardar crop si se solicita
                    if args.save_crops and bbox:
                        crop_name = f"crop_{img_file.stem}.jpg"
                        crop.save(output_path / crop_name)
                else:
                    # Modo: Solo TrOCR (imagen completa)
                    crop = Image.open(img_file).convert("RGB")
                    ocr_text = None
                    print(f"   Usando imagen completa (sin EasyOCR)")
                    
                    # Guardar imagen completa si se solicita
                    if args.save_crops:
                        crop_name = f"full_{img_file.stem}.jpg"
                        crop.save(output_path / crop_name)
                
                # Preprocesar
                pixel_values = processor(crop, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(device)

                # Generar predicción
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                print(f"   TrOCR predice: '{generated_text}'")
                if ocr_text:
                    print(f"   EasyOCR detectó: '{ocr_text}'")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
