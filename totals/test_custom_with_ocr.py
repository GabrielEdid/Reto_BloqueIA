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
    Usa EasyOCR para encontrar la región que contiene "TOTAL" y el número asociado.
    Retorna el crop de toda la línea con el número.
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
        return image, None, "No se encontró 'TOTAL' en la imagen"
    
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
        # Verificar si contiene dígitos
        if not any(c.isdigit() for c in text):
            continue
        
        # Calcular centro vertical del bbox
        bbox_ys = [point[1] for point in bbox]
        bbox_y_center = (min(bbox_ys) + max(bbox_ys)) / 2
        
        # Verificar si está en la misma línea (±20px) o justo debajo (hasta +50px)
        y_diff = bbox_y_center - total_y_center
        if -20 <= y_diff <= 50:
            bbox_xs = [point[0] for point in bbox]
            nearby_numbers.append({
                'bbox': bbox,
                'text': text,
                'conf': conf,
                'x_min': int(min(bbox_xs)),
                'x_max': int(max(bbox_xs)),
                'y_min': int(min(bbox_ys)),
                'y_max': int(max(bbox_ys)),
                'y_diff': abs(y_diff)
            })
    
    # Si no hay números cercanos, usar solo el bbox de TOTAL expandido
    if not nearby_numbers:
        x_min, x_max = total_x_min, total_x_max
        y_min, y_max = total_y_min, total_y_max
        info = f"'{total_text}' sin número visible"
    else:
        # Ordenar por cercanía vertical y tomar el más cercano
        nearby_numbers.sort(key=lambda x: x['y_diff'])
        best_number = nearby_numbers[0]
        
        # Combinar bboxes (TOTAL + número)
        x_min = min(total_x_min, best_number['x_min'])
        x_max = max(total_x_max, best_number['x_max'])
        y_min = min(total_y_min, best_number['y_min'])
        y_max = max(total_y_max, best_number['y_max'])
        
        info = f"'{total_text}' + número '{best_number['text']}'"
    
    # Añadir margen generoso
    margin = 15
    width, height = image.size
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(width, x_max + margin)
    y_max = min(height, y_max + margin)
    
    # Hacer crop
    crop = image.crop((x_min, y_min, x_max, y_max))
    
    return crop, (x_min, y_min, x_max, y_max), info


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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    # Inicializar EasyOCR
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
                
                # Detectar región del total
                crop, bbox, info = find_total_region(img_file, reader)
                print(f"   {info}")
                
                # Guardar crop si se solicita
                if args.save_crops and bbox:
                    crop_name = f"crop_{img_file.stem}.jpg"
                    crop.save(output_path / crop_name)
                
                # Preprocesar
                pixel_values = processor(crop, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(device)

                # Generar predicción
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                print(f"   Predicción: '{generated_text}'")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
