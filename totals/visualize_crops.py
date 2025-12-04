#!/usr/bin/env python3
"""
Script para visualizar los crops de totales del dataset CORD.
Útil para verificar que los bounding boxes funcionan correctamente.
"""

import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset


def extract_total_info(ground_truth_str: str) -> dict:
    """
    Extrae el campo 'total_price' del dataset CORD junto con su bounding box.
    """
    try:
        gt = json.loads(ground_truth_str)
    except Exception:
        return None

    valid_lines = gt.get("valid_line", [])
    
    for line in valid_lines:
        words = line.get("words", [])
        for word in words:
            text = word.get("text", "").strip()
            quad = word.get("quad", {})
            
            if text and quad:
                clean_text = text.replace("Rp", "").replace(".", "").replace(",", "").replace(" ", "").strip()
                if clean_text.isdigit() and len(clean_text) >= 4:
                    try:
                        x_coords = [quad.get(f"x{i}", 0) for i in range(1, 5)]
                        y_coords = [quad.get(f"y{i}", 0) for i in range(1, 5)]
                        
                        x_min = min(x_coords)
                        y_min = min(y_coords)
                        x_max = max(x_coords)
                        y_max = max(y_coords)
                        
                        margin = 10
                        
                        return {
                            "text": clean_text,
                            "bbox": {
                                "x_min": max(0, x_min - margin),
                                "y_min": max(0, y_min - margin),
                                "x_max": x_max + margin,
                                "y_max": y_max + margin,
                            }
                        }
                    except Exception:
                        continue
    
    parse = gt.get("gt_parse", {})
    total = parse.get("total")
    
    if isinstance(total, dict):
        total_price = str(total.get("total_price", "")).strip()
    elif isinstance(total, str):
        total_price = total.strip()
    else:
        return None
        
    if total_price:
        clean = total_price.replace("Rp", "").replace(".", "").replace(",", "").replace(" ", "").strip()
        if clean.isdigit():
            return {"text": clean, "bbox": None}
    
    return None


def visualize_crops(num_samples=10, split="train", save_dir="totals_visualizations"):
    """
    Visualiza los crops de totales y guarda las imágenes.
    """
    print(f"Cargando dataset CORD-v2 (split: {split})...")
    ds = load_dataset("naver-clova-ix/cord-v2")[split]
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    print(f"Guardando visualizaciones en: {save_path.absolute()}\n")
    
    saved = 0
    processed = 0
    no_bbox_count = 0
    no_text_count = 0
    
    for idx, sample in enumerate(ds):
        if saved >= num_samples:
            break
            
        processed += 1
        
        info = extract_total_info(sample["ground_truth"])
        if info is None:
            no_text_count += 1
            continue
        
        text = info["text"]
        bbox = info["bbox"]
        
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Crear figura con imagen original y crop
        if bbox is not None:
            # Dibujar bbox en imagen original
            img_with_bbox = image.copy()
            draw = ImageDraw.Draw(img_with_bbox)
            
            x_min = int(bbox["x_min"])
            y_min = int(bbox["y_min"])
            x_max = int(bbox["x_max"])
            y_max = int(bbox["y_max"])
            
            # Validar bbox
            w, h = image.size
            x_min = max(0, min(x_min, w))
            y_min = max(0, min(y_min, h))
            x_max = max(x_min + 1, min(x_max, w))
            y_max = max(y_min + 1, min(y_max, h))
            
            # Dibujar rectángulo rojo
            draw.rectangle(
                [(x_min, y_min), (x_max, y_max)],
                outline="red",
                width=3
            )
            
            # Hacer crop
            cropped = image.crop((x_min, y_min, x_max, y_max))
            
            # Crear imagen combinada
            combined_width = img_with_bbox.width + cropped.width + 20
            combined_height = max(img_with_bbox.height, cropped.height) + 60
            
            combined = Image.new("RGB", (combined_width, combined_height), "white")
            combined.paste(img_with_bbox, (0, 30))
            combined.paste(cropped, (img_with_bbox.width + 20, 30))
            
            # Añadir texto con PIL
            draw = ImageDraw.Draw(combined)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, 5), f"Sample {idx} - Total: {text}", fill="black", font=font)
            draw.text((10, combined_height - 25), "Original con bbox", fill="red", font=font)
            draw.text((img_with_bbox.width + 30, combined_height - 25), "Crop", fill="green", font=font)
            
            status = "✅ CON BBOX"
            
        else:
            # No hay bbox, solo mostrar imagen completa
            no_bbox_count += 1
            combined = image.copy()
            draw = ImageDraw.Draw(combined)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            except:
                font = ImageFont.load_default()
            draw.text((10, 5), f"Sample {idx} - Total: {text} [SIN BBOX]", fill="red", font=font)
            status = "⚠️  SIN BBOX"
        
        # Guardar
        filename = f"sample_{saved:03d}_total_{text}_{status.replace(' ', '_')}.png"
        combined.save(save_path / filename)
        
        print(f"[{saved+1}/{num_samples}] Guardado: {filename} - {status}")
        
        saved += 1
    
    print(f"\n{'='*60}")
    print(f"Resumen:")
    print(f"  Samples procesados: {processed}")
    print(f"  Visualizaciones guardadas: {saved}")
    print(f"  Samples sin bbox: {no_bbox_count}")
    print(f"  Samples sin texto: {no_text_count}")
    print(f"  Ubicación: {save_path.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualiza crops de totales del dataset CORD"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=20,
        help="Número de ejemplos a visualizar"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Split del dataset a usar"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="totals_visualizations",
        help="Directorio donde guardar las imágenes"
    )
    
    args = parser.parse_args()
    
    visualize_crops(
        num_samples=args.num,
        split=args.split,
        save_dir=args.output
    )
