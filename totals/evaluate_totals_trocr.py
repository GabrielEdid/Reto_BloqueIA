import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def clean_number(s: str) -> int:
    """
    Limpia strings tipo 'Rp 13,500' o '13.500' y los pasa a entero.
    """
    if s is None:
        return 0
    s = str(s)
    s = s.replace("Rp", "")
    s = s.replace(" ", "")
    s = s.replace(",", "")
    s = s.replace(".", "")
    s = s.strip()
    if not s.isdigit():
        return 0
    return int(s)


def extract_total_info(ground_truth_str: str) -> dict:
    """
    Extrae el campo 'total_price' del dataset CORD junto con su bounding box.
    Busca específicamente en líneas que contengan 'total'.
    """
    try:
        gt = json.loads(ground_truth_str)
    except Exception:
        return None

    valid_lines = gt.get("valid_line", [])
    
    # Primero buscar "grand total", luego "total" sin "sub"
    search_patterns = [("grand", "total"), ("total",)]
    
    for patterns in search_patterns:
        for line in valid_lines:
            words = line.get("words", [])
            line_text = " ".join([w.get("text", "") for w in words]).lower()
            
            if all(p in line_text for p in patterns) and "sub" not in line_text:
                for word in reversed(words):
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
    
    # Fallback: buscar en gt_parse.total.total_price sin bbox
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Ruta al .ckpt de Lightning (por ejemplo trocr_checkpoints/totals/totals-epoch=04-val_loss=1.555.ckpt)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # IMPORTANTE: cargamos primero un modelo base de HF y luego aplicamos el state_dict del checkpoint
    print("Cargando modelo base de HuggingFace...")
    base_model_name = "microsoft/trocr-base-printed"
    processor = TrOCRProcessor.from_pretrained(base_model_name)
    model = VisionEncoderDecoderModel.from_pretrained(base_model_name)
    model.to(device)

    # Cargar state_dict desde el checkpoint de Lightning
    print(f"Cargando pesos desde checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["state_dict"]

    # El LightningModule guardó los pesos bajo la clave "model.*"
    # Hay que quitar el prefijo "model." para que coincida con el VisionEncoderDecoderModel.
    new_state_dict = {}
    prefix = "model."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_k = k[len(prefix) :]
            new_state_dict[new_k] = v
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys al cargar:", missing)
    print("Unexpected keys al cargar:", unexpected)

    model.eval()

    # Dataset de validación desde HuggingFace
    print("Cargando dataset CORD-v2 (split validation) desde HuggingFace...")
    cord = load_dataset("naver-clova-ix/cord-v2")
    val_ds = cord["validation"]
    print(f"Ejemplos en validation: {len(val_ds)}")

    maes = []
    n_used = 0
    n_no_bbox = 0

    with torch.no_grad():
        for sample in val_ds:
            info = extract_total_info(sample["ground_truth"])
            if info is None:
                continue

            gt_str = info["text"]
            bbox = info["bbox"]
            
            gt = clean_number(gt_str)
            if gt == 0:
                continue

            image = sample["image"]
            
            # Hacer crop si hay bbox
            if bbox is not None:
                try:
                    from PIL import Image
                    if not isinstance(image, Image.Image):
                        image = Image.fromarray(image)
                    
                    x_min = int(bbox["x_min"])
                    y_min = int(bbox["y_min"])
                    x_max = int(bbox["x_max"])
                    y_max = int(bbox["y_max"])
                    
                    w, h = image.size
                    x_min = max(0, min(x_min, w))
                    y_min = max(0, min(y_min, h))
                    x_max = max(x_min + 1, min(x_max, w))
                    y_max = max(y_min + 1, min(y_max, h))
                    
                    image = image.crop((x_min, y_min, x_max, y_max))
                except Exception as e:
                    print(f"Warning: Failed to crop, using full image: {e}")
            else:
                n_no_bbox += 1
            
            enc = processor(images=image, return_tensors="pt").to(device)

            out = model.generate(enc.pixel_values, max_length=32)
            txt = processor.batch_decode(out, skip_special_tokens=True)[0]

            pred_digits = "".join(c for c in txt if c.isdigit())
            pred = int(pred_digits) if pred_digits else 0

            maes.append(abs(gt - pred))
            n_used += 1

    if n_used == 0:
        print("No se encontraron samples válidos para calcular MAE.")
        return

    mae = sum(maes) / len(maes)
    median_ae = sorted(maes)[len(maes) // 2]
    max_error = max(maes)
    min_error = min(maes)
    
    print("=====================================")
    print(f"Samples evaluados: {n_used}")
    print(f"Samples sin bbox: {n_no_bbox}")
    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"Median AE: {median_ae:.2f}")
    print(f"Min Error: {min_error:.2f}")
    print(f"Max Error: {max_error:.2f}")
    print("=====================================")


if __name__ == "__main__":
    main()
