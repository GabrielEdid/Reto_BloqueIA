import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def clean_number(s: str) -> int:
    """
    Limpia strings tipo “Rp 13,500” a 13500.
    """
    if s is None:
        return 0
    s = str(s)
    s = s.replace("Rp", "")
    s = s.replace(" ", "")
    s = s.replace(",", "")
    s = s.replace(".", "")
    s = s.strip()
    return int(s) if s.isdigit() else 0


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num", type=int, default=30, help="Ejemplos a mostrar")
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Split del dataset (por defecto: validation)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # Modelo base + pesos del checkpoint
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    model.to(device)

    print(f"Cargando checkpoint Lightning: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd = ckpt["state_dict"]

    fixed_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            fixed_sd[k.replace("model.", "")] = v

    model.load_state_dict(fixed_sd, strict=False)
    model.eval()

    print(f"Cargando dataset CORD-v2 ({args.split})...")
    ds = load_dataset("naver-clova-ix/cord-v2")[args.split]

    shown = 0
    n_no_bbox = 0
    print("\n================ EJEMPLOS ================\n")

    for sample in ds:
        if shown >= args.num:
            break

        info = extract_total_info(sample["ground_truth"])
        if info is None:
            continue
            
        gt_str = info["text"]
        bbox = info["bbox"]
        gt_num = clean_number(gt_str)
        
        if not gt_num:
            continue

        image = sample["image"]
        from PIL import Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Hacer crop si hay bbox
        has_bbox = bbox is not None
        if bbox is not None:
            try:
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
                print(f"Warning: Failed to crop: {e}")
                has_bbox = False
        else:
            n_no_bbox += 1
            
        enc = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(enc.pixel_values, max_length=32)

        text = processor.batch_decode(out, skip_special_tokens=True)[0]

        # extraer solo dígitos
        digits = "".join(c for c in text if c.isdigit())
        pred_num = int(digits) if digits else 0

        print(f"Imagen: {sample['file_name'] if 'file_name' in sample else '[HF_stream]'}")
        print(f"Usado bbox crop: {'SÍ' if has_bbox else 'NO (imagen completa)'}")
        print(f"GT total_price: {gt_num}")
        print(f"Predicción completa: \"{text}\"")
        print(f"Predicción numérica extraída: {pred_num}")
        print(f"Error absoluto: {abs(gt_num - pred_num)}")
        print("-----------------------------------------")

        shown += 1

    print(f"\nSamples sin bbox: {n_no_bbox}")
    print("\n============= FIN DE EJEMPLOS =============\n")
