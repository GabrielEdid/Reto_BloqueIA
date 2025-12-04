#!/usr/bin/env python3
"""
Script para probar que extract_total_info() funciona correctamente.
"""

import json
from datasets import load_dataset


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
    print("Cargando dataset CORD-v2...")
    ds = load_dataset("naver-clova-ix/cord-v2", split="train")
    
    print(f"\nProbando extract_total_info() en {min(10, len(ds))} ejemplos:\n")
    print("=" * 80)
    
    matches = 0
    errors = 0
    
    for i in range(min(10, len(ds))):
        sample = ds[i]
        gt_str = sample["ground_truth"]
        gt_dict = json.loads(gt_str)
        
        # Ground truth real del dataset
        gt_parse = gt_dict.get("gt_parse", {})
        gt_total = gt_parse.get("total", {})
        if isinstance(gt_total, dict):
            expected = gt_total.get("total_price", "").replace(".", "").replace(",", "").replace(" ", "").strip()
        else:
            expected = ""
        
        # Lo que extrae nuestra función
        info = extract_total_info(gt_str)
        
        if info:
            extracted = info["text"]
            has_bbox = info["bbox"] is not None
            
            # Verificar si coincide
            match = extracted == expected
            if match:
                matches += 1
                status = "✅ MATCH"
            else:
                errors += 1
                status = "❌ MISMATCH"
            
            print(f"Sample {i}:")
            print(f"  Expected:  {expected}")
            print(f"  Extracted: {extracted}")
            print(f"  Has bbox:  {has_bbox}")
            print(f"  Status:    {status}")
            print("-" * 80)
        else:
            errors += 1
            print(f"Sample {i}:")
            print(f"  Expected:  {expected}")
            print(f"  Extracted: None")
            print(f"  Status:    ❌ NO DATA")
            print("-" * 80)
    
    print("\n" + "=" * 80)
    print(f"RESUMEN:")
    print(f"  Matches:  {matches}/10  ({matches*10}%)")
    print(f"  Errors:   {errors}/10  ({errors*10}%)")
    print("=" * 80)
    
    if matches == 10:
        print("\n🎉 ¡Perfecto! La función extrae correctamente los totales.")
    elif matches >= 8:
        print("\n✅ Muy bien, la mayoría de extracciones son correctas.")
    else:
        print("\n⚠️  Atención: Hay varios errores en la extracción.")


if __name__ == "__main__":
    main()
