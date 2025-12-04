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


def extract_total_price(ground_truth_str: str) -> str:
    """
    Extrae total_price de ground_truth JSON string.
    """
    try:
        gt = json.loads(ground_truth_str)
    except Exception:
        return ""
    parse = gt.get("gt_parse", {})
    total = parse.get("total")
    if isinstance(total, dict):
        return str(total.get("total_price", "")).strip()
    elif isinstance(total, str):
        return total.strip()
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num", type=int, default=30, help="Ejemplos a mostrar")
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

    print("Cargando dataset CORD-v2 (validation)...")
    ds = load_dataset("naver-clova-ix/cord-v2")["validation"]

    shown = 0
    print("\n================ EJEMPLOS ================\n")

    for sample in ds:
        if shown >= args.num:
            break

        gt_str = extract_total_price(sample["ground_truth"])
        gt_num = clean_number(gt_str)
        if not gt_num:
            continue

        image = sample["image"]
        enc = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(enc.pixel_values, max_length=16)

        text = processor.batch_decode(out, skip_special_tokens=True)[0]

        # extraer solo dígitos
        digits = "".join(c for c in text if c.isdigit())
        pred_num = int(digits) if digits else 0

        print(f"Imagen: {sample['file_name'] if 'file_name' in sample else '[HF_stream]'}")
        print(f"GT total_price: {gt_num}")
        print(f"Predicción completa: \"{text}\"")
        print(f"Predicción numérica extraída: {pred_num}")
        print(f"Error absoluto: {abs(gt_num - pred_num)}")
        print("-----------------------------------------")

        shown += 1

    print("\n============= FIN DE EJEMPLOS =============\n")
