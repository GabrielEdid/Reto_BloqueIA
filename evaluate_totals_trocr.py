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


def extract_total_price(ground_truth_str: str) -> str:
    """
    Extrae el campo total_price de ground_truth (string JSON en CORD-v2).
    Espera algo tipo:
      {
        "gt_parse": {
          ...
          "total": {
            "total_price": "...",
            ...
          }
        }
      }
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
    else:
        return ""


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

    with torch.no_grad():
        for sample in val_ds:
            gt_str = extract_total_price(sample["ground_truth"])
            if not gt_str:
                continue

            gt = clean_number(gt_str)
            if gt == 0:
                continue

            image = sample["image"]
            enc = processor(images=image, return_tensors="pt").to(device)

            out = model.generate(enc.pixel_values, max_length=16)
            txt = processor.batch_decode(out, skip_special_tokens=True)[0]

            pred_digits = "".join(c for c in txt if c.isdigit())
            pred = int(pred_digits) if pred_digits else 0

            maes.append(abs(gt - pred))
            n_used += 1

    if n_used == 0:
        print("No se encontraron samples válidos para calcular MAE.")
        return

    mae = sum(maes) / len(maes)
    print("=====================================")
    print(f"Samples evaluados: {n_used}")
    print(f"MAE total_price (validation): {mae:.2f}")
    print("=====================================")


if __name__ == "__main__":
    main()
