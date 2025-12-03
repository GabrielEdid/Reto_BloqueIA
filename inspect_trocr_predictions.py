#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def extract_total_from_ground_truth(ground_truth_str: str) -> Optional[int]:
    """
    Extrae el total numérico desde el JSON de CORD-v2.
    Intenta leer:
      gt_parse.total.total_price
    y si no, busca el primer número grande dentro de total.
    Devuelve un entero sin comas ni puntos (por ejemplo 41,000 -> 41000).
    """
    try:
        gt = json.loads(ground_truth_str)
    except Exception:
        return None

    parse = gt.get("gt_parse", {})
    total_field = parse.get("total", None)

    # Caso típico: dict con 'total_price'
    if isinstance(total_field, dict):
        for key in ["total_price", "totalprice", "price"]:
            if key in total_field:
                value_str = str(total_field[key])
                digits = re.sub(r"[^\d]", "", value_str)
                if digits:
                    try:
                        return int(digits)
                    except ValueError:
                        pass

    # Fallback: buscar números dentro de la representación en string
    combined = str(total_field)
    matches = re.findall(r"\d[\d.,]*", combined)
    if matches:
        digits = re.sub(r"[^\d]", "", matches[0])
        if digits:
            try:
                return int(digits)
            except ValueError:
                return None

    return None


def extract_number_from_text(text: str) -> Optional[int]:
    """
    Extrae el primer número "grande" de un texto como:
      'TOTAL 41,000' -> 41000
      'total 20.000' -> 20000
    Si no encuentra número, devuelve None.
    """
    matches = re.findall(r"\d[\d.,]*", text)
    if not matches:
        return None

    digits = re.sub(r"[^\d]", "", matches[0])
    if not digits:
        return None

    try:
        return int(digits)
    except ValueError:
        return None


def load_model_and_processor(checkpoint_path: str, device: torch.device):
    print(f"Device: {device}")
    print(f"Using checkpoint: {checkpoint_path}")

    print("Loading base model and processor...")
    base_model_name = "microsoft/trocr-base-printed"
    processor = TrOCRProcessor.from_pretrained(base_model_name)
    model = VisionEncoderDecoderModel.from_pretrained(base_model_name)

    # Cargar weights desde checkpoint de Lightning
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    # Los pesos del modelo en Lightning suelen ir con prefijo "model."
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_key = k.replace("model.", "", 1)
            new_state_dict[new_key] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()

    return model, processor


def evaluate_split(
    model,
    processor,
    split_name: str,
    num_examples_to_print: int,
    device: torch.device
):
    print(f"Loading CORD-v2 {split_name} split...")
    dataset = load_dataset("naver-clova-ix/cord-v2")[split_name]

    n = len(dataset)
    print(f"Total examples in {split_name}: {n}")

    abs_errors: List[float] = []
    rel_errors: List[float] = []
    printed = 0

    for idx, sample in enumerate(dataset):
        image = sample["image"]
        gt_str = sample["ground_truth"]

        gt_total = extract_total_from_ground_truth(gt_str)
        if gt_total is None:
            continue

        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=16,
                num_beams=4,
                early_stopping=True
            )

        pred_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        pred_total = extract_number_from_text(pred_text)

        if pred_total is None:
            continue

        err = abs(pred_total - gt_total)
        abs_errors.append(err)

        if gt_total > 0:
            rel_errors.append(err / gt_total)

        if printed < num_examples_to_print:
            printed += 1
            print("=" * 80)
            print(f"Ejemplo {idx}")
            print(f"- GROUND TRUTH raw gt_parse total: {gt_str}")
            print(f"- GROUND TRUTH total_num: {gt_total}")
            print(f"- PRED TEXT: {pred_text}")
            print(f"- PRED total_num: {pred_total}")
            print(f"- ABS ERROR: {err}")
            if gt_total > 0:
                print(f"- REL ERROR: {err / gt_total:.4f}")

    print("=" * 80)
    if abs_errors:
        mae = sum(abs_errors) / len(abs_errors)
        print(f"Evaluated samples with valid numeric totals: {len(abs_errors)}")
        print(f"Mean Absolute Error (MAE) sobre el total: {mae:.2f}")

        if rel_errors:
            mape = sum(rel_errors) / len(rel_errors)
            print(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")
    else:
        print("No se pudieron calcular errores, no hubo ejemplos válidos.")


def main():
    parser = argparse.ArgumentParser(
        description="Inspecciona y evalúa predicciones de TOTAL en CORD-v2 usando un checkpoint de TrOCR"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Ruta al checkpoint de Lightning (.ckpt)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Split de CORD-v2 a evaluar"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Número de ejemplos que se imprimirán en detalle"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model_and_processor(args.checkpoint, device)

    evaluate_split(
        model=model,
        processor=processor,
        split_name=args.split,
        num_examples_to_print=args.num_examples,
        device=device
    )


if __name__ == "__main__":
    main()
