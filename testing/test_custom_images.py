#!/usr/bin/env python3
"""
Script para probar el modelo TrOCR entrenado con imágenes de tickets personalizadas.
"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

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
                # Cargar imagen
                image = Image.open(img_file).convert("RGB")
                
                # Preprocesar
                pixel_values = processor(image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(device)

                # Generar predicción
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                print(f"\n📄 {img_file.name}")
                print(f"   Predicción: {generated_text}")
                
            except Exception as e:
                print(f"\n❌ Error procesando {img_file.name}: {e}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
