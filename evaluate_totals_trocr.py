import argparse
import json
import torch
from pathlib import Path
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

def clean_number(s):
    s = s.replace("Rp", "").replace(" ", "").replace(",", "").replace(".", "")
    return int(s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint).cuda()

    ann = json.load(open(f"{args.root}/json/validation.json"))
    maes = []

    for item in ann:
        total = item["gt_parse"].get("total", {})
        t = total.get("total_price", "")
        if not t:
            continue
        gt = clean_number(t)

        image = Image.open(f"{args.root}/image/{item['file_name']}").convert("RGB")
        enc = processor(images=image, return_tensors="pt").to("cuda")

        out = model.generate(enc.pixel_values, max_length=12)
        txt = processor.batch_decode(out, skip_special_tokens=True)[0]

        pred = "".join(c for c in txt if c.isdigit())
        pred = int(pred) if pred else 0

        maes.append(abs(gt - pred))

    print("MAE:", sum(maes) / len(maes))
    print("Samples:", len(maes))
