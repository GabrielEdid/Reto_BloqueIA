"""
Test docTR OCR on Real Tickets
Evaluates pre-trained docTR on real-world receipt images from tickets_de_prueba/
"""

import os
import csv
from datetime import datetime
from typing import List, Tuple
from pathlib import Path

import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor


def load_ticket_images(folder_path: str) -> List[Tuple[str, np.ndarray]]:
    """Load all ticket images from folder."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = []
    
    folder = Path(folder_path)
    for img_path in sorted(folder.iterdir()):
        if img_path.suffix.lower() in valid_extensions:
            img = cv2.imread(str(img_path))
            if img is not None:
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append((img_path.name, img_rgb))
                print(f"✓ Loaded: {img_path.name} - Shape: {img_rgb.shape}")
    
    return images


def extract_text_from_ticket(model, image: np.ndarray) -> str:
    """Extract text from ticket image using docTR."""
    # docTR expects uint8 RGB
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Run OCR
    result = model([image])
    
    # Extract text hierarchically: pages -> blocks -> lines -> words
    all_text = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join([word.value for word in line.words])
                if line_text.strip():
                    all_text.append(line_text)
    
    return "\n".join(all_text)


def main():
    print("=" * 80)
    print("docTR OCR - Real Ticket Evaluation")
    print("=" * 80)
    
    # Configuration
    tickets_folder = "./tickets_de_prueba"
    det_arch = "db_resnet50"
    reco_arch = "crnn_vgg16_bn"
    
    print(f"\nConfiguration:")
    print(f"  Tickets folder: {tickets_folder}")
    print(f"  Detection: {det_arch}")
    print(f"  Recognition: {reco_arch}")
    print()
    
    # Load model
    print("Loading docTR model...")
    model = ocr_predictor(
        det_arch=det_arch,
        reco_arch=reco_arch,
        pretrained=True,
    )
    print("✓ Model loaded successfully")
    print()
    
    # Load ticket images
    print("Loading ticket images...")
    tickets = load_ticket_images(tickets_folder)
    print(f"\n✓ Loaded {len(tickets)} ticket images")
    print()
    
    # Process each ticket
    print("=" * 80)
    print("Processing tickets...")
    print("=" * 80)
    
    results = []
    for i, (filename, image) in enumerate(tickets, 1):
        print(f"\n[{i}/{len(tickets)}] Processing: {filename}")
        
        # Extract text
        extracted_text = extract_text_from_ticket(model, image)
        
        # Store result
        results.append({
            'filename': filename,
            'extracted_text': extracted_text,
            'char_count': len(extracted_text),
            'line_count': len(extracted_text.split('\n'))
        })
        
        # Print preview
        preview = extracted_text[:200].replace('\n', ' ')
        print(f"  Extracted {len(extracted_text)} chars, {len(extracted_text.split())} lines")
        print(f"  Preview: {preview}...")
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./real_tickets_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Detailed results CSV
    csv_file = os.path.join(results_dir, "ticket_predictions.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'char_count', 'line_count', 'extracted_text'])
        for result in results:
            writer.writerow([
                result['filename'],
                result['char_count'],
                result['line_count'],
                result['extracted_text']
            ])
    
    # Save individual text files for each ticket
    for result in results:
        txt_file = os.path.join(results_dir, f"{Path(result['filename']).stem}.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(result['extracted_text'])
    
    # Print summary
    print()
    print("=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total tickets processed: {len(results)}")
    print(f"Results saved to: {results_dir}/")
    print(f"  - CSV summary: {csv_file}")
    print(f"  - Individual .txt files for each ticket")
    print()
    print("Sample results:")
    for i, result in enumerate(results[:3], 1):
        print(f"\n[{i}] {result['filename']}")
        print(f"    Characters: {result['char_count']}, Lines: {result['line_count']}")
        preview = result['extracted_text'][:150].replace('\n', ' | ')
        print(f"    Text: {preview}...")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
