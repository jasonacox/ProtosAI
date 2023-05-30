#!/usr/bin/python3
"""
Handwriting Recognition 

Author: Jason A. Cox
29 May 2023
https://github.com/jasonacox/ProtosAI

This uses the HuggingFace transformer https://huggingface.co/

Credit for microsoft/trocr-base-handwritten model:
M. Li, T. Lv, L. Cui, Y. Lu, D. Florencio, C. Zhang, Z. Li, and F. Wei, 
"TrOCR: Transformer-based Optical Character Recognition with Pre-trained 
Models," arXiv:2109.10282 [cs.CL], 2021.

"""
# import
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import transformers
transformers.logging.set_verbosity_error()
from PIL import Image
import requests

def convert(source):
        
    # load image
    if source.startswith("http"):
        image = Image.open(requests.get(source, stream=True).raw).convert("RGB")
    else:
        image = Image.open(source).convert("RGB")
    
    # load model
    print("Loading transformer...")
    model_id = "microsoft/trocr-base-handwritten"
    print(f" * {model_id}")
    processor = transformers.TrOCRProcessor.from_pretrained(model_id)
    model = transformers.VisionEncoderDecoderModel.from_pretrained(model_id)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # text recognition
    print(f"\nAnalyzing handwriting from {source}...")
    generated_ids = model.generate(pixel_values, max_length=200)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # output
    print("\nResulting text:")
    print(generated_text)

# main
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <filename or URL to image>")
        sys.exit(1)
    
    fn = sys.argv[1]
    print(f"Converting image to text: {fn}\n")
    convert(fn)

