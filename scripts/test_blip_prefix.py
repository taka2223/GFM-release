"""
Compare BLIP captioning strategies on MorphBench castle images.

Tests:
  - models: blip-base vs blip-large
  - prefix conditioning: none / "a photo of" / "a detailed photograph of"
  - generation: greedy / beam-search / beam + repetition_penalty

Prints a table of (model, prefix, gen) -> caption per image.
"""

import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

CACHE_DIR = "/cns/USERS/zzhixuan/weights"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGES = {
    "castle_0": "/cns/USERS/zzhixuan/data/MorphBench/Metamorphosis/castle_0.png",
    "castle_1": "/cns/USERS/zzhixuan/data/MorphBench/Metamorphosis/castle_1.png",
}

MODELS = [
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip-image-captioning-large",
]

PREFIXES = [None, "a photo of", "a detailed photograph of"]

GEN_CONFIGS = {
    "greedy":      dict(num_beams=1),
    "beam3":       dict(num_beams=3),
    "beam3+rep1.5": dict(num_beams=3, repetition_penalty=1.5),
    "beam5+rep1.5": dict(num_beams=5, repetition_penalty=1.5),
}


def run_for_model(model_id: str):
    print(f"\n{'='*78}\nModel: {model_id}\n{'='*78}")
    processor = BlipProcessor.from_pretrained(model_id, cache_dir=CACHE_DIR)
    model = BlipForConditionalGeneration.from_pretrained(
        model_id, cache_dir=CACHE_DIR, torch_dtype=torch.float16
    ).to(DEVICE)
    model.eval()

    for img_name, img_path in IMAGES.items():
        image = Image.open(img_path).convert("RGB")
        print(f"\n--- {img_name} ---")
        for prefix in PREFIXES:
            for gen_name, gen_kwargs in GEN_CONFIGS.items():
                if prefix is None:
                    inputs = processor(image, return_tensors="pt").to(DEVICE, torch.float16)
                else:
                    inputs = processor(image, text=prefix, return_tensors="pt").to(
                        DEVICE, torch.float16
                    )
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=40, **gen_kwargs)
                caption = processor.decode(out[0], skip_special_tokens=True)
                tag = f"prefix={prefix!r:<30}  gen={gen_name:<14}"
                print(f"  {tag} -> {caption}")

    del model, processor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for model_id in MODELS:
        run_for_model(model_id)
