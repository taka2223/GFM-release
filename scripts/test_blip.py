from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",cache_dir="/cns/USERS/zzhixuan/weights")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",cache_dir="/cns/USERS/zzhixuan/weights")

image = Image.open("/cns/USERS/zzhixuan/data/MorphBench/Animation/counterfeit_0.png")
inputs = processor(image, return_tensors="pt",)
caption = processor.decode(model.generate(**inputs)[0], skip_special_tokens=True)
print(caption)
# → "a white cat sitting on a table"