import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "sd2-community/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16,cache_dir="/cns/USERS/zzhixuan/weights")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a photo of a white cat"
image = pipe(prompt).images[0]
    
image.save("astronaut_rides_horse.png")
