from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from generator import generator
from fastapi.middleware.cors import CORSMiddleware
import numpy
from io import BytesIO
from PIL import Image
import base64
from Diffusion import model, ema, sample

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

weights_path = "weights/generator_weights.pth"
generator.load_state_dict(torch.load(weights_path, map_location=torch.device("cuda")))
generator.eval()

checkpoint = torch.load("weights/model_final.pth", weights_only=True, map_location=torch.device("cuda"))
model.load_state_dict(checkpoint)
device = torch.device("cuda")
generator=generator.to(device)

class GenerateRequest(BaseModel):
    model: str

@app.post("/generate")
def generate(request: GenerateRequest):
    model_type = request.model
    if model_type == "dcgan":
        latent_vector = torch.randn(1, 256, 1, 1).to(device)
        with torch.no_grad():
            generated_image = generator(latent_vector).squeeze(0)

        generated_image = (generated_image + 1) / 2  # Normalize to [0, 1]
        generated_image = generated_image.permute(1, 2, 0).cpu().numpy()
        pil_image = Image.fromarray((generated_image * 255).astype("uint8"))

    elif model_type == "diffusion":
        generated_images = sample(
            model=model.to(device),  
            n_samples=1,         
            device=device.type,   
            img_size=128,      
            channels=3,        
            sample_steps=200,  
            eta=0.2                 
        )

        generated_image = generated_images[0]

        generated_image = (generated_image + 1) / 2  # Normalize to [0, 1]
        generated_image = generated_image.permute(1, 2, 0).cpu().numpy()
        pil_image = Image.fromarray((generated_image * 255).astype("uint8"))

    else:
        return {"error": "Unsupported model type"}

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.read()).decode("utf-8")

    return {"generated_image": f"data:image/png;base64,{base64_image}"}