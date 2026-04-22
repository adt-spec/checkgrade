import io
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

app = FastAPI()

# 1. Load a pre-trained Vision Model (ResNet18)
# We remove the final classification layer because we just want the feature embeddings, not to classify cats vs dogs.
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = nn.Identity() 
model.eval() # Set to evaluation mode

# 2. Standard image preprocessing for PyTorch models
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_embedding(image_bytes):
    """Converts raw image bytes into a mathematical feature vector."""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) # Create a mini-batch as expected by the model
    
    with torch.no_grad():
        embedding = model(input_batch)
    return embedding

@app.post("/analyze-zone")
async def analyze_zone(standard_img: UploadFile = File(...), actual_img: UploadFile = File(...)):
    """API Endpoint to compare two images and return a 5S score."""
    
    # Read image bytes
    standard_bytes = await standard_img.read()
    actual_bytes = await actual_img.read()
    
    # Get embeddings for both images
    embed_standard = get_image_embedding(standard_bytes)
    embed_actual = get_image_embedding(actual_bytes)
    
    # Calculate Cosine Similarity (Result is between -1.0 and 1.0)
    similarity = F.cosine_similarity(embed_standard, embed_actual).item()
    
    # Map the similarity score to your CheckGrade 1 to 5 scale
    # Assuming a similarity > 0.4 is essentially a 1, and 0.95+ is a perfect 5
    mapped_score = max(1.0, min(5.0, ((similarity - 0.4) / (0.95 - 0.4)) * 4 + 1))
    
    # Round to one decimal place for the app UI
    final_score = round(mapped_score, 1)

    return {
        "similarity_raw": round(similarity, 3),
        "checkgrade_score": final_score
    }

# To run this server locally, you would type this in your terminal:
# uvicorn main:app --reload