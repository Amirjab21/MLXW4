from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import numpy as np

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load CLIP model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def process_image(image_bytes):
    """Helper function to process image bytes into PIL Image"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

@app.post("/submit")
async def process_upload(
    image: UploadFile = File(...),
    text: str = Form(...)
):
    print("image")
    try:
        # Validate image file
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File uploaded is not an image")
        
        # Read image content
        image_content = await image.read()
        
        # Process image
        try:
            image_pil = process_image(image_content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
        
        # Validate text
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text query cannot be empty")
        
        # Process image and text with CLIP
        try:
            inputs = processor(
                images=image_pil,
                text=[text],
                return_tensors="pt",
                padding=True
            ).to(device)

            # Get image and text features
            with torch.no_grad():
                outputs = model(**inputs)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds

            # Calculate similarity
            similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
            
            # Convert to Python float for JSON serialization
            similarity_score = similarity.item()

            return {
                "results": [
                    {
                        "document": "Image-Text Similarity Score",
                        "distance": f"{similarity_score:.4f}"
                    }
                ]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing with CLIP: {str(e)}")

    except HTTPException as he:
        return {"error": "1"}
    except Exception as e:
        return {"error": "2"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)