from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import torch
import torch.nn as nn
from Transformer import Transformer, Decoder, DecoderLayer
from Attention import MultiHeadAttention
import transformers
from datasets import load_dataset
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Copy all the setup code from inference.py

device = torch.device("cpu")
CLIP = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

tokenizer = processor.tokenizer
vocab_size = tokenizer.vocab_size
print("first vocab size", vocab_size)
tokenizer.add_special_tokens({"pad_token": "<<<PAD>>>"})
vocab = tokenizer.get_vocab()  # Returns dict of {token: index}
tokenizer = processor.tokenizer
vocab_size = tokenizer.vocab_size
vocab_size_final = vocab_size + 1 #add 1 here
print("second vocab size", vocab_size)
reverse_vocab = {idx: token for token, idx in vocab.items()}

clip_text_model = CLIP.text_model

d_model = 512
text_dimension_embedding = 512
image_encoder_output_dim = 768
n_loops = 6
num_heads = 8

# self_attn_layer = Attention_Layer(d_model=d_model, num_heads=1)
self_attn_layer = MultiHeadAttention(encoder_output_dim=d_model, decoder_dim=d_model, d_model=d_model, num_heads=num_heads)
cross_attn_layer = MultiHeadAttention(encoder_output_dim=image_encoder_output_dim, decoder_dim=text_dimension_embedding, d_model=d_model, num_heads=num_heads)

feed_forward = nn.Sequential(nn.Linear(d_model, 2048), nn.ReLU(), nn.Linear(2048, d_model))

text_model = CLIP.text_model
text_embedder = text_model.embeddings

decoder_layer = DecoderLayer(input_dim=text_dimension_embedding, tgt_vocab_size=vocab_size_final, intermediate_attn_dim=d_model, n_loops=n_loops, feed_forward=feed_forward, self_attn_layer=self_attn_layer, cross_attn_layer=cross_attn_layer)



decoder = Decoder(vocab_size_final, pad_token=tokenizer.pad_token_id, embedding_layer=text_embedder, layer=decoder_layer, n_loops=n_loops,d_model=d_model)
transformer = Transformer(d_model=d_model, text_encoder=text_embedder, image_encoder=CLIP.vision_model, decoder=decoder, tgt_vocab_size=vocab_size_final, pad_token=tokenizer.pad_token_id)

# data = __getitem__(0, transformed_images)
checkpoint = torch.load("checkpoints/model_exp_decay_corrected.pt", map_location=torch.device("cpu"))
transformer.load_state_dict(checkpoint['model_state_dict'])
print("here")
def evaluate_topk(transformer, data, k):
    transformer.eval()
    MAX_SEQ_LENGTH = 35
    START_TOKEN_ID = 49406
    # PAD_TOKEN_ID = 49408
    BEAM_WIDTH = k
    
    # Initialize beam with start token
    current_sequences = torch.full((BEAM_WIDTH, 1), START_TOKEN_ID)
    predicted_indices_list = []
    print("data", data)
    print(data['image']['pixel_values'].unsqueeze(0).shape)
    # Generate first token with beam search
    output = transformer.forward(data['image']['pixel_values'].repeat(BEAM_WIDTH, 1, 1, 1), current_sequences)
    print("output", output)
    output_probabilities = torch.softmax(output, dim=2)
    print("output_probabilities", output_probabilities)
    top_k_values, top_k_indices = torch.topk(output_probabilities[0, 0], k=BEAM_WIDTH)
    print("top_k_values", top_k_values)
    print("top_k_indices", top_k_indices)
    
    # Create BEAM_WIDTH different sequences
    for beam_idx in range(BEAM_WIDTH):
        current_sequence = torch.full((1, 1), START_TOKEN_ID)
        predicted_indices = torch.zeros((1, 1))
        
        # Use the beam_idx-th best first token
        first_token = top_k_indices[beam_idx].item()
        current_sequence = torch.cat((current_sequence, torch.tensor([first_token]).unsqueeze(0)), dim=1)
        predicted_indices[0, 0] = first_token
        
        # Continue generating remaining tokens
        for pos in range(1, MAX_SEQ_LENGTH):
            output = transformer.forward(data['image']['pixel_values'], current_sequence)
            output_probabilities = torch.softmax(output, dim=2)
            predicted_digit = torch.argmax(output_probabilities[0, pos])
            
            if pos < MAX_SEQ_LENGTH - 1:
                current_sequence = torch.cat((current_sequence, torch.tensor([predicted_digit.item()]).unsqueeze(0)), dim=1)
            
            predicted_indices = torch.cat((predicted_indices, predicted_digit.unsqueeze(0).unsqueeze(0)), dim=1)
        
        # Convert to text and store
        predicted_digits = [reverse_vocab[idx.item()] for idx in predicted_indices.squeeze(0)]
        cleaned_text = ' '.join(token.replace('</w>', '') for token in predicted_digits 
                              if token not in ['<|endoftext|>', '<<<PAD>>>'])
        predicted_indices_list.append(cleaned_text)
    print(predicted_indices_list, "PREDICTED INDICES LIST")
    return predicted_indices_list
model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def __getitem__(image):
    processed_image = processor(images=image, return_tensors="pt")
    return {
        'image': processed_image,
        'image_processed': processed_image,
    }
def newest(image_path="man.jpg"):

    print("image_path", image_path)
    # item = __getitem__(id, transformed_images, vocab_size_final)
    # Load the single image
    # if image_path is not None:
    # processed_image = transform(image_pa . th)
    image = Image.open(image_path)
    # else:/
        # image = item['original_image']

    # List of captions (5 captions)
    print("image", image)
    item = __getitem__(image)
    captions = evaluate_topk(transformer, item, 5)

    # Preprocess the image and captions
    inputs = processor(text=captions, images=image, return_tensors="pt", padding="max_length", max_length=35, truncation=True)

    # Get the model's output
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the logits per caption
    logits_per_caption = outputs.logits_per_text
    print(logits_per_caption, "logits_per_caption")

    # Find the caption index with the highest score
    best_caption_idx = torch.argmax(logits_per_caption).item()

    # Print the best matching caption
    print(f"The best matching caption is: {captions[best_caption_idx]}")
    best_caption = captions[best_caption_idx]
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
@app.post("/submit")
async def upload_image(file: UploadFile):
    print("Received request")
    
    try:
        # Read the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        print("image", image)
        
        # Convert RGBA to RGB if necessary
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
            else:
                background.paste(image, mask=image.split()[1])  # 1 is the alpha channel
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save it
        image.save("uploaded_image.jpg", "JPEG")
        # item = {
        #     'image': image,
        #     'image_processed': image,
        # }
        # processed_image = transform(image)
        captions = newest(image_path="uploaded_image.jpg")
        
        return {"message": "Image uploaded successfully"}
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}

# @app.post("/submit")
# async def submit(request, file: UploadFile = File(...)):  # Changed parameter definition
#     logger.debug("Endpoint hit")
#     print("Endpoint hit")
    
#     try:
#         form = await request.form()
#         logger.debug(f"Form data received: {form}")
        
#         if "file" not in form:
#             return {"error": "No file in form data"}
            
#         file = form["file"]
#         logger.debug(f"File received: {file.filename}")
        
#         # Read and process the uploaded image
#         image_content = await file.read()
#         image = Image.open(io.BytesIO(image_content))
        
#         # Save the image temporarily
#         temp_path = "temp_uploaded_image.jpg"
#         image.save(temp_path)
#         logger.debug(f"Saved image to {temp_path}")
#         # Call the newest() function with the temporary image path
#         # Modify the newest function call to capture the caption instead of showing the plot
#         captions = newest(image_path=temp_path)
        
#         # return {"caption": best_caption, "all_captions": captions}
#     except Exception as e:
#         return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "Image Caption Generator API"}