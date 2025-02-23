import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import io
import transformers
from Transformer import Transformer, Decoder, DecoderLayer
from Attention import MultiHeadAttention
import matplotlib.pyplot as plt
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Set page config
st.set_page_config(
    page_title="Amirs Image Caption Generator",
    page_icon="🖼️",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .caption-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    CLIP = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    tokenizer = processor.tokenizer
    vocab_size = tokenizer.vocab_size
    tokenizer.add_special_tokens({"pad_token": "<<<PAD>>>"})
    vocab = tokenizer.get_vocab()
    tokenizer = processor.tokenizer
    vocab_size = tokenizer.vocab_size
    vocab_size_final = vocab_size + 1 #add 1 here
    reverse_vocab = {idx: token for token, idx in vocab.items()}

    # clip_text_model = CLIP.text_model

    # Model parameters
    d_model = 512
    text_dimension_embedding = 512
    image_encoder_output_dim = 768
    n_loops = 6
    num_heads = 8
    hidden_dim = 2048

    self_attn_layer = MultiHeadAttention(encoder_output_dim=d_model, decoder_dim=d_model, d_model=d_model, num_heads=num_heads)
    cross_attn_layer = MultiHeadAttention(encoder_output_dim=image_encoder_output_dim, decoder_dim=text_dimension_embedding, d_model=d_model, num_heads=num_heads)
    feed_forward = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_model))

    text_model = CLIP.text_model
    text_embedder = text_model.embeddings

    decoder_layer = DecoderLayer(
        input_dim=text_dimension_embedding,
        tgt_vocab_size=vocab_size_final,
        intermediate_attn_dim=d_model,
        n_loops=n_loops,
        feed_forward=feed_forward,
        self_attn_layer=self_attn_layer,
        cross_attn_layer=cross_attn_layer
    )

    decoder = Decoder(
        vocab_size_final,
        pad_token=tokenizer.pad_token_id,
        embedding_layer=text_embedder,
        layer=decoder_layer,
        n_loops=n_loops,
        d_model=d_model
    )

    transformer = Transformer(
        d_model=d_model,
        text_encoder=text_embedder,
        image_encoder=CLIP.vision_model,
        decoder=decoder,
        tgt_vocab_size=vocab_size_final,
        pad_token=tokenizer.pad_token_id
    )

    checkpoint = torch.load("checkpoints/best_model_sat.pt", map_location=torch.device("cpu"))
    transformer.load_state_dict(checkpoint['model_state_dict'])
    
    return transformer, processor, CLIP, reverse_vocab, tokenizer

def process_image(image):
    if image.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'RGBA':
            background.paste(image, mask=image.split()[3])
        else:
            background.paste(image, mask=image.split()[1])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def generate_captions(transformer, data, k, reverse_vocab):
    logging.info(f"Starting generate_captions with k={k}")
    transformer.eval()
    MAX_SEQ_LENGTH = 35
    START_TOKEN_ID = 49406
    BEAM_WIDTH = k
    
    captions = []  # Store captions in a list instead of yielding
    # Initialize beam with start token
    current_sequences = torch.full((BEAM_WIDTH, 1), START_TOKEN_ID)
    
    # Generate first token with beam search
    print(data, "data")
    output = transformer.forward(data['pixel_values'].repeat(BEAM_WIDTH, 1, 1, 1), current_sequences)
    logging.info('eyo2')
    output_probabilities = torch.softmax(output, dim=2)
    top_k_values, top_k_indices = torch.topk(output_probabilities[0, 0], k=BEAM_WIDTH)
    print('does it reach')
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
            output = transformer.forward(data['pixel_values'], current_sequence)
            output_probabilities = torch.softmax(output, dim=2)
            predicted_digit = torch.argmax(output_probabilities[0, pos])
            
            if pos < MAX_SEQ_LENGTH - 1:
                current_sequence = torch.cat((current_sequence, torch.tensor([predicted_digit.item()]).unsqueeze(0)), dim=1)
            
            predicted_indices = torch.cat((predicted_indices, predicted_digit.unsqueeze(0).unsqueeze(0)), dim=1)
        
        # Convert to text and yield immediately
        predicted_digits = [reverse_vocab[idx.item()] for idx in predicted_indices.squeeze(0)]
        cleaned_text = ' '.join(token.replace('</w>', '') for token in predicted_digits 
                              if token not in ['<|endoftext|>', '<<<PAD>>>'])
        logging.info(f"Generated caption: {cleaned_text}")
        captions.append(cleaned_text)  # Append instead of yield
    
    return captions  # Return the list of captions

def get_best_caption(image, captions, model, processor):
    print(captions, "xxxxx", image)

    image_inputs = processor(images=image, return_tensors="pt")
    text_inputs = processor.tokenizer(
        captions, 
        padding=True, 
        truncation=True, 
        max_length=77, 
        return_tensors="pt",
        add_special_tokens=True  # Ensure special tokens are added
    )
    
    # Adjust input_ids to account for the extra token
    input_ids = text_inputs["input_ids"]
    # Shift indices greater than or equal to the PAD token id up by 1
    pad_token_id = processor.tokenizer.pad_token_id
    input_ids[input_ids >= pad_token_id] += 1
    
    # Combine inputs
    inputs = {
        "input_ids": input_ids,
        "attention_mask": text_inputs["attention_mask"],
        "pixel_values": image_inputs["pixel_values"]
    }
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits_per_caption = outputs.logits_per_text
    best_caption_idx = torch.argmax(logits_per_caption).item()
    
    return captions[best_caption_idx]

def main():
    st.title("🖼️ Amir's Image Caption Generator")
    st.write("Upload an image and get AI-generated captions!")

    # Add GitHub link
    st.markdown("[View on GitHub](https://github.com/Amirjab21/MLXW4)", unsafe_allow_html=True)


    # Load models
    with st.spinner("Loading models..."):
        transformer, processor, clip_model, reverse_vocab, tokenizer = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Create columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Generated Captions")
            # Create a placeholder for captions
            captions_placeholder = st.empty()
            
            # Initialize an empty list to store captions
            captions = []
            
            # Process the image
            processed_image = process_image(image)
            data = processor(images=processed_image, return_tensors="pt")
            
            # Generate captions with streaming
            transformer.eval()
            MAX_SEQ_LENGTH = 35
            START_TOKEN_ID = 49406
            BEAM_WIDTH = 5
            
            # Initialize beam with start token
            current_sequences = torch.full((BEAM_WIDTH, 1), START_TOKEN_ID)
            
            # Generate first token with beam search
            output = transformer.forward(data['pixel_values'].repeat(BEAM_WIDTH, 1, 1, 1), current_sequences)
            output_probabilities = torch.softmax(output, dim=2)
            top_k_values, top_k_indices = torch.topk(output_probabilities[0, 0], k=BEAM_WIDTH)
            
            # Create progress bar
            progress_text = "Generating captions..."
            caption_progress = st.progress(0)
            
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
                    output = transformer.forward(data['pixel_values'], current_sequence)
                    output_probabilities = torch.softmax(output, dim=2)
                    predicted_digit = torch.argmax(output_probabilities[0, pos])
                    if predicted_digit.item() in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                        break
                    if pos < MAX_SEQ_LENGTH - 1:
                        current_sequence = torch.cat((current_sequence, torch.tensor([predicted_digit.item()]).unsqueeze(0)), dim=1)
                    
                    predicted_indices = torch.cat((predicted_indices, predicted_digit.unsqueeze(0).unsqueeze(0)), dim=1)
                
                # Convert to text and display immediately
                predicted_digits = [reverse_vocab[idx.item()] for idx in predicted_indices.squeeze(0)]
                cleaned_text = ' '.join(token.replace('</w>', '') for token in predicted_digits 
                                      if token not in ['<|endoftext|>', '<<<PAD>>>'])
                
                # Add the new caption to the list
                captions.append(cleaned_text)
                
                # Update the display with all captions so far
                caption_html = ""
                for idx, caption in enumerate(captions, 1):
                    caption_html += f'<div class="caption-box">{idx}. {caption}</div>'
                captions_placeholder.markdown(caption_html, unsafe_allow_html=True)
                
                # Update progress bar
                caption_progress.progress((beam_idx + 1) / BEAM_WIDTH)
            
            # Clear the progress bar when done
            caption_progress.empty()
            
            # Final update with all captions
            st.success("Caption generation complete!")

if __name__ == "__main__":
    main() 