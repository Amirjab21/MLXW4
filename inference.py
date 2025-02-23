import torch
import torch.nn as nn
from Transformer import Transformer, Decoder, DecoderLayer
from Attention import MultiHeadAttention
import transformers
from datasets import load_dataset
from torchvision import transforms
import matplotlib
matplotlib.use('TkAgg')  # Specify backend before importing pyplot
import matplotlib.pyplot as plt




dataset = load_dataset("nlphuji/flickr30k")
sample_size = 31000 if torch.cuda.is_available() else 250
dataset = dataset['test'].select(range(sample_size))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
])

def transform_images(examples):
    image = examples['image']
    processed_image = transform(image)
    # processed_image = np.array(make_square(image))
    return {
        'image': examples['image'],  
        'image_processed': processed_image, 
    }

transformed_images = dataset.map(transform_images)

device = torch.device("cpu")
CLIP = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

tokenizer = processor.tokenizer
vocab_size = tokenizer.vocab_size
print("vocab size", vocab_size)
tokenizer.add_special_tokens({"pad_token": "<<<PAD>>>"})
vocab = tokenizer.get_vocab()  # Returns dict of {token: index}
tokenizer = processor.tokenizer
vocab_size = tokenizer.vocab_size
vocab_size_final = vocab_size + 1 #add 1 here
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

def __getitem__(idx, dataset, vocab_size):
    dataset = dataset[idx]
    original_image = dataset['image']
    image = transform(dataset['image']) #height, width, channels
    captions = dataset['caption']

    random_caption_idx = torch.randint(0, len(captions), (1,)).item()
    selected_caption = captions[random_caption_idx]

    tokenized_caption = tokenizer(selected_caption, return_tensors="pt", padding="max_length", max_length=35, truncation=True)
    input_ids = tokenized_caption['input_ids']

    if vocab_size == 49409:
        eos_positions = (input_ids == tokenizer.eos_token_id).nonzero()
        if len(eos_positions) > 0:
            first_eos_pos = eos_positions[0][1]
            input_ids[0, first_eos_pos+1:] = 49408

    tokenized_caption['input_ids'] = input_ids
            
    return {
        'image': image,
        "original_image": original_image,
        'caption': tokenized_caption['input_ids'].squeeze(0),
        "original caption": selected_caption
    }

# data = __getitem__(0, transformed_images)
checkpoint = torch.load("checkpoints/best_model_sun.pt", map_location=torch.device("cpu"))
transformer.load_state_dict(checkpoint['model_state_dict'])

def evaluate(transformer, data, temperature):
    transformer.eval()
    MAX_SEQ_LENGTH = 35
    START_TOKEN_ID = 49406
    A_ID = 599
    # temperature =   # Add temperature parameter
    # PAD_TOKEN_ID = 49408
    current_sequence = torch.full((1, 1), START_TOKEN_ID)
    predicted_indices = torch.zeros((1, 1))
    
    # Generate one digit at a time
    for pos in range(MAX_SEQ_LENGTH):
        # print(data['image'].unsqueeze(0).shape, current_sequence.shape, "ok")
        output = transformer.forward(data['image'].unsqueeze(0), current_sequence)
         

        logits = output / temperature
        output_probabilities = torch.softmax(logits, dim=2)

        predicted_digits = torch.multinomial(output_probabilities[0, pos], num_samples=1)
        # print(predicted_digits, "pred")

        if pos < MAX_SEQ_LENGTH - 1:
            current_sequence = torch.cat((current_sequence, torch.tensor([predicted_digits.item()]).unsqueeze(0)), dim=1)
        # print(current_sequence)
        if pos == 0:
            predicted_indices[0, 0] = predicted_digits.item()
        else:
            predicted_indices = torch.cat((predicted_indices, predicted_digits.unsqueeze(0)), dim=1)
        


    predicted_digits = [reverse_vocab[idx.item()] for idx in predicted_indices.squeeze(0)]
    cleaned_text = ' '.join(token.replace('</w>', '') for token in predicted_digits 
                          if token not in ['<|endoftext|>', '<<<PAD>>>'])
    return cleaned_text
    # plt.imshow(data['original_image'])
    # plt.title(cleaned_text, wrap=True)
    # plt.axis('off')  # Hide axes
    # plt.show()
    
# for i in range(10):
# evaluate(transformer, __getitem__(20, transformed_images))
def evaluate_topk(transformer, data, k, temperature):
    transformer.eval()
    MAX_SEQ_LENGTH = 35
    START_TOKEN_ID = 49406
    # PAD_TOKEN_ID = 49408
    BEAM_WIDTH = k
    
    # Initialize beam with start token
    current_sequences = torch.full((BEAM_WIDTH, 1), START_TOKEN_ID)
    predicted_indices_list = []
    transformer.eval()
    # Generate first token with beam search
    # print(data['image'].unsqueeze(0).repeat(BEAM_WIDTH, 1, 1, 1), "dataimage" )
    output = transformer.forward(data['image'].unsqueeze(0).repeat(BEAM_WIDTH, 1, 1, 1), current_sequences)
    logits = output / temperature
    output_probabilities = torch.softmax(logits, dim=2)
    
    # Sample BEAM_WIDTH tokens using multinomial
    first_tokens = torch.multinomial(output_probabilities[0, 0], num_samples=BEAM_WIDTH)
    output_probabilities = torch.softmax(output, dim=2)
    top_k_values, top_k_indices = torch.topk(output_probabilities[0, 0], k=BEAM_WIDTH)
    
    # Create BEAM_WIDTH different sequences
    for beam_idx in range(BEAM_WIDTH):
        current_sequence = torch.full((1, 1), START_TOKEN_ID)
        predicted_indices = torch.zeros((1, 1))
        
        # Use the sampled first token
        first_token = first_tokens[beam_idx].item()
        current_sequence = torch.cat((current_sequence, torch.tensor([first_token]).unsqueeze(0)), dim=1)
        predicted_indices[0, 0] = first_token
        
        # Continue generating remaining tokens
        for pos in range(1, MAX_SEQ_LENGTH):
            output = transformer.forward(data['image'].unsqueeze(0), current_sequence)
            output_probabilities = torch.softmax(output, dim=2)
            predicted_digit = torch.argmax(output_probabilities[0, -1])
            if predicted_digit.item() in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                break
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
    # plt.imshow(data['original_image'])
    # # Join all captions with newlines and display them as the title
    # all_captions = '\n'.join(f"Caption {i+1}: {text}" for i, text in enumerate(predicted_indices_list))
    # plt.title(all_captions, wrap=True, pad=15)  # Added more padding to accommodate multiple lines
    # plt.axis('off')  # Hide axes
    # plt.show()

# evaluate_topk(transformer, __getitem__(27, transformed_images), 10)




def newest(id=108,image_path=None, temperature=2, k=3):
    model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    item = __getitem__(id, transformed_images, vocab_size_final)
    # Load the single image
    if image_path is not None:
        from PIL import Image
        image = Image.open(image_path)
    else:
        image = item['original_image']

    # List of captions (5 captions)
    captions = evaluate_topk(transformer, item, k, temperature)

    inputs = processor(text=captions, images=image, return_tensors="pt", padding="max_length", max_length=45, truncation=True)

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

    plt.imshow(image)
    # Join all captions with newlines and display them as the title
    # all_captions = '\n'.join(f"Caption {i+1}: {text}" for i, text in enumerate(captions))
    plt.title(best_caption, wrap=True, pad=15)  # Added more padding to accommodate multiple lines
    plt.axis('off')  # Hide axes
    plt.show()

# newest("twogirls.jpg")



def generate_padding_mask(caption, pad_token):
    """
    Generate combined padding and causal mask for decoder self-attention.
    The first token (CLS/image token) can be attended to by all tokens.
    All other tokens can only attend to previous tokens and the CLS token.
    """
    batch_size, seq_length = caption.shape
    total_length = seq_length + 1  # Add 1 for CLS token
    
    # Create causal mask (can attend to self and previous tokens)
    causal_mask = torch.triu(torch.ones(total_length, total_length), diagonal=1).bool()
    
    # Modify first column of causal mask to allow all tokens to attend to CLS token
    causal_mask[:, 0] = False
    
    # Move to correct device
    causal_mask = causal_mask.to(caption.device)
    
    # Create padding mask that includes CLS token
    padding_mask = torch.ones(batch_size, total_length, device=caption.device).bool()
    # Set the padding mask for all tokens after CLS
    padding_mask[:, 1:] = (caption != pad_token)
    false_indices = []
    for i in range(batch_size):
        # Find the first False position in this batch item's padding mask
        false_pos = (padding_mask[i] == False).nonzero()
        # If no padding token found, use the sequence length
        false_idx = false_pos[0].item() if len(false_pos) > 0 else total_length
        false_indices.append(false_idx)
    
    # Expand padding mask for broadcasting
    padding_mask = padding_mask.unsqueeze(1).expand(batch_size, total_length, total_length)
    
    # Create a copy and set all positions after each example's false_index to False
    cloned = torch.clone(padding_mask)
    for i in range(batch_size):
        cloned[i, false_indices[i]:, :] = False  # Set rows after false_index to False
        cloned[i, :, false_indices[i]:] = False  # Set columns after false_index to False
    # print(cloned, "cloned")
    # Combine masks: allow attention where causal_mask is False AND padding_mask is True
    final_mask = ~causal_mask & cloned
    
    return final_mask

def generate_caption(
    model,
    image: torch.Tensor,
    max_length: int = 30,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    min_length: int = 5,
    repetition_penalty: float = 1.2,
    length_penalty: float = 1.0
):
    """Generate a caption for the image using nucleus sampling."""
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # print(image.shape, "image")
        # Get image features
        image_features = model.clip.get_image_features(image['image'].unsqueeze(0))
        
        # Initialize caption with start token
        caption = torch.tensor([[model.tokenizer.bos_token_id]], device=device)
        
        # Keep track of generated tokens for repetition penalty
        generated_tokens = set()
        
        # Generate tokens one by one
        for _ in range(max_length):
            # Get text embeddings for current caption
            text_embeddings = model.text_embedding(caption)
            
            # Combine with image features
            sequence = torch.cat([
                image_features.unsqueeze(1),
                text_embeddings
            ], dim=1)
            
            # Add positional encoding
            sequence = model.positional_encoding(sequence)
            
            # Create causal mask
            seq_length = sequence.size(1)
            # causal_mask = torch.triu(
            #     torch.ones(seq_length, seq_length, device=device) * float('-inf'),
            #     diagonal=1
            # )
            # print(caption, "caption")
            mask = generate_padding_mask(caption, 49408)
            
            # Get decoder output
            decoder_output = model.decoder(sequence, mask=mask)
            
            # Get next token logits
            next_token_logits = model.fc(decoder_output[:, -1, :])
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            for token_id in generated_tokens:
                next_token_logits[0, token_id] /= repetition_penalty
            
            # Apply length penalty
            if caption.size(1) > min_length:
                next_token_logits = next_token_logits / (caption.size(1) ** length_penalty)
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1,
                    index=sorted_indices,
                    src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to generated tokens set
            generated_tokens.add(next_token.item())
            
            # Append next token to caption
            caption = torch.cat([caption, next_token], dim=1)
            
            # Stop if we generate end token or period
            if (next_token.item() == model.tokenizer.eos_token_id or 
                next_token.item() == model.tokenizer.encode('.')[0]):
                break
    
    # Decode caption and clean up
    caption = model.tokenizer.decode(caption[0], skip_special_tokens=True)
    print(caption, "caption")

    # Clean up the caption
    caption = caption.strip()
    if not caption.endswith('.'):
        caption += '.'
    
    # Remove any trailing fragments after the last period
    last_period = caption.rfind('.')
    if last_period > 0:
        caption = caption[:last_period + 1]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image['original_image'])
    plt.title(caption, wrap=True, pad=15)
    plt.axis('off')
    plt.show()
    
    return caption

if __name__ == "__main__":
    try:
        while True:  # Infinite loop
            # Generate random image ID from the dataset
            random_id = torch.randint(0, len(transformed_images), (1,)).item()
            newest(id=random_id, temperature=1.5)  # Test with random image
    except KeyboardInterrupt:
        print("\nTest stopped by user")