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
from torch.utils.data import DataLoader
from dataset import Flickr30kDataset


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
print("first vocab size", vocab_size)
tokenizer.add_special_tokens({"pad_token": "<<<PAD>>>"})
vocab = tokenizer.get_vocab()  # Returns dict of {token: index}
tokenizer = processor.tokenizer
vocab_size = tokenizer.vocab_size
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

decoder_layer = DecoderLayer(input_dim=text_dimension_embedding, tgt_vocab_size=vocab_size + 1, intermediate_attn_dim=d_model, n_loops=n_loops, feed_forward=feed_forward, self_attn_layer=self_attn_layer, cross_attn_layer=cross_attn_layer)



decoder = Decoder(vocab_size + 1, pad_token=tokenizer.pad_token_id, embedding_layer=text_embedder, layer=decoder_layer, n_loops=n_loops,d_model=d_model)
transformer = Transformer(d_model=d_model, text_encoder=text_embedder, image_encoder=CLIP.vision_model, decoder=decoder, tgt_vocab_size=vocab_size + 1, pad_token=tokenizer.pad_token_id)

dataset = Flickr30kDataset(transformed_images, tokenizer)

# data = __getitem__(0, transformed_images)
checkpoint = torch.load("checkpoints/best_model_2.pt", map_location=torch.device("cpu"))
transformer.load_state_dict(checkpoint['model_state_dict'])

def evaluate(transformer, data):
    transformer.eval()
    MAX_SEQ_LENGTH = 35
    START_TOKEN_ID = 49406
    temperature = 0.7  # Add temperature parameter
    current_sequence = torch.full((1, 1), START_TOKEN_ID)
    predicted_indices = torch.zeros((1, 1))
    
    # Generate one digit at a time
    for pos in range(MAX_SEQ_LENGTH):
        output = transformer.forward(data['image'].unsqueeze(0), current_sequence)

        # Apply temperature to logits
        logits = output[0, pos] / temperature
        output_probabilities = torch.softmax(logits, dim=0)
        
        if pos == 1:
            top_3_values, top_3_indices = torch.topk(output_probabilities, k=3)
            print("\nTop 3 predictions for position 0:")
            for i, (idx, prob) in enumerate(zip(top_3_indices, top_3_values)):
                token = reverse_vocab[idx.item()]
                print(f"{i+1}. Token: {token}, Probability: {prob.item():.4f}")

        # Sample from the distribution instead of argmax
        predicted_digits = torch.multinomial(output_probabilities, num_samples=1)

        if pos < MAX_SEQ_LENGTH - 1:
            current_sequence = torch.cat((current_sequence, predicted_digits.unsqueeze(0)), dim=1)

        if pos == 0:
            predicted_indices[0, 0] = predicted_digits.item()
        else:
            predicted_indices = torch.cat((predicted_indices, predicted_digits.unsqueeze(0).unsqueeze(0)), dim=1)

    predicted_digits = [reverse_vocab[idx.item()] for idx in predicted_indices.squeeze(0)]
    cleaned_text = ' '.join(token.replace('</w>', '') for token in predicted_digits 
                          if token not in ['<|endoftext|>', '<<<PAD>>>'])
    return cleaned_text

batch_size = 1
val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6 if torch.cuda.is_available() else 0, pin_memory=True)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001)

def evaluate_new(transformer, val_loader):
    transformer.eval()
    total_loss = 0
    num_batches = len(val_loader)
    temperature = 0.7  # Add temperature parameter
    
    for _, batch in enumerate(val_loader):
        batch_loss = 0
        image, caption = batch['image'].to(device), batch['caption'].to(device)
        
        optimizer.zero_grad()
        captionwithoutend = caption[:, :-1]
        output = transformer.forward(image, captionwithoutend)

        true_indices = caption[:, 1:]

        # Apply temperature and sample
        logits = output / temperature
        output_probabilities = torch.softmax(logits, dim=2)
        # Sample from the distribution
        predicted_digits = torch.multinomial(output_probabilities.reshape(-1, output_probabilities.size(-1)), num_samples=1).reshape(output_probabilities.size(0), -1)
        
        for i in range(min(3, len(predicted_digits))):
            pred_words = [reverse_vocab[idx.item()] for idx in predicted_digits[i][:]]
            true_words = [reverse_vocab[idx.item()] for idx in true_indices[i][:]]
            print(f"Example {i+1}:")
            print(f"Predicted: {pred_words}")
            print(f"True: {true_words}\n")
    
    return total_loss / num_batches


evaluate_new(transformer, val_loader)