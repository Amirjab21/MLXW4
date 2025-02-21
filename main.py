from datasets import load_dataset, concatenate_datasets
from torchvision import transforms
import numpy as np
from PIL import Image, ImageOps
import transformers
from dataset import Flickr30kDataset
from Transformer import Transformer, Decoder, DecoderLayer
from Attention import MultiHeadAttention
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import torch
import wandb
from datetime import datetime
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("nlphuji/flickr30k")
sample_size = 31000 if torch.cuda.is_available() else 2
dataset = dataset['test'].select(range(sample_size))
# .select(range(sample_size))
# .select(range(sample_size))


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

CLIP = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

tokenizer = processor.tokenizer
tokenizer.add_special_tokens({"pad_token": "<<<PAD>>>"})
pad_token_id = 49408
vocab = tokenizer.get_vocab()  # Returns dict of {token: index}
vocab_size = tokenizer.vocab_size + 1
reverse_vocab = {idx: token for token, idx in vocab.items()}

clip_text_model = CLIP.text_model

dataset = Flickr30kDataset(transformed_images, tokenizer)

train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, 
    [train_size, val_size]
)



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

decoder_layer = DecoderLayer(input_dim=text_dimension_embedding, tgt_vocab_size=vocab_size, intermediate_attn_dim=d_model, n_loops=n_loops, feed_forward=feed_forward, self_attn_layer=self_attn_layer, cross_attn_layer=cross_attn_layer)



decoder = Decoder(vocab_size, pad_token=pad_token_id, embedding_layer=text_embedder, layer=decoder_layer, n_loops=n_loops,d_model=d_model)


transformer = Transformer(d_model=d_model, text_encoder=text_embedder, image_encoder=CLIP.vision_model, decoder=decoder, tgt_vocab_size=vocab_size, pad_token=pad_token_id)















batch_size = 48 if torch.cuda.is_available() else 1
learning_rate = 0.0005
optimizer = torch.optim.Adam(transformer.parameters(), learning_rate)
# scheduler = get_linear_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=1000,
#             num_training_steps=total_steps
#         )
epochs = 10
criterion = nn.CrossEntropyLoss(ignore_index=49408)

dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6 if torch.cuda.is_available() else 0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=6 if torch.cuda.is_available() else 0, pin_memory=True)
transformer.to(device)

def create_position_weights(seq_length, device, alpha=0.1):
    """
    Creates weights that increase along the sequence.
    alpha controls how quickly the weights increase (smaller alpha = slower increase)
    """
    # positions = torch.arange(seq_length, device=device)
    # weights = torch.exp(alpha * positions)
    positions = torch.arange(seq_length - 1, -1, -1, device=device)  # Reversed positions
    weights = torch.exp(alpha * positions)
    normalized_weights = weights / weights.mean()  # Normalize so average weight is 1
    return normalized_weights

def debug_transformer(transformer, image_batch, caption_batch, criterion):
    """
    Debug function to inspect transformer's behavior on individual examples
    """
    transformer.train()
    batch_size = image_batch.size(0)
    
    # 1. Process input
    print("\n=== Input Processing ===")
    print(f"Image batch shape: {image_batch.shape}")
    print(f"Caption batch shape: {caption_batch.shape}")
    
    # 2. Get transformer outputs with attention weights
    captionwithoutend = caption_batch[:, :-1]
    true_indices = caption_batch[:, 1:]
    
    # Enable attention weight collection
    transformer.decoder.layer.self_attn_layer.store_attention_weights = True
    transformer.decoder.layer.cross_attn_layer.store_attention_weights = True
    
    output = transformer(image_batch, captionwithoutend)
    
    # 3. Print attention patterns
    print("\n=== Attention Patterns ===")
    self_attention_weights = transformer.decoder.layer.self_attn_layer.last_attention_weights
    cross_attention_weights = transformer.decoder.layer.cross_attn_layer.last_attention_weights
    
    print(f"Self-attention shape: {self_attention_weights.shape}")
    print(f"Cross-attention shape: {cross_attention_weights.shape}")
    
    # 4. Print predictions vs actual
    print("\n=== Predictions vs Actual ===")
    output_probabilities = torch.softmax(output, dim=2)
    predicted_indices = torch.argmax(output_probabilities, dim=2)
    
    for i in range(batch_size):
        print(f"\nExample {i+1}:")
        pred_words = [reverse_vocab[idx.item()] for idx in predicted_indices[i]]
        true_words = [reverse_vocab[idx.item()] for idx in true_indices[i]]
        print(f"Predicted: {' '.join(pred_words)}")
        print(f"True: {' '.join(true_words)}")
        
        # Calculate per-token loss
        example_output = output[i].unsqueeze(0)
        example_target = true_indices[i].unsqueeze(0)
        token_losses = torch.nn.functional.cross_entropy(
            example_output.view(-1, output.size(-1)),
            example_target.view(-1),
            ignore_index=49408,
            reduction='none'
        )
        print(f"Token-wise losses: {token_losses}")
        
        # Print attention visualization for first head
        print("\nSelf-attention first head (first 5 tokens):")
        print(self_attention_weights[i, 0, :5, :5])
        print("\nCross-attention first head (first 5 tokens):")
        print(cross_attention_weights[i, 0, :5, :5])
    
    return output, self_attention_weights, cross_attention_weights

# Modify the loss calculation in both train() and evaluate() functions
def weighted_cross_entropy(output, target, seq_length, ignore_index):
    # Reshape output and target
    output_flat = output.reshape(-1, output.size(-1))
    target_flat = target.reshape(-1)
    
    # Create position weights
    position_weights = create_position_weights(seq_length, output.device, alpha=0.1)
    # Repeat weights for each item in the batch
    weights_flat = position_weights.repeat(output.size(0))
    
    # Calculate cross entropy for each element
    loss = torch.nn.functional.cross_entropy(output_flat, target_flat, 
                          ignore_index=ignore_index,
                          reduction='none')
    
    # Apply position-based weights where target isn't padding
    mask = (target_flat != ignore_index)
    weighted_loss = (loss * weights_flat * mask).sum() / mask.sum()

    
    return weighted_loss

def evaluate(transformer, val_loader):
    transformer.eval()
    total_loss = 0
    num_batches = len(val_loader)
    
    # progress_bar = tqdm.tqdm(val_loader, desc="Evaluating")
    
    for _, batch in enumerate(val_loader):
            
        batch_loss = 0
        image, caption = batch['image'].to(device), batch['caption'].to(device)
        
        optimizer.zero_grad()
        captionwithoutend = caption[:, :-1]
        output = transformer.forward(image, captionwithoutend)

        true_indices = caption[:, 1:]


        batch_loss = weighted_cross_entropy(output, true_indices, true_indices.size(1), 49408)
        
        total_loss += batch_loss.item() 
        
        # progress_bar.set_postfix({"val_loss": total_loss / (num_batches)})
    
    return total_loss / num_batches

def train(log_wandb=True):
    if log_wandb:
        wandb.init(project="flickr30k", name=f"flickr30k{datetime.now()}", config={"batch_size": batch_size, "learning_rate": learning_rate, "epochs": epochs})

    best_val_loss = float('inf')
    
    total_loss = 0
    for epoch in range(epochs):
        transformer.train()
        epoch_loss = 0
        progress_bar = tqdm.tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            
            batch_loss = 0
            image, caption = batch['image'].to(device), batch['caption'].to(device)
            
            optimizer.zero_grad()
            captionwithoutend = caption[:, :-1]
            output = transformer.forward(image, captionwithoutend)

            true_indices = caption[:, 1:]
  
            

            

            # output = output.reshape(-1, output.size(-1))  # Changed view to reshape
            # true_indices = true_indices.reshape(-1)  # Changed view to reshape and removed squeeze

            batch_loss = weighted_cross_entropy(output, true_indices, true_indices.size(1), 49408)
            # batch_loss = criterion(output.reshape(-1, output.size(-1)), true_indices.reshape(-1))
            if log_wandb:
                wandb.log({"batch_loss": batch_loss.item() })
            progress_bar.set_postfix({"batch_loss": batch_loss.item() })

            epoch_loss += batch_loss.item()
            total_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            if batch_idx == len(progress_bar) - 1:  # Print every 20 batches
                output_probabilities = torch.softmax(output, dim=2)
                predicted_digits = torch.argmax(output_probabilities, dim=2)
                for i in range(min(3, len(predicted_digits))):
                    pred_words = [reverse_vocab[idx.item()] for idx in predicted_digits[i][:]]
                    true_words = [reverse_vocab[idx.item()] for idx in true_indices[i][:]]
                    print(f"Example {i+1}:")
                    print(f"Predicted: {pred_words}")
                    print(f"True: {true_words}\n")
        val_loss = evaluate(transformer, val_loader)
        if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    transformer.state_dict(),
                    f"best_model.pth"
                )
                print(f"Saved new best model with val_loss={val_loss:.4f}")
        if log_wandb:
            wandb.log({"epoch_loss": epoch_loss / len(dataloader.dataset), "total_loss": total_loss / (epoch + 1), "val_loss": val_loss})
        epoch_loss = epoch_loss / len(dataloader.dataset)
        print(f"Total {epoch + 1}/{epochs}, Loss: {total_loss / (epoch + 1):.4f}")
        

        
train(True)
checkpoint_path = 'best_model.pt'
torch.save({ 'model_state_dict': transformer.state_dict()}, checkpoint_path)
artifact = wandb.Artifact('model-weights', type='model')
artifact.add_file(checkpoint_path)
wandb.log_artifact(artifact)


