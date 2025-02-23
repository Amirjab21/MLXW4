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
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb_true = True if torch.cuda.is_available() else False

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
hidden_dim = 2048
text_dimension_embedding = 512
image_encoder_output_dim = 768
n_loops = 6
num_heads = 8

# self_attn_layer = Attention_Layer(d_model=d_model, num_heads=1)
self_attn_layer = MultiHeadAttention(encoder_output_dim=d_model, decoder_dim=d_model, d_model=d_model, num_heads=num_heads)
cross_attn_layer = MultiHeadAttention(encoder_output_dim=image_encoder_output_dim, decoder_dim=text_dimension_embedding, d_model=d_model, num_heads=num_heads)

feed_forward = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_model))

text_model = CLIP.text_model
text_embedder = text_model.embeddings

decoder_layer = DecoderLayer(input_dim=text_dimension_embedding, tgt_vocab_size=vocab_size, intermediate_attn_dim=d_model, n_loops=n_loops, feed_forward=feed_forward, self_attn_layer=self_attn_layer, cross_attn_layer=cross_attn_layer)



decoder = Decoder(vocab_size, pad_token=pad_token_id, embedding_layer=text_embedder, layer=decoder_layer, n_loops=n_loops,d_model=d_model)


transformer = Transformer(d_model=d_model, text_encoder=text_embedder, image_encoder=CLIP.vision_model, decoder=decoder, tgt_vocab_size=vocab_size, pad_token=pad_token_id)















batch_size = 256 if torch.cuda.is_available() else 1
learning_rate = 0.001
optimizer = torch.optim.Adam(transformer.parameters(), learning_rate)
# scheduler = get_linear_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=1000,
#             num_training_steps=total_steps
#         )
epochs = 20
criterion = nn.CrossEntropyLoss(ignore_index=49408)

dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6 if torch.cuda.is_available() else 0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=6 if torch.cuda.is_available() else 0, pin_memory=True)
transformer.to(device)

def create_position_weights(seq_length, device, first_n=5, weight_factor=5.0):
    """
    Creates weights where the first n words have weight_factor times more weight than the remaining words.
    Args:
        seq_length: Length of the sequence
        device: Device to create tensor on
        first_n: Number of initial positions to give higher weight to
        weight_factor: How much more weight to give to the first n positions
    """
    weights = torch.ones(seq_length, device=device)
    weights[:first_n] = weight_factor
    normalized_weights = weights / weights.mean()  # Normalize so average weight is 1
    return normalized_weights

# Modify the loss calculation in both train() and evaluate() functions
def weighted_cross_entropy(output, target, seq_length, ignore_index):
    # Reshape output and target
    output_flat = output.reshape(-1, output.size(-1))
    target_flat = target.reshape(-1)
    
    # Create position weights
    position_weights = create_position_weights(seq_length, output.device, first_n=7, weight_factor=4.0)
    # Repeat weights for each item in the batch
    weights_flat = position_weights.repeat(output.size(0))
    
    # Calculate cross entropy for each element
    loss = torch.nn.functional.cross_entropy(output_flat, target_flat, 
                          ignore_index=ignore_index,
                          reduction='none')
    
    # Apply position-based weights where target isn't padding
    mask = (target_flat != ignore_index)
    weighted_loss = (loss * weights_flat * mask).sum() / mask.sum()

    # print(f"\nFirst element unweighted loss: {loss[0].item():.4f}")
    # print(f"Batch weighted loss: {loss[0] * weights_flat[0] * mask[0] .item():.4f}")

    # print(f"\fifth element unweighted loss: {loss[6].item():.4f}")
    # print(f"Batch weighted loss: {loss[6] * weights_flat[6] * mask[6] .item():.4f}")

    
    return weighted_loss

def calculate_batch_bleu(predicted_batch, reference_batch):
    """
    Calculate BLEU scores for a batch of predictions against their references.
    Args:
        predicted_batch: List of lists of predicted tokens
        reference_batch: List of lists of reference tokens
    Returns:
        Average BLEU score for the batch
    """
    batch_bleu_scores = []
    smoother = SmoothingFunction()
    
    for pred_tokens, ref_tokens in zip(predicted_batch, reference_batch):
        # Remove special tokens and split into words
        pred_tokens = [token for token in pred_tokens if token not in ['<|endoftext|>', '<<<PAD>>>']]
        ref_tokens = [token for token in ref_tokens if token not in ['<|endoftext|>', '<<<PAD>>>']]
        
        # Calculate BLEU score for this prediction
        bleu_score = sentence_bleu([ref_tokens], pred_tokens, 
                                 smoothing_function=smoother.method1,
                                 weights=(0.25, 0.25, 0.25, 0.25))  # Using BLEU-4
        batch_bleu_scores.append(bleu_score)
    
    # Return average BLEU score for the batch
    return sum(batch_bleu_scores) / len(batch_bleu_scores) if batch_bleu_scores else 0.0

def evaluate(transformer, val_loader):
    transformer.eval()
    total_loss = 0
    total_bleu = 0
    num_batches = len(val_loader)
    
    # progress_bar = tqdm.tqdm(val_loader, desc="Evaluating")
    
    for _, batch in enumerate(val_loader):
            
        batch_loss = 0
        images, captions = batch['images'].to(device), batch['captions'].to(device)

        _, num_captions, seq_len = captions.shape
        
        images = images.repeat_interleave(num_captions, dim=0)
        captions = captions.view(-1, seq_len)
        
        optimizer.zero_grad()
        captionwithoutend = captions[:, :-1]
        output = transformer.forward(images, captionwithoutend)

        true_indices = captions[:, 1:]

        batch_loss = weighted_cross_entropy(output, true_indices, true_indices.size(1), 49408)
        
        # Calculate BLEU score for the batch
        output_probabilities = torch.softmax(output, dim=2)
        predicted_digits = torch.argmax(output_probabilities, dim=2)
        
        # Convert predictions and references to token lists
        pred_token_lists = []
        true_token_lists = []
        for i in range(len(predicted_digits)):
            pred_words = [reverse_vocab[idx.item()] for idx in predicted_digits[i][:]]
            true_words = [reverse_vocab[idx.item()] for idx in true_indices[i][:]]
            pred_token_lists.append(pred_words)
            true_token_lists.append(true_words)
        
        # Calculate BLEU score
        batch_bleu = calculate_batch_bleu(pred_token_lists, true_token_lists)
        total_bleu += batch_bleu
        total_loss += batch_loss.item()
        
    return total_loss / num_batches, total_bleu / num_batches

def train(log_wandb):
    if log_wandb == True:
        wandb.init(project="flickr30k", name=f"flickr30k{datetime.now()}", config={"batch_size": batch_size, "learning_rate": learning_rate, "epochs": epochs})

    best_val_loss = float('inf')
    best_val_bleu = 0.0
    
    total_loss = 0
    for epoch in range(epochs):
        transformer.train()
        epoch_loss = 0
        progress_bar = tqdm.tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            batch_loss = 0
            images, captions = batch['images'], batch['captions']
            _, num_captions, seq_len = captions.shape
            
            # Repeat each image for its captions
            images = images.repeat_interleave(num_captions, dim=0)
            captions = captions.view(-1, seq_len)
            
            images = images.to(device)
            captions = captions.to(device)
            
            optimizer.zero_grad()
            captionwithoutend = captions[:, :-1]
            output = transformer.forward(images, captionwithoutend)

            true_indices = captions[:, 1:]

            # batch_loss = weighted_cross_entropy(output, true_indices, true_indices.size(1), 49408)
            batch_loss = criterion(output.reshape(-1, output.size(-1)), true_indices.reshape(-1))
            if log_wandb == True:
                wandb.log({"batch_loss": batch_loss.item()})
            progress_bar.set_postfix({"batch_loss": batch_loss.item()})

            epoch_loss += batch_loss.item()
            total_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            
            if batch_idx == len(progress_bar) - 1:  # Print examples at the end of each epoch
                output_probabilities = torch.softmax(output, dim=2)
                predicted_digits = torch.argmax(output_probabilities, dim=2)
                for i in range(min(3, len(predicted_digits))):
                    pred_words = [reverse_vocab[idx.item()] for idx in predicted_digits[i][:]]
                    true_words = [reverse_vocab[idx.item()] for idx in true_indices[i][:]]
                    print(f"\nExample {i+1}:")
                    print(f"Predicted: {pred_words}")
                    print(f"True: {true_words}")
        
        # Evaluate on validation set
        val_loss, val_bleu = evaluate(transformer, val_loader)
        print(f"\nValidation metrics - Loss: {val_loss:.4f}, BLEU: {val_bleu:.4f}")
        
        # Save model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_bleu = val_bleu
            torch.save(
                transformer.state_dict(),
                f"best_model.pth"
            )
            print(f"Saved new best model with val_loss={val_loss:.4f}, val_bleu={val_bleu:.4f}")
        
        if log_wandb == True:
            wandb.log({
                "epoch_loss": epoch_loss / len(dataloader.dataset),
                "total_loss": total_loss / (epoch + 1),
                "val_loss": val_loss,
                "val_bleu": val_bleu
            })
        
        epoch_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / (epoch + 1):.4f}, Best BLEU: {best_val_bleu:.4f}")
        
train(wandb_true)
checkpoint_path = 'best_model.pt'
torch.save({ 'model_state_dict': transformer.state_dict()}, checkpoint_path)
artifact = wandb.Artifact('model-weights', type='model')
artifact.add_file(checkpoint_path)
wandb.log_artifact(artifact)



