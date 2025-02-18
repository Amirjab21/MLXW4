from datasets import load_dataset, concatenate_datasets
from torchvision import transforms
import numpy as np
from PIL import Image, ImageOps
import transformers
from dataset import Flickr30kDataset
from Transformer import Transformer, Decoder, DecoderLayer
from Attention import Attention_Layer, Cross_Attention_Layer, MultiHeadAttention
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import torch
import wandb
from datetime import datetime
torch.cuda.empty_cache()

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device, "device")

dataset = load_dataset("nlphuji/flickr30k")
sample_size = 15000 if torch.cuda.is_available() else 10
dataset = dataset['test'].select(range(sample_size))

def make_square(image):
    size = max(image.size)
    new_image = ImageOps.pad(image, (size, size), color='white')
    new_image = new_image.resize((224, 224))

    return new_image

def transform_images(examples):
    image = examples['image']
    processed_image = np.array(make_square(image))
    return {
        'image': examples['image'],  
        'image_processed': processed_image, 
    }

transformed_images = dataset.map(transform_images)

CLIP = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

tokenizer = processor.tokenizer

vocab = tokenizer.get_vocab()  # Returns dict of {token: index}
vocab_size = tokenizer.vocab_size
reverse_vocab = {idx: token for token, idx in vocab.items()}

clip_text_model = CLIP.text_model

dataset = Flickr30kDataset(transformed_images, tokenizer)



d_model = 512
text_dimension_embedding = 512
image_encoder_output_dim = 768
n_loops = 6

# self_attn_layer = Attention_Layer(d_model=d_model, num_heads=1)
self_attn_layer = MultiHeadAttention(encoder_output_dim=d_model, decoder_dim=d_model, d_model=d_model, num_heads=8)
cross_attn_layer = MultiHeadAttention(encoder_output_dim=image_encoder_output_dim, decoder_dim=text_dimension_embedding, d_model=d_model, num_heads=8)

feed_forward = nn.Sequential(nn.Linear(d_model, 2048), nn.ReLU(), nn.Linear(2048, d_model))

text_model = CLIP.text_model
text_embedder = text_model.embeddings

decoder_layer = DecoderLayer(input_dim=text_dimension_embedding, tgt_vocab_size=vocab_size, intermediate_attn_dim=d_model, n_loops=n_loops, feed_forward=feed_forward, self_attn_layer=self_attn_layer, cross_attn_layer=cross_attn_layer)



decoder = Decoder(vocab_size, pad_token=tokenizer.pad_token_id, embedding_layer=text_embedder, layer=decoder_layer, n_loops=n_loops,d_model=d_model)


transformer = Transformer(d_model=d_model, text_encoder=text_embedder, image_encoder=CLIP.vision_model, decoder=decoder, tgt_vocab_size=vocab_size)















batch_size = 32 if torch.cuda.is_available() else 2
# print(batch_size, "batch_size")
learning_rate = 0.001
optimizer = torch.optim.Adam(transformer.parameters(), learning_rate)
epochs = 10
criterion = nn.CrossEntropyLoss()

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4 if torch.cuda.is_available() else 0, pin_memory=True)

transformer.to(device)

def train():
    wandb.init(project="flickr30k", name=f"flickr30k{datetime.now()}", config={"batch_size": batch_size, "learning_rate": learning_rate, "epochs": epochs})
    transformer.train()
    total_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm.tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            
            batch_loss = 0
            image, caption = batch['image'].to(device), batch['caption'].to(device)
            # print(caption[0], 'caption.shape')
            
            optimizer.zero_grad()
            captionwithoutend = caption[:, :-1]
            # print(captionwithoutend.shape ,"captionwithoutend shape")
            output = transformer.forward(image, captionwithoutend)


            # Apply softmax across the vocabulary dimension (dim=2)

            # print(predicted_digits[0], "predicted digits")
            true_indices = caption[:, 1:]
            # true_indices = torch.argmax(caption[:,:, 1:], dim=1)  # Skip first position (START token)
            # print(true_indices.shape, "true indices")
            # print(predicted_digits.shape, "predicted digits")
            # end_token = torch.full((label.size(0), 1), token_to_idx['<END>'], device=device)
            # true_indices =   # Add END token
            # print(true_indices[0])

            # predicted_digits = [[reverse_vocab[idx.item()] for idx in pred] for pred in predicted_digits.squeeze(1)]
            # true_digits = [[reverse_vocab[idx.item()] for idx in true] for true in true_indices.squeeze(1)]
            
            if batch_idx % 5 == 0:  # Print every 5 batches
                output_probabilities = torch.softmax(output, dim=2)
                predicted_digits = torch.argmax(output_probabilities, dim=2)  # Shape: [batch_size, seq_length, 1
                # true_indices = torch.argmax(caption[:, 1:], dim=1)  # Skip first position (START token)
                for j in range(min(1, len(predicted_digits))):  # Show first 3 examples
                    # print(f"Predicted: {predicted_digits[j]} | True: {true_indices[j]}")
                    pred_words = [reverse_vocab[idx.item()] for idx in predicted_digits[j][:6]]
                    true_words = [reverse_vocab[idx.item()] for idx in true_indices[j][:6]]
                    print(f"Predicted: {pred_words} | True: {true_words}")
            
            # print(f"Output shape: {output.shape}, dtype: {output.dtype}")
            # print(f"True indices shape: {true_indices.shape}, dtype: {true_indices.dtype}")
            # print(output.shape, "output.shape")
            # print(true_indices.shape, "true_indices.shape")
            # for i in range(5):  # For the 4 digits
            #     batch_loss += criterion(output[:, i, :], true_indices[:, i])
            # batch_loss = criterion(output.view(-1, output.size(-1)), true_indices.squeeze(1).view(-1)) #ALTERNATIVE LOSS FUNCTION
            # print(output[0][2])
            output = output.reshape(-1, output.size(-1))  # Changed view to reshape
            true_indices = true_indices.reshape(-1).long()  # Changed view to reshape and removed squeeze

            # print(f"After reshape:")
            # print(f"Output shape: {output.shape}, dtype: {output.dtype}")
            # print(f"True indices shape: {true_indices.shape}, dtype: {true_indices.dtype}")
            
            batch_loss = criterion(output, true_indices)
            # batch_loss = torch.tensor(100.0, device=device, requires_grad=True)
            # batch_loss = 100
            wandb.log({"batch_loss": batch_loss.item() })
            progress_bar.set_postfix({"batch_loss": batch_loss.item() })

            epoch_loss += batch_loss.item()
            total_loss += batch_loss.item()
            batch_loss.backward()
            # optimizer.step()
        wandb.log({"epoch_loss": epoch_loss / len(dataloader.dataset)})
        wandb.log({"total_loss": total_loss / (epoch + 1)})
        # validate(transformer,100)
        epoch_loss = epoch_loss / len(dataloader.dataset)
        # total_loss += epoch_loss
        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        print(f"Total {epoch + 1}/{epochs}, Loss: {total_loss / (epoch + 1):.4f}")

        
train()


