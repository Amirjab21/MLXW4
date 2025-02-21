import torch
from datasets import load_dataset
from torchvision import transforms
import transformers
from dataset import Flickr30kDataset
from Transformer import Transformer, Decoder, DecoderLayer
from Attention import MultiHeadAttention
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and transform a small subset of the dataset
dataset = load_dataset("nlphuji/flickr30k")
dataset = dataset['test'].select(range(2))  # Just 2 examples

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
    return {
        'image': examples['image'],
        'image_processed': processed_image,
    }

transformed_images = dataset.map(transform_images)

# Setup model components
CLIP = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

tokenizer = processor.tokenizer
tokenizer.add_special_tokens({"pad_token": "<<<PAD>>>"})
vocab = tokenizer.get_vocab()
vocab_size = tokenizer.vocab_size + 1
reverse_vocab = {idx: token for token, idx in vocab.items()}

# Create dataset
dataset = Flickr30kDataset(transformed_images, tokenizer)

# Model parameters
d_model = 512
text_dimension_embedding = 512
image_encoder_output_dim = 768
n_loops = 6
num_heads = 8

# Create model components
self_attn_layer = MultiHeadAttention(
    encoder_output_dim=d_model,
    decoder_dim=d_model,
    d_model=d_model,
    num_heads=num_heads
)
cross_attn_layer = MultiHeadAttention(
    encoder_output_dim=image_encoder_output_dim,
    decoder_dim=text_dimension_embedding,
    d_model=d_model,
    num_heads=num_heads
)

feed_forward = nn.Sequential(
    nn.Linear(d_model, 2048),
    nn.ReLU(),
    nn.Linear(2048, d_model)
)

text_embedder = CLIP.text_model.embeddings

decoder_layer = DecoderLayer(
    input_dim=text_dimension_embedding,
    tgt_vocab_size=vocab_size,
    intermediate_attn_dim=d_model,
    n_loops=n_loops,
    feed_forward=feed_forward,
    self_attn_layer=self_attn_layer,
    cross_attn_layer=cross_attn_layer
)

decoder = Decoder(
    vocab_size,
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
    tgt_vocab_size=vocab_size,
    pad_token=tokenizer.pad_token_id
)

def plot_attention_weights(attention_weights, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights.cpu().numpy(), cmap='viridis')
    plt.title(title)
    plt.show()

def debug_example(transformer, example_idx, dataset):
    print(f"\n=== Debugging Example {example_idx} ===")
    
    # Get example data
    example = dataset[example_idx]
    image = example['image'].unsqueeze(0).to(device)
    caption = example['caption'].unsqueeze(0).to(device)
    
    # Enable attention weight storage for all layers
    for layer in transformer.decoder.layers:
        layer.self_attn_layer.store_attention_weights = True
        # layer.cross_attn_layer.store_attention_weights = True
    
    # Forward pass
    print("\nInput shapes:")
    print(f"Image shape: {image.shape}")
    print(f"Caption shape: {caption.shape}")
    
    captionwithoutend = caption[:, :-1]
    true_indices = caption[:, 1:]
    
    output = transformer(image, captionwithoutend)
    
    # Print predictions vs actual
    print("\nPredictions vs Actual:")
    output_probabilities = torch.softmax(output, dim=2)
    predicted_indices = torch.argmax(output_probabilities, dim=2)
    
    pred_words = [reverse_vocab[idx.item()] for idx in predicted_indices[0]]
    true_words = [reverse_vocab[idx.item()] for idx in true_indices[0]]
    
    print("Predicted:", ' '.join(pred_words))
    print("True:", ' '.join(true_words))
    
    # Calculate and print per-token loss
    token_losses = torch.nn.functional.cross_entropy(
        output.view(-1, output.size(-1)),
        true_indices.view(-1),
        ignore_index=tokenizer.pad_token_id,
        reduction='none'
    )
    token_losses = token_losses.view(true_indices.shape)
    
    print("\nPer-token losses:")
    for i, (word, loss) in enumerate(zip(true_words, token_losses[0])):
        print(f"{word}: {loss.item():.4f}")
    
    # Plot attention patterns for each layer
    print("\nPlotting attention patterns...")
    
    for layer_idx, layer in enumerate(transformer.decoder.layers):
        # Self-attention weights
        self_attn_weights = layer.self_attn_layer.last_attention_weights
        print(f"\nLayer {layer_idx + 1} self-attention shape: {self_attn_weights.shape}")
        plot_attention_weights(
            self_attn_weights[0, 0],  # First head, first example
            f"Layer {layer_idx + 1} Self-attention weights (first head)"
        )
        
        # Cross-attention weights
        # cross_attn_weights = layer.cross_attn_layer.last_attention_weights
        # print(f"Layer {layer_idx + 1} cross-attention shape: {cross_attn_weights.shape}")
        # plot_attention_weights(
        #     cross_attn_weights[0, 0],  # First head, first example
        #     f"Layer {layer_idx + 1} Cross-attention weitghts (first head)"
        # )

# Debug both examples
transformer = transformer.to(device)
transformer.eval()

with torch.no_grad():
    for i in range(2):
        debug_example(transformer, i, dataset)