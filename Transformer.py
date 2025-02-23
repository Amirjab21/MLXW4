from torch import nn
import torch
import copy
from transformers import CLIPModel, CLIPTokenizer
import math
class Transformer(nn.Module):
    def __init__(self, d_model, text_encoder, image_encoder, decoder, tgt_vocab_size, dropout=0.1, pad_token=None):
        super(Transformer, self).__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.clip.parameters():
            param.requires_grad = False

        self.decoder = decoder
        self.text_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.pad_token = pad_token
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, image, caption):
        # Get image features and reshape for concatenation
        image_features = self.clip.get_image_features(image).unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        # Get text embeddings
        text_embeddings = self.text_embedding(caption)  # [batch_size, seq_len, d_model]
        
        # Concatenate image features with text embeddings
        sequence = torch.cat((image_features, text_embeddings), dim=1)
        
        # Add positional encoding
        sequence = self.positional_encoding(sequence)
        
        # Generate mask
        mask = self.generate_padding_mask(caption)
        
        # Pass through decoder
        dec_output = self.decoder(sequence, mask)
        
        # Remove CLS token and project to vocabulary
        output = self.fc(dec_output[:, 1:])
        return output
    
    def generate_padding_mask(self, caption):
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
        padding_mask[:, 1:] = (caption != self.pad_token)
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

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class DecoderLayer(nn.Module):
    def __init__(self, input_dim, tgt_vocab_size, intermediate_attn_dim, n_loops, feed_forward, self_attn_layer, cross_attn_layer):
        super(DecoderLayer, self).__init__()
        self.self_attn_layer = self_attn_layer
        self.cross_attn_layer = cross_attn_layer
        self.FF_layer = feed_forward
        self.tgt_vocab_size = tgt_vocab_size
        self.input_dim = input_dim
        self.n_loops = n_loops
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        
        self.norm1 = torch.nn.LayerNorm(input_dim)
        self.norm2 = torch.nn.LayerNorm(input_dim)
        self.norm3 = torch.nn.LayerNorm(input_dim)

    def forward(self, x, encoder_output, mask):
        # Self attention
        attn1, _ = self.self_attn_layer(x, x, x, mask)
        attn1 = self.dropout1(attn1)
        x = self.norm1(attn1 + x)
        
        # Cross attention with image features
        # attn2, _ = self.cross_attn_layer(x, encoder_output, encoder_output)
        # attn2 = self.dropout2(attn2)
        # x = self.norm2(attn2 + x)
        
        # Feed forward
        ff_output = self.FF_layer(x)
        ff_output = self.dropout3(ff_output)
        x = self.norm3(x + ff_output)
        
        return x

class Decoder(nn.Module):
     def __init__(self, tgt_vocab_size, pad_token, embedding_layer, layer, n_loops, d_model):
        super(Decoder, self).__init__()
        self.embedding_layer = embedding_layer #convert token IDs to embeddings
        self.pad_token = pad_token
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.layers = clones(layer, n_loops)

        # self.projectbacktovocab = torch.nn.Linear(512, tgt_vocab_size)


     def forward(self, sequence, mask):
        # mask = self.generate_padding_mask(sequence)
        # mask = True
        # x = self.embedding_layer.forward(sequence).squeeze(1)
        x = sequence
        for layer in self.layers:
            x = layer(sequence, None, mask)
        # x = self.projectbacktovocab(x)
        x = self.norm1(x)
        return x
     

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)