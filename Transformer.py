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
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.decoder = decoder
        self.text_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.pad_token = pad_token
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, image, caption):
        final_mask, padding_mask = self.generate_padding_mask(caption)
        # text_embeddings = self.text_encoder(caption)
        image_encoder_output = self.clip.get_image_features(image)
        # print(image_encoder_output, "image_encoder_output")
        text_embeddings = self.text_embedding(caption)
        
        sequence = torch.cat((image_encoder_output.unsqueeze(1), text_embeddings), dim=1)
        sequence = self.positional_encoding(sequence)
        dec_output = self.decoder.forward(sequence, final_mask)
        output = self.fc(dec_output[:,1:])
        return output
    
    def generate_padding_mask(self, caption):
        """
        Generate combined padding and causal mask for decoder self-attention.
        Args:
            caption: Input caption tensor of shape (batch_size, seq_len, vocab_size)
        Returns:
            Attention mask of shape (batch_size, seq_len, seq_len) where:
            - pad tokens are masked with 0
            - future tokens are masked with 0 (causal masking)
            - valid tokens are marked with 1
        """
        # print(caption.shape, "CAPTION", caption.squeeze(1).shape)
        padding_mask = (caption != self.pad_token).bool()  # [batch_size, seq_len]
        # print(padding_mask.shape, "PADDING MASK")
        padding_mask = torch.cat([torch.ones(padding_mask.shape[0], 1, device=padding_mask.device, dtype=torch.bool), padding_mask], dim=1)
        # Each item in the batch gets its own mask because:
        # 1. padding_mask is [batch_size, seq_len]
        # 2. When we do the unsqueeze operations, we maintain the batch dimension:
        padding_mask_self = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)
        final_mask = padding_mask_self
        # print(final_mask, "final_mask")
        return final_mask, padding_mask

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
        
        # self.projectbacktovocab = torch.nn.Linear(intermediate_attn_dim, tgt_vocab_size)

        self.norm1 = torch.nn.LayerNorm(input_dim)
        self.norm2 = torch.nn.LayerNorm(input_dim)
        self.norm3 = torch.nn.LayerNorm(input_dim)

    def forward(self, x, encoder_output, mask):
        embedding = x
        attn, prob = self.self_attn_layer.forward(embedding, embedding, embedding, mask)
        attn = self.dropout1(attn)
        x = self.norm1(attn + embedding)
        # attn, prob = self.cross_attn_layer.forward(query_input=x, key_input=encoder_output, value_input=encoder_output, mask=None)
        # x = self.norm2(x + attn)
        ff_output = self.FF_layer(x)
        ff_output = self.dropout2(ff_output)
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
     
     def generate_padding_mask(self, caption):
        """
        Generate combined padding and causal mask for decoder self-attention.
        Args:
            caption: Input caption tensor of shape (batch_size, seq_len, vocab_size)
        Returns:
            Attention mask of shape (batch_size, seq_len, seq_len) where:
            - pad tokens are masked with 0
            - future tokens are masked with 0 (causal masking)
            - valid tokens are marked with 1
        """
        # batch_size, seq_length, _ = caption.shape
        
        # Get padding mask by checking if the last index (pad token) is 1
        padding_mask = (caption.squeeze(1) != self.pad_token).bool()  # [batch_size, seq_len]
        # Each item in the batch gets its own mask because:
        # 1. padding_mask is [batch_size, seq_len]
        # 2. When we do the unsqueeze operations, we maintain the batch dimension:
        padding_mask_self = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)
        # Create final mask by combining padding and causal masks
        final_mask = padding_mask_self
        cross_attn_mask = padding_mask
        # Create final mask by combining padding and causal masks
        
        return final_mask, cross_attn_mask
     
   
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