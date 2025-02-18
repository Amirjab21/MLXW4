from torch import nn
import torch
import copy

class Transformer(nn.Module):
    def __init__(self, d_model, text_encoder, image_encoder, decoder, tgt_vocab_size):
        super(Transformer, self).__init__()

        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.decoder = decoder

        self.fc = nn.Linear(tgt_vocab_size, tgt_vocab_size)

    def forward(self, image, caption):
        # text_encoder_output = self.text_encoder.forward(caption)
        image_encoder_output = self.image_encoder.forward(image)
        dec_output = self.decoder.forward(caption, image_encoder_output.last_hidden_state)
        output = self.fc(dec_output)
        return output

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

        self.projectbacktovocab = torch.nn.Linear(intermediate_attn_dim, tgt_vocab_size)

        self.norm1 = torch.nn.LayerNorm(input_dim)
        self.norm2 = torch.nn.LayerNorm(input_dim)
        self.norm3 = torch.nn.LayerNorm(input_dim)

    def forward(self, x, encoder_output, mask):
        embedding = x
        attn, prob = self.self_attn_layer.forward(embedding, embedding, embedding, mask)
        x = self.norm1(attn + embedding)
        attn, prob = self.cross_attn_layer.forward(query_input=x, key_input=encoder_output, value_input=encoder_output)
        x = self.norm2(x + attn)
        ff_output = self.FF_layer(x)
        x = self.norm3(x + ff_output)
        return x

class Decoder(nn.Module):
     def __init__(self, tgt_vocab_size, pad_token, embedding_layer, layer, n_loops):
        super(Decoder, self).__init__()
        self.embedding_layer = embedding_layer #convert token IDs to embeddings
        self.pad_token = pad_token
        self.norm1 = torch.nn.LayerNorm(tgt_vocab_size)
        self.layers = clones(layer, n_loops)

        self.projectbacktovocab = torch.nn.Linear(512, tgt_vocab_size)


     def forward(self, x, encoder_output):
        # mask = self.generate_padding_mask(x)
        mask = True
        x = self.embedding_layer.forward(x).squeeze(1)
        for layer in self.layers:
            x = layer(x, encoder_output, mask)
        x = self.projectbacktovocab(x)
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
            padding_mask = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)
            # Create final mask by combining padding and causal masks
            final_mask = padding_mask
            
            return final_mask
     
   