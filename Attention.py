import torch
import math
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, encoder_output_dim, decoder_dim, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # For storing attention weights during debugging
        self.store_attention_weights = False
        self.last_attention_weights = None

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(decoder_dim, d_model) # Query transformation
        self.W_k = nn.Linear(encoder_output_dim, d_model) # Key transformation
        self.W_v = nn.Linear(encoder_output_dim, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation

    
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, dim = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            padding_mask = mask.unsqueeze(1)  # [batch, 1, 1, seq_len]
            padding_mask = padding_mask.expand(-1, -1, attn_scores.size(-2), -1)  # [batch, 1, seq_len, seq_len]
            
            # Debug prints for masking
            if self.store_attention_weights:
                print("\nAttention scores before masking:")
                print(attn_scores[0, 0, :5, :5])  # Show first 5x5 elements
            
            # Apply padding mask
            attn_scores = attn_scores.masked_fill(~padding_mask, float(1e-9))
            
            # Create and apply causal mask
            nopeak_mask = (1 - torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1)), diagonal=1)).bool().to(attn_scores.device)
            
            # Debug prints for causal masking
            if self.store_attention_weights:
                print("\nCausal mask pattern:")
                print(nopeak_mask[:5, :5])
            
            attn_scores = attn_scores.masked_fill(~nopeak_mask, float('-inf'))
            
            # Debug prints after masking
            if self.store_attention_weights:
                print("\nAttention scores after masking:")
                print(attn_scores[0, 0, :5, :5])
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Store attention weights if in debug mode
        if self.store_attention_weights:
            self.last_attention_weights = attn_probs.detach()
            print("\nAttention probabilities:")
            print(attn_probs[0, 0, :5, :5])
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output, attn_probs
    
    def forward(self, query_input, key_input, value_input, mask=None):
        # Apply linear transformations and split heads
        query = self.split_heads(self.W_q(query_input))
        key = self.split_heads(self.W_k(key_input))
        value = self.split_heads(self.W_v(value_input))
        
        # Perform scaled dot-product attention
        attn_output, attn_probs = self.scaled_dot_product_attention(query, key, value, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output, attn_probs
    


