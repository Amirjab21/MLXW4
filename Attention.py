import torch
import math
import torch.nn as nn

class Attention_Layer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Attention_Layer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
    
    def forward(self, query_input, key_input, value_input, mask=None):
        dim_k = self.d_model // self.num_heads
        query = self.W_q(query_input)
        key = self.W_k(key_input)
        value = self.W_v(value_input)
        

        
        query_key = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(dim_k)

        if mask is not None:
            nopeak_mask = (1 - torch.triu(torch.ones(query_key.size(-2), query_key.size(-1)), diagonal=1)).bool()
            nopeak_mask = nopeak_mask.to(query_key.device)
            query_key = query_key.masked_fill(~nopeak_mask, float('-inf'))
            # print(mask.shape, 'mask.shape', query_key.shape)
            # print("Mask values:", torch.unique(mask))  # Check unique values in mask
            # print("Mask shape:", mask.shape)
            # print("Query key shape:", query_key.shape)
            # print(query_key[0], 'first query key')
            # query_key = query_key.masked_fill(~mask, float('-inf'))
            # print(query_key[0], 'final query key')

        prob = query_key.softmax(dim=-1)
        # print(prob[0], 'prob')
        weighted_attention = torch.matmul(prob, value)
        return weighted_attention, prob
    
class Cross_Attention_Layer(nn.Module):
    def __init__(self, encoder_output_dim, decoder_dim, d_model, num_heads):
        super(Cross_Attention_Layer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.W_q = torch.nn.Linear(decoder_dim, d_model)

        self.W_k = torch.nn.Linear(encoder_output_dim, d_model)
        self.W_v = torch.nn.Linear(encoder_output_dim, d_model)
    
    def forward(self, query_input, key_input, value_input, mask=None):
        dim_k = self.d_model // self.num_heads
        query = self.W_q(query_input)
        key = self.W_k(key_input)
        value = self.W_v(value_input)
        query_key = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(dim_k)

        # if mask is not None:
        #     # nopeak_mask = (1 - torch.triu(torch.ones(query_key.size(-2), query_key.size(-1)), diagonal=1)).bool()
        #     all_false_rows = (~mask).all(dim=-1)  # Shape: [batch_size, seq_len]
        #     if all_false_rows.any():
        #         # Get the first index where all values are False for each sequence
        #         first_all_false_idx = torch.argmax((all_false_rows).int(), dim=-1)
                
        #         # Create a mask for all positions from first_all_false_idx onwards
        #         row_indices = torch.arange(query_key.size(-2), device=query_key.device)
        #         row_mask = row_indices.unsqueeze(0) >= first_all_false_idx.unsqueeze(1)
        #         row_mask = row_mask.unsqueeze(-1).expand_as(query_key)
                
        #         # Apply the mask
        #         query_key = query_key.masked_fill(row_mask, float('-inf'))
        #     print(query_key[0], 'final query key')
        

        prob = query_key.softmax(dim=-1)
        weighted_attention = torch.matmul(prob, value)
        return weighted_attention, prob
    

class MultiHeadAttention(nn.Module):
    def __init__(self, encoder_output_dim, decoder_dim, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value

        self.attn_probs = None

                # Linear layers for transforming inputs
        self.W_q = nn.Linear(decoder_dim, d_model) # Query transformation
        self.W_k = nn.Linear(encoder_output_dim, d_model) # Key transformation
        self.W_v = nn.Linear(encoder_output_dim, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation

    
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
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
            nopeak_mask = (1 - torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1)), diagonal=1)).bool().to(attn_scores.device)
            attn_scores = attn_scores.masked_fill(~nopeak_mask, float('-inf'))
            # print(mask.shape, 'mask.shape', query_key.shape)
            # attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
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
        self.attn_probs = attn_probs
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output, attn_probs
    


