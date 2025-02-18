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
            query_key = query_key.masked_fill(~nopeak_mask, float('-inf'))
            # print(mask.shape, 'mask.shape', query_key.shape)
            # query_key = query_key.masked_fill(~mask, float('-inf'))

        prob = query_key.softmax(dim=-1)
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
        print(query.shape, "query.shape")
        print(key.shape, "key.shape")
        print(value.shape, "value.shape")
        query_key = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(dim_k)
        print(query_key.shape, "query_key.shape")
        prob = query_key.softmax(dim=-1)
        print(prob.shape, "prob.shape")
        weighted_attention = torch.matmul(prob, value)
        print(weighted_attention.shape, "weighted_attention.shape")
        return weighted_attention, prob

