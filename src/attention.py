import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D/H]
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D/H]
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D/H]
        
        # Generate attention mask if needed (for padding or causal attention)
        if mask is None and query.size(1) > 1:  # For sequence data
            mask = torch.ones(query.size(1), key.size(1), device=query.device)
            # Optional: make it causal (lower triangular) for decoder
            # mask = torch.tril(mask)
        
        # Scaled dot-product attention as per formula
        d_k = self.head_dim
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [B, H, T, T]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads explicitly as shown in math
        # First, collect outputs from each head
        head_outputs = []
        for h in range(self.num_heads):
            head_output = context[:, h:h+1, :, :]  # [B, 1, T, D/H]
            head_outputs.append(head_output)
        
        # Concatenate all heads
        multi_head = torch.cat(head_outputs, dim=1)  # [B, H, T, D/H]
        
        # Reshape and apply final projection
        context = multi_head.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        output = self.out_linear(context)  # Final W^O projection
        
        return output, attention_weights