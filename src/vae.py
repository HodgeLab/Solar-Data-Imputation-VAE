# VAE code is here
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention

class VAELSTMAttention(nn.Module):
    def __init__(self, 
                 static_input_dim,
                 dynamic_input_dim,
                 hidden_dim,
                 latent_dim,
                 num_heads=4,
                 num_layers=1,
                 dropout=0.1,
                 padding_value=-1):  # Added padding value
        
        self.padding_value = padding_value
        super().__init__()
        
        self.static_input_dim = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Encoder components
        self.lstm_encoder = nn.LSTM(
            input_size=dynamic_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Self-attention for dynamic input
        self.dynamic_attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Static input processing
        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Cross-attention between static and dynamic features
        self.cross_attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Combine encodings
        self.combine_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # VAE components
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder components
        self.decoder_input = nn.Linear(latent_dim + static_input_dim, hidden_dim)
        
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder attention
        self.decoder_attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.output_layer = nn.Linear(hidden_dim, dynamic_input_dim)
        
    def encode(self, dynamic_x, static_x):
        # Process dynamic input through LSTM
        lstm_out, (h_n, _) = self.lstm_encoder(dynamic_x)
        
        # Apply self-attention to LSTM output (temporal attention)
        dynamic_attended, dynamic_weights = self.dynamic_attention(lstm_out, lstm_out, lstm_out)
        
        # Process static input
        static_encoding = self.static_encoder(static_x)
        
        # Bidirectional cross-attention
        # 1. Dynamic features attending to static
        static_encoding_expanded = static_encoding.unsqueeze(1).expand(-1, dynamic_attended.size(1), -1)
        dynamic_to_static, d2s_weights = self.cross_attention(
            dynamic_attended, static_encoding_expanded, static_encoding_expanded
        )
        
        # 2. Static features attending to dynamic
        static_to_dynamic, s2d_weights = self.cross_attention(
            static_encoding.unsqueeze(1), dynamic_attended, dynamic_attended
        )
        
        # Combine attended features
        dynamic_pooled = torch.mean(dynamic_to_static, dim=1)  # Pool temporal dimension
        static_pooled = static_to_dynamic.squeeze(1)  # Remove temporal dimension
        
        # Combine encodings with both attention directions
        combined = torch.cat([dynamic_pooled, static_pooled], dim=1)
        hidden = self.combine_encoder(combined)
        
        # Generate latent distribution parameters
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z, static_x, seq_len):
        # Combine latent vector with static input
        decoder_input = torch.cat([z, static_x], dim=1)
        hidden = self.decoder_input(decoder_input)
        
        # Expand for sequence generation
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Generate sequence with LSTM
        lstm_out, _ = self.lstm_decoder(hidden)
        
        # Apply attention to decoder output
        attended_out, _ = self.decoder_attention(lstm_out, lstm_out, lstm_out)
        
        # Generate final output
        output = self.output_layer(attended_out)
        
        return output
    
    def forward(self, dynamic_x, static_x, lengths=None):
        seq_len = dynamic_x.size(1)
        batch_size = dynamic_x.size(0)
        
        # Create attention mask based on sequence lengths
        if lengths is not None:
            # Create mask for padding
            mask = torch.arange(seq_len).expand(batch_size, seq_len).to(dynamic_x.device)
            mask = mask < lengths.unsqueeze(1)
            
            # Create mask for missing values
            padding_mask = (dynamic_x != self.padding_value).any(dim=-1)
            
            # Combine both masks
            mask = mask & padding_mask
        else:
            mask = None
        
        # Encode
        mu, log_var = self.encode(dynamic_x, static_x)
        
        # Sample from latent space
        z = self.reparameterize(mu, log_var)
        
        # Decode
        recon_x = self.decode(z, static_x, seq_len)
        
        return recon_x, mu, log_var
    
    def loss_function(self, recon_x, x, mu, log_var, beta=1.0):
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss + beta * kl_loss