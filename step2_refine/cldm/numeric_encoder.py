"""
Numeric Condition Encoder based on Time2Vec
Encodes heading_num and range_num into embeddings for cross-attention conditioning.
"""

import torch
import torch.nn as nn
import math


class SineActivation(nn.Module):
    """Time2Vec style sine activation for periodic encoding"""
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(out_features - 1))
    
    def forward(self, tau):
        # tau: (batch, in_features)
        v1 = torch.sin(torch.matmul(tau, self.w) + self.b)  # periodic component
        v2 = torch.matmul(tau, self.w0) + self.b0  # linear component
        return torch.cat([v1, v2], dim=-1)


class NumericConditionEncoder(nn.Module):
    """
    Encodes heading_num and range_num into cross-attention compatible embeddings.
    
    Architecture:
    - heading_num -> Time2Vec -> MLP
    - range_num -> Time2Vec -> MLP
    - Concatenate and project to context_dim
    - Reshape to (batch, seq_len, context_dim) for cross-attention
    """
    def __init__(self, context_dim=768, hidden_dim=256, seq_len=77):
        super(NumericConditionEncoder, self).__init__()
        self.context_dim = context_dim
        self.seq_len = seq_len
        
        # Time2Vec encoding for heading (periodic, good for angles)
        self.heading_t2v = SineActivation(1, hidden_dim)
        
        # Time2Vec encoding for range
        self.range_t2v = SineActivation(1, hidden_dim)
        
        # MLP to process combined features
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, context_dim * seq_len),
        )
        
        # Layer norm for stable training
        self.ln = nn.LayerNorm(context_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, heading_num, range_num):
        """
        Args:
            heading_num: (batch,) or (batch, 1) - heading angle value
            range_num: (batch,) or (batch, 1) - range value
        
        Returns:
            context: (batch, seq_len, context_dim) - cross-attention context
        """
        batch_size = heading_num.shape[0]
        
        # Ensure correct shape (batch, 1)
        if heading_num.dim() == 1:
            heading_num = heading_num.unsqueeze(-1)
        if range_num.dim() == 1:
            range_num = range_num.unsqueeze(-1)
        
        # Time2Vec encoding
        heading_feat = self.heading_t2v(heading_num.float())  # (batch, hidden_dim)
        range_feat = self.range_t2v(range_num.float())  # (batch, hidden_dim)
        
        # Combine features
        combined = torch.cat([heading_feat, range_feat], dim=-1)  # (batch, hidden_dim * 2)
        
        # Project to full context
        context = self.mlp(combined)  # (batch, context_dim * seq_len)
        
        # Reshape for cross-attention
        context = context.view(batch_size, self.seq_len, self.context_dim)
        
        # Layer norm
        context = self.ln(context)
        
        return context


class NumericConditionEncoderSimple(nn.Module):
    """
    Simpler version: outputs a single token embedding that gets repeated.
    More parameter efficient.
    """
    def __init__(self, context_dim=768, hidden_dim=256, seq_len=77):
        super(NumericConditionEncoderSimple, self).__init__()
        self.context_dim = context_dim
        self.seq_len = seq_len
        
        # Time2Vec encoding for heading
        self.heading_t2v = SineActivation(1, hidden_dim)
        
        # Time2Vec encoding for range
        self.range_t2v = SineActivation(1, hidden_dim)
        
        # Project to context_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, context_dim),
        )
        
        # Learnable positional embeddings for the sequence
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, context_dim) * 0.02)
        
        self.ln = nn.LayerNorm(context_dim)
    
    def forward(self, heading_num, range_num):
        batch_size = heading_num.shape[0]
        
        if heading_num.dim() == 1:
            heading_num = heading_num.unsqueeze(-1)
        if range_num.dim() == 1:
            range_num = range_num.unsqueeze(-1)
        
        # Time2Vec encoding
        heading_feat = self.heading_t2v(heading_num.float())
        range_feat = self.range_t2v(range_num.float())
        
        # Combine and project
        combined = torch.cat([heading_feat, range_feat], dim=-1)
        token_embed = self.proj(combined)  # (batch, context_dim)
        
        # Expand to sequence and add positional embeddings
        context = token_embed.unsqueeze(1).expand(-1, self.seq_len, -1)
        context = context + self.pos_embed
        
        context = self.ln(context)
        
        return context
