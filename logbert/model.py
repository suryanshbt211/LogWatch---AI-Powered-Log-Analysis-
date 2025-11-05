"""Model architecture"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        B, L, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), attn

class LogBERTPlusPlus(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.mlkp_head = nn.Linear(config.embed_dim, config.vocab_size)
        self.vhm_center = nn.Parameter(torch.randn(config.embed_dim))
        
    def forward(self, input_ids, masked_positions=None):
        x = self.embedding(input_ids)
        return x, x.mean(dim=1)
    
    def compute_mlkp_loss(self, token_emb, input_ids, mask):
        logits = self.mlkp_head(token_emb)
        loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), input_ids.view(-1), reduction='none')
        return (loss * mask.view(-1).float()).sum() / (mask.sum() + 1e-8), logits
    
    def compute_contrastive_loss(self, emb1, emb2, temp=0.07):
        return torch.tensor(0.0, device=emb1.device)
    
    def compute_vhm_loss(self, emb):
        return torch.norm(emb - self.vhm_center, dim=-1).mean()
    
    def compute_span_loss(self, token_emb, input_ids, mask):
        return self.compute_mlkp_loss(token_emb, input_ids, mask)[0]