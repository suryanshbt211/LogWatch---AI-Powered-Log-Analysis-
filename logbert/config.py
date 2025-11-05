"""Configuration dataclass for LogBERT++"""
from dataclasses import dataclass
import torch

@dataclass
class Config:
    """Single source of truth for all hyperparameters"""
    max_seq_len: int = 512
    max_vocab_size: int = 5000
    window_size: int = 150
    stride: int = 75
    vocab_size: int = 5000
    embed_dim: int = 128
    hidden_dim: int = 256
    num_heads: int = 4
    num_local_layers: int = 2
    num_global_layers: int = 1
    local_window_size: int = 64
    dropout: float = 0.1
    batch_size: int = 8
    num_epochs: int = 150
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    mask_ratio: float = 0.15
    span_mask_ratio: float = 0.1
    temperature: float = 0.07
    mlkp_weight: float = 1.0
    contrastive_weight: float = 0.2
    vhm_weight: float = 0.05
    span_weight: float = 0.1
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    top_g_candidates: int = 10