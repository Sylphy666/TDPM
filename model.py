import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_rotary_pos_emb(q, k):
    seq_len = q.size(1)
    dim = q.size(-1)
    
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=q.device).float() / dim))
    t = torch.arange(seq_len, device=q.device).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    
    q_out = torch.view_as_real(q_ * freqs_cis.unsqueeze(0).unsqueeze(2)).flatten(3)
    k_out = torch.view_as_real(k_ * freqs_cis.unsqueeze(0).unsqueeze(2)).flatten(3)
    return q_out.type_as(q), k_out.type_as(k)

class RoPEAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, L, n_heads, head_dim)
        
        q, k = apply_rotary_pos_emb(q, k)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)

class RoPETransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = RoPEAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, padding_mask):
        x = x + self.attn(self.norm1(x), padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x

class TDPM(nn.Module):
    def __init__(self, args, total_vocab_size):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.sid_len = args.sid_len
        self.embedding = nn.Embedding(total_vocab_size, self.embed_dim)
        
        self.time_proj = nn.Linear(1, self.embed_dim)

        self.layers = nn.ModuleList([
            RoPETransformerBlock(args.embed_dim, args.nhead, args.dim_feedforward, args.dropout)
            for _ in range(args.num_layers)
        ])
        self.norm = nn.LayerNorm(args.embed_dim)
        self.predictor = nn.Linear(args.embed_dim, total_vocab_size)

    def forward(self, src, src_key_padding_mask, time_embed=None):
        B, L = src.shape
        x = self.embedding(src) * math.sqrt(self.embed_dim)
        
        if time_embed is not None:
            x = x + self.time_proj(time_embed.unsqueeze(-1))
        
        for layer in self.layers:
            x = layer(x, padding_mask=src_key_padding_mask)
            
        x = self.norm(x)
        logits = self.predictor(x)
        return logits