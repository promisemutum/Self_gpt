import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GPTConfig


# Rotatory positional embedding

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precomputes the frequency tensor for complex exponentials.
    
    Args:
        dim: Dimension of the head (n_embd / n_head).
        end: Maximum context length (block_size * 2 to be safe).
        theta: Scaling factor (10000.0 is standard, larger for long context).
    """
    # Create a list of frequencies: 1 / (theta ^ (2i / d))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # Create a list of positions: [0, 1, 2, ..., end]
    t = torch.arange(end, device=freqs.device) 
    
    # Outer product: combine positions and frequencies to get angles for every pos/dim
    freqs = torch.outer(t, freqs).float()
    
    # Turn angles into complex numbers: cos(a) + i*sin(a)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Applies the rotation to Queries (xq) and Keys (xk).

    Args:
        xq: Query tensor of shape (Batch, Time, Head, Head_Dim)
        xk: Key tensor of shape (Batch, Time, Head, Head_Dim)
        freqs_cis: Precomputed complex frequencies
    """
    # 1. Reshape real numbers into complex numbers (pairs of 2 become 1 complex number)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 2. Slice frequencies to match the current sequence length
    freqs_cis = freqs_cis[:xq.shape[1]]

    # 3. Add singleton dimensions for batch and head to enable broadcasting
    # freqs_cis: (T, D//2) -> (1, T, 1, D//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # 4. Rotate! (Complex multiplication applies rotation)
    # (a+bi) * (c+di) -> rotates vector (a,b) by angle of (c,d)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

# Root Mean Square Layer Normalization

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scaling parameter (gamma), initialized to 1s.
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Calculate RMS: sqrt(mean(x^2))
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Normalize and then scale by the learnable weight
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# Causal Self-Attention(with RoPE)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Linear projections for Key, Query, Value
        # We project all 3 at once (3 * n_embd) for efficiency
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Setup RoPE Cache
        self.head_dim = config.n_embd // config.n_head
        # Precompute rotational frequencies once
        freqs = precompute_freqs_cis(self.head_dim, config.block_size * 2)
        # register_buffer means it's part of state_dict but not a learnable parameter (no gradients)
        self.register_buffer("freqs_cis", freqs, persistent=False)

    def forward(self, x):
        # Batch, Time, Channels
        B, T, C = x.size() 

        # 1. Calculate Query, Key, Value
        # split(..., dim=2) separates the big 3*C tensor into three C tensors
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # 2. Reshape for Multi-Head Attention
        # (B, T, C) -> (B, T, n_head, head_dim) -> transpose -> (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 

        # 3. Apply RoPE
        # We transpose (B, nh, T, hs) -> (B, T, nh, hs) because our RoPE helper expects T at dim 1
        q_rope = q.transpose(1, 2)
        k_rope = k.transpose(1, 2)
        
        # Get frequencies for current sequence length T
        freqs_cis = self.freqs_cis[:T]
        
        # Rotate Q and K
        q_rotated, k_rotated = apply_rotary_emb(q_rope, k_rope, freqs_cis)
        
        # Transpose back to (B, nh, T, hs) for attention
        q = q_rotated.transpose(1, 2)
        k = k_rotated.transpose(1, 2)

        # 4. Flash Attention
        # Instead of manual (Q @ K) / sqrt(d), masking, and Softmax...
        # is_causal=True handles the masking (tokens can't see future tokens).
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout if self.training else 0, 
            is_causal=True
        )

        # 5. Reassemble Heads
        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        
        # 6. Output Projection
        y = self.c_proj(y)
        return y

# Multi-Layer Perceptron

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Expansion layer: n_embd -> 4 * n_embd
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # Activation: GELU 
        self.gelu    = nn.GELU()
        # Projection layer: 4 * n_embd -> n_embd
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# Transformer Block

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Residual connection (x + ...) is CRITICAL for deep networks.
        # It allows gradients to flow through unchanged, preventing vanishing gradients.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# GPT Model

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # A ModuleDict is just a dictionary that holds PyTorch modules
        self.transformer = nn.ModuleDict(dict(
            # Token embeddings: converts integer IDs (0, 523, ...) to vectors
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            
            # No Positional Embedding layer 
            # We use RoPE inside the attention block instead.
            
            drop = nn.Dropout(config.dropout),
            
            # The stack of transformer blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            
            # Final normalization before the output
            ln_f = RMSNorm(config.n_embd),
        ))
        
        # Projects final vectors 
        # This gives us the probability for the next token.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # WEIGHT TYING: 
        # We share weights between the token embedding layer and the output head.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        # 1. Get Token Embeddings
        tok_emb = self.transformer.wte(idx) # (Batch, Time, n_embd)
        x = self.transformer.drop(tok_emb)
        
        # 2. Run through all Transformer Blocks
        for block in self.transformer.h:
            x = block(x)
            
        # 3. Final Norm
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training Mode: Calculate Loss
            # lm_head produces logits (unnormalized scores) for next token
            logits = self.lm_head(x)
            
            # Flatten to (Batch * Time, Vocab_Size) for Cross Entropy
            # This compares predicted logits vs actual target tokens
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference Mode: Only predict the *next* token (last time step)
            # This is an optimization. We don't need predictions for the past.
            logits = self.lm_head(x[:, [-1], :]) 
            loss = None

        return logits, loss