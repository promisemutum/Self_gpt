from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024        # Context length: how much text the model sees at once
    vocab_size: int = 50304       # GPT-2 vocab size (rounded up from 50257 for efficiency)
    n_layer: int = 12             # Number of transformer layers (depth)
    n_head: int = 12              # Number of attention heads
    n_embd: int = 768             # Embedding dimension (width of the network)
    dropout: float = 0.0          # Dropout rate (0.0 is fine for pre-training)
    bias: bool = False            # True: bias in Linears/LayerNorms (like GPT-2). False: faster/better.
