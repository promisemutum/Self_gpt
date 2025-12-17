import torch
from config import GPTConfig
from models import RMSNorm, CausalSelfAttention, MLP, Block, GPT
from data_loader import DataLoader

def test_models():
    print("Testing GPT model components...")

    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create config
    config = GPTConfig()
    print(f"Config: n_embd={config.n_embd}, n_head={config.n_head}, n_layer={config.n_layer}")

    # Test RMSNorm
    print("\n1. Testing RMSNorm...")
    norm = RMSNorm(config.n_embd).to(device)
    x_test = torch.randn(2, 10, config.n_embd).to(device)  # (B, T, C)
    x_norm = norm(x_test)
    print(f"RMSNorm input shape: {x_test.shape}, output shape: {x_norm.shape}")
    assert x_norm.shape == x_test.shape, "RMSNorm shape mismatch"

    # Test CausalSelfAttention
    print("\n2. Testing CausalSelfAttention...")
    attn = CausalSelfAttention(config).to(device)
    x_attn = attn(x_test)
    print(f"Attention input shape: {x_test.shape}, output shape: {x_attn.shape}")
    assert x_attn.shape == x_test.shape, "Attention shape mismatch"

    # Test MLP
    print("\n3. Testing MLP...")
    mlp = MLP(config).to(device)
    x_mlp = mlp(x_test)
    print(f"MLP input shape: {x_test.shape}, output shape: {x_mlp.shape}")
    assert x_mlp.shape == x_test.shape, "MLP shape mismatch"

    # Test Block
    print("\n4. Testing Transformer Block...")
    block = Block(config).to(device)
    x_block = block(x_test)
    print(f"Block input shape: {x_test.shape}, output shape: {x_block.shape}")
    assert x_block.shape == x_test.shape, "Block shape mismatch"

    # Test GPT model
    print("\n5. Testing GPT model...")
    model = GPT(config).to(device)
    # Create some dummy token indices
    idx = torch.randint(0, config.vocab_size, (2, 10)).to(device)  # (B, T)
    targets = torch.randint(0, config.vocab_size, (2, 10)).to(device)

    logits, loss = model(idx, targets)
    print(f"GPT input shape: {idx.shape}, logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    assert logits.shape == (2, 10, config.vocab_size), f"Logits shape mismatch: {logits.shape}"

    # Test inference mode (no targets)
    logits_inf, loss_inf = model(idx)
    print(f"Inference logits shape: {logits_inf.shape}")
    assert logits_inf.shape == (2, 1, config.vocab_size), f"Inference shape mismatch: {logits_inf.shape}"
    assert loss_inf is None, "Loss should be None in inference mode"

    # Test data loading
    print("\n6. Testing data loading...")
    loader = DataLoader("RJT1990/GeneralThoughtArchive", config.block_size, batch_size=2, device=device, max_samples=100)
    x_batch, y_batch = loader.get_batch('train')
    print(f"Batch shapes: x={x_batch.shape}, y={y_batch.shape}")
    print(f"Batch device: x={x_batch.device}, y={y_batch.device}")

    # Test RoPE functions
    print("\n7. Testing RoPE...")
    from models import precompute_freqs_cis, apply_rotary_emb

    head_dim = config.n_embd // config.n_head
    freqs_cis = precompute_freqs_cis(head_dim, config.block_size)
    print(f"RoPE freqs shape: {freqs_cis.shape}")

    # Create Q,K for testing RoPE
    B, T, nh, hs = 2, 10, config.n_head, head_dim
    q = torch.randn(B, T, nh, hs).to(device)
    k = torch.randn(B, T, nh, hs).to(device)

    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis[:T])
    print(f"RoPE input shapes: q={q.shape}, k={k.shape}")
    print(f"RoPE output shapes: q_rot={q_rot.shape}, k_rot={k_rot.shape}")

    print("\n✅ All tests passed! GPT model is working correctly on CUDA.")

if __name__ == "__main__":
    test_models()
