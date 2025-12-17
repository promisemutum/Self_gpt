# main.py
import torch
import os
from config import GPTConfig
from models import GPT
from data_loader import DataLoader

# ==========================================
# 1. GLOBAL CUDA SETUP
# ==========================================
if torch.cuda.is_available():
    torch.set_default_device('cuda')
    device = 'cuda'
    print(f"Global default device: {torch.get_default_device()}")
else:
    device = 'cpu'
    print("Running on CPU.")

# ---------------- CONFIG ----------------
dataset_name = "RJT1990/GeneralThoughtArchive"
# ----------------------------------------

config = GPTConfig()

# Initialize loader
# (The loader class now handles keeping the big dataset on CPU automatically)
print("Initializing DataLoader...")
loader = DataLoader(
    dataset_name, 
    config.block_size, 
    batch_size=4, 
    device=device,
    max_samples=1000 
)

# Initialize model
# (Model is created on GPU automatically because of set_default_device)
print("Initializing Model...")
model = GPT(config)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Test Forward Pass
print("Testing Forward Pass...")
x, y = loader.get_batch('train')

# Ensure batch is on the correct device (sanity check)
print(f"Input batch device: {x.device}") 

logits, loss = model(x, y)
print(f"Forward pass successful. Loss: {loss.item():.4f}")

# Test Generation (Optional - to see if it spits out text)
print("\n--- Test Generation ---")
# Create a start token (0 is usually safe/padding, or use enc.encode("\n"))
start_context = torch.zeros((1, 1), dtype=torch.long, device=device)

# We need a simple generate function (copying logic from Karpathy's original for testing)
# You can add this method to your GPT class in model.py later for cleaner code
def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] 
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# Generate 20 tokens
generated_ids = generate(model, start_context, max_new_tokens=20)
print(f"Generated output: {loader.decode(generated_ids[0].tolist())}")