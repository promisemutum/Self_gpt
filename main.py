import torch
from config import GPTConfig
from data_loader import DataLoader

# ---------------- CONFIG ----------------
dataset_name = "RJT1990/GeneralThoughtArchive"
# ----------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = GPTConfig()

# Initialize loader with HF dataset name
# max_samples=1000 is enough for a quick test. Increase later.
loader = DataLoader(
    dataset_name, 
    config.block_size, 
    batch_size=4, 
    device=device,
    max_samples=1000 
)

x, y = loader.get_batch('train')
print(f"\n--- Sample Data ---\n{loader.decode(x[0].tolist())[:200]}...")
