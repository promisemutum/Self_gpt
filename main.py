import torch
from config import GPTConfig
from data_loader import DataLoader
from models import GPT

dataset_name = "RJT1990/GeneralThoughtArchive"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = GPTConfig()

# Initialize
loader = DataLoader(
    dataset_name, 
    config.block_size, 
    batch_size=4, 
    device=device,
    max_samples=1000 
)

print("Model Initialization...")
model = GPT(config).to(device)
print("Model initialized successfully...")

model = torch.compile(model)
print("Model compiled successfully...")

print("Training started...")    
x, y =  loader.get_batch('train')
logits, loss = model(x,y)
print(f"Logits shape: {logits.shape}, Loss: {loss.item():.4f}")