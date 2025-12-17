import torch
import tiktoken
from datasets import load_dataset

class DataLoader:
    def __init__(self, dataset_name, block_size, batch_size, device, max_samples=10000):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        
        print(f"Streaming dataset '{dataset_name}' from Hugging Face...")
        
        # Load dataset in streaming mode 
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        
        self.enc = tiktoken.get_encoding("gpt2")
        self.data = []
        
        print(f"Processing first {max_samples} samples...")
        count = 0
        
        # Iterate and format
        for item in dataset:
            if count >= max_samples:
                break
                
            # Extract fields (handle potential missing values)
            q = item.get('question', '')
            r = item.get('model_reasoning', '')
            a = item.get('model_answer', '')
            
            if not q or not a: continue # Skip empty
            
            # Format: User -> Reasoning -> Answer
            text = f"User: {q}\nReasoning: {r}\nAnswer: {a}\n<|endoftext|>\n"
            
            # Encode
            tokens = self.enc.encode(text, allowed_special={'<|endoftext|>'})
            self.data.extend(tokens)
            count += 1
            
        # Convert to tensor
        self.data = torch.tensor(self.data, dtype=torch.long)
        print(f"Loaded {len(self.data):,} tokens from {count} samples.")
        
        # Train/Val split
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        
        # 1. Generate random indices
        # If global default is CUDA, this is already on CUDA.
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        
        # 2. Move indices to CPU to slice the main dataset (which is on CPU)
        ix_cpu = ix.to('cpu')
        
        # 3. Slice the data
        # 'data' is on CPU, so this operation happens on CPU
        x = torch.stack([data[i:i+self.block_size] for i in ix_cpu])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix_cpu])
        
        # 4. Move final batch to GPU
        # We REMOVE .pin_memory() because it conflicts with some setups
        # and just move strictly to device.
        if self.device == 'cuda':
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
            
        return x, y


    def decode(self, token_ids):
        return self.enc.decode(token_ids)
