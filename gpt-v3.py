import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer
import pickle
import argparse
import math
import os
import platform

def detect_and_setup_device():
    """
    Detect available devices and choose the best one.
    Priority: NVIDIA GPU > Intel GPU > CPU
    """
    print("Detecting available devices...")
    
    # Check for NVIDIA GPU (CUDA)
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ NVIDIA GPU detected: {gpu_name}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device
    
    # Check for Intel GPU (DirectML on Windows, OpenCL/ROCm on Linux)
    intel_gpu_available = False
    
    # For Windows - check DirectML
    if platform.system() == "Windows":
        try:
            import torch_directml
            if torch_directml.is_available():
                device = torch_directml.device()
                print(f"✓ Intel GPU detected via DirectML")
                print(f"  Device: {device}")
                intel_gpu_available = True
                return device
        except ImportError:
            print("  DirectML not available (install torch-directml for Intel GPU support on Windows)")
    
    # For Linux - check for Intel GPU via OpenCL backend
    # Note: This requires special PyTorch builds with OpenCL support
    try:
        # Check if Intel GPU is available via PyTorch XPU (experimental)
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device = 'xpu'
            print("✓ Intel GPU detected via XPU backend")
            intel_gpu_available = True
            return device
    except:
        pass
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("✓ Apple Silicon GPU detected (MPS)")
        return device
    
    # Fallback to CPU
    print("✓ Using CPU")
    print("  To use Intel GPU:")
    print("    - Windows: Install torch-directml (pip install torch-directml)")
    print("    - Linux: Use Intel Extension for PyTorch")
    
    return 'cpu'

def optimize_for_device(device):
    """
    Optimize settings based on the device type
    """
    if device == 'cuda':
        # NVIDIA GPU settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        return {
            'batch_size': 8,
            'block_size': 512,
            'n_embd': 384,
            'n_head': 6,
            'n_layer': 6,
            'use_mixed_precision': True
        }
    
    elif 'directml' in str(device) or device == 'xpu':
        # Intel GPU settings - more conservative
        return {
            'batch_size': 4,
            'block_size': 256,
            'n_embd': 256,
            'n_head': 4,
            'n_layer': 4,
            'use_mixed_precision': False
        }
    
    elif device == 'mps':
        # Apple Silicon settings
        return {
            'batch_size': 6,
            'block_size': 512,
            'n_embd': 320,
            'n_head': 5,
            'n_layer': 5,
            'use_mixed_precision': False
        }
    
    else:
        # CPU settings - very conservative
        return {
            'batch_size': 2,
            'block_size': 128,
            'n_embd': 128,
            'n_head': 2,
            'n_layer': 2,
            'use_mixed_precision': False
        }

# Device setup
device = detect_and_setup_device()
device_config = optimize_for_device(device)

# Hyperparameters - adjusted based on device
batch_size = device_config['batch_size']
block_size = device_config['block_size'] 
max_iters = 10000
learning_rate = 6e-4
eval_iters = 200
n_embd = device_config['n_embd']
n_head = device_config['n_head']
n_layer = device_config['n_layer']
dropout = 0.1
warmup_iters = 2000
lr_decay_iters = 10000
min_lr = 6e-5
grad_clip = 1.0
weight_decay = 1e-1
use_mixed_precision = device_config['use_mixed_precision']

print(f"\nModel configuration for {device}:")
print(f"  Batch size: {batch_size}")
print(f"  Block size: {block_size}")
print(f"  Model size: {n_embd}d, {n_head} heads, {n_layer} layers")
print(f"  Mixed precision: {use_mixed_precision}")

# Setup mixed precision training
scaler = torch.cuda.GradScaler() if use_mixed_precision and device == 'cuda' else None

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size
encode = lambda s: tokenizer.encode(s, add_special_tokens=True)
decode = lambda l: tokenizer.decode(l, skip_special_tokens=True)

def load_half_dataset_into_memory(filename):
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found!")
        return ""
    with open(filename, 'r', encoding='utf-8') as f:
        f.seek(0, 2)
        half_point = f.tell() // 200
        f.seek(0)
        data = f.read(half_point)
    return data

train_data = load_half_dataset_into_memory("train_split.txt")
val_data = load_half_dataset_into_memory("val_split.txt")

def encode_dataset(text, chunk_size=512, max_tokens=None):
    if max_tokens is None:
        max_tokens = block_size
    tokens = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        token_ids = encode(chunk)
        token_ids = token_ids[:max_tokens]
        tokens.extend(token_ids)
    return tokens

train_encoded = torch.tensor(encode_dataset(train_data), dtype=torch.long)
val_encoded = torch.tensor(encode_dataset(val_data), dtype=torch.long)

print(f"Train tokens: {len(train_encoded)}, Val tokens: {len(val_encoded)}")

def get_batch(split):
    data = train_encoded if split == 'train' else val_encoded
    if len(data) <= block_size:
        print(f"Warning: Dataset too small for block_size {block_size}")
        return torch.zeros(batch_size, block_size, dtype=torch.long).to(device), torch.zeros(batch_size, block_size, dtype=torch.long).to(device)
    
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.head_dim = n_embd // n_head
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x).view(B, T, self.n_head, 3 * self.head_dim)
        q, k, v = qkv.split(self.head_dim, dim=-1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.mlp = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying
        self.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {num_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= block_size, f"Cannot forward sequence of length {t}, block size is only {block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.h:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

model = GPTLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            
            # Use autocast for mixed precision
            if use_mixed_precision and device == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _, loss = model(X, Y)
            else:
                _, loss = model(X, Y)
            
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training loop
best_val_loss = float('inf')
print(f"\nStarting training on {device}...")

for iter in range(max_iters):
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if iter % eval_iters == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Step {iter} | LR: {lr:.2e} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
        
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': losses['val'],
                'iter': iter,
                'device': str(device),
                'config': {
                    'n_embd': n_embd,
                    'n_head': n_head,
                    'n_layer': n_layer,
                    'block_size': block_size,
                    'vocab_size': vocab_size,
                    'dropout': dropout,
                }
            }
            torch.save(checkpoint, f'best_model_{str(device).replace(":", "_")}.pth')
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
    
    # Training step
    xb, yb = get_batch('train')
    
    # Mixed precision training
    if use_mixed_precision and device == 'cuda':
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = model(xb, yb)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    else:
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

# Save final model
model_filename = f'model-{str(device).replace(":", "_")}.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)
print(f"Final model saved as {model_filename}!")

# Test generation
print(f"\nTesting generation on {device}:")
try:
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=50, temperature=0.8, top_k=50)
    print(decode(generated[0].tolist()))
except Exception as e:
    print(f"Generation test failed: {e}")