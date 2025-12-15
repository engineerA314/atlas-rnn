"""
Training script for RNN Memory models (Titans-RNN and OmegaNet-RNN).

This script trains a simple language model using the RNN memory architecture
derived from the Titans and Atlas papers.
"""

import argparse
import gzip
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from atlas_pytorch.rnn_memory import RNNMemory, RNNMemoryCell, OmegaRNNMemoryCell

# ============================================================================
# Configuration
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train RNN Memory Language Model')
    
    # Model architecture
    parser.add_argument('--dim', type=int, default=256, help='Model dimension')
    parser.add_argument('--dim-head', type=int, default=64, help='Head dimension')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--depth', type=int, default=4, help='Number of layers')
    
    # Memory configuration
    parser.add_argument('--omega-window', type=int, default=1, 
                       help='Omega window size (1=Titans, >1=OmegaNet)')
    parser.add_argument('--use-omega-gate', action='store_true',
                       help='Use learnable omega gates')
    parser.add_argument('--use-momentum', action='store_true', default=True,
                       help='Use momentum in memory updates')
    parser.add_argument('--poly-degree', type=int, default=1,
                       help='Polynomial feature degree')
    parser.add_argument('--poly-mode', type=str, default='off',
                       choices=['off', 'elementwise', 'tensor'],
                       help='Polynomial feature mode')
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=256, help='Sequence length')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Limit number of training steps (batches) for smoke tests')
    
    # Data
    parser.add_argument('--data-path', type=str, default='data/enwik8.gz',
                       help='Path to enwik8.gz data file')
    parser.add_argument('--data-bytes', type=int, default=95_000_000,
                       help='How many bytes to read from data file (default full 95MB)')
    
    # Logging
    parser.add_argument('--log-interval', type=int, default=1,
                       help='Log every N steps')
    parser.add_argument('--save-path', type=str, default='checkpoints',
                       help='Path to save checkpoints')
    
    return parser.parse_args()

# ============================================================================
# Data
# ============================================================================

class TextDataset(Dataset):
    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return (len(self.data) - 1) // self.seq_len
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        return chunk[:-1], chunk[1:]

def load_enwik8(path: str, num_tokens: int = 95_000_000):
    """Load enwik8 dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    with gzip.open(path, 'rb') as f:
        data = f.read(num_tokens)
    
    # Convert to tensor of byte values
    data = torch.tensor(list(data), dtype=torch.long)
    return data

# ============================================================================
# Model
# ============================================================================

class RNNMemoryLM(nn.Module):
    """
    Simple language model using RNN Memory.
    """
    
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 4,
        omega_window: int = 1,
        use_omega_gate: bool = False,
        use_momentum: bool = True,
        poly_degree: int = 1,
        poly_mode: str = 'off',
    ):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.dim = dim
        self.depth = depth
        
        # Token embedding
        self.embed = nn.Embedding(num_tokens, dim)
        self.embed_norm = nn.LayerNorm(dim)
        
        # RNN Memory layers
        self.layers = nn.ModuleList()
        for _ in range(depth):
            if omega_window > 1 or use_omega_gate:
                cell = OmegaRNNMemoryCell(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    omega_window=omega_window,
                    use_omega_gate=use_omega_gate,
                    use_momentum=use_momentum,
                    poly_degree=poly_degree,
                    poly_mode=poly_mode,
                )
            else:
                cell = RNNMemoryCell(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    use_momentum=use_momentum,
                    poly_degree=poly_degree,
                    poly_mode=poly_mode,
                )
            self.layers.append(cell)
        
        # Output projection
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        # Small init for embeddings (RWKV-style)
        nn.init.uniform_(self.embed.weight, -1e-4, 1e-4)
        # Zero init for output projection
        nn.init.zeros_(self.to_logits.weight)
    
    def forward(self, x, states=None, return_loss=False, targets=None):
        """
        Forward pass.
        
        Args:
            x: Input token ids [batch, seq_len]
            states: List of layer states (optional)
            return_loss: Whether to return loss
            targets: Target token ids for loss computation
        """
        batch, seq_len = x.shape
        
        # Embed
        h = self.embed(x)
        h = self.embed_norm(h)
        
        # Initialize states if needed
        if states is None:
            states = [None] * self.depth
        
        next_states = []
        
        # Process through layers
        for layer, state in zip(self.layers, states):
            out, next_state = layer(h, state)
            h = h + out  # Residual connection
            next_states.append(next_state)
        
        # Output
        h = self.norm(h)
        logits = self.to_logits(h)
        
        if return_loss:
            if targets is None:
                raise ValueError("targets required when return_loss=True")
            loss = F.cross_entropy(
                logits.view(-1, self.num_tokens),
                targets.view(-1)
            )
            return loss, logits, next_states
        
        return logits, next_states

# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, args):
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    
    for step, (x, y) in enumerate(dataloader):
        if args.max_steps is not None and step >= args.max_steps:
            break
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        loss, _, _ = model(x, return_loss=True, targets=y)
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()
        
        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            cur_loss = loss.item()
            cur_ppl = math.exp(min(cur_loss, 20))
            avg_loss = total_loss / total_tokens
            avg_ppl = math.exp(min(avg_loss, 20))  # Cap for numerical stability
            tokens_per_sec = total_tokens / elapsed
            
            print(f"Step {step+1} | loss {cur_loss:.4f} (avg {avg_loss:.4f}) | "
                  f"ppl {cur_ppl:.2f} (avg {avg_ppl:.2f}) | tokens/s: {tokens_per_sec:.0f}")
    
    return total_loss / total_tokens

def main():
    args = parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    try:
        data = load_enwik8(args.data_path, num_tokens=args.data_bytes)
    except FileNotFoundError:
        print(f"Data file not found. Please download enwik8.gz to {args.data_path}")
        print("You can download it from: https://data.deepai.org/enwik8.zip")
        return
    
    # Split data
    n = len(data)
    train_data = data[:int(0.9 * n)]
    val_data = data[int(0.9 * n):int(0.95 * n)]
    
    print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")
    
    # Create datasets
    train_dataset = TextDataset(train_data, args.seq_len)
    val_dataset = TextDataset(val_data, args.seq_len)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = RNNMemoryLM(
        num_tokens=256,  # Byte-level
        dim=args.dim,
        depth=args.depth,
        dim_head=args.dim_head,
        heads=args.heads,
        omega_window=args.omega_window,
        use_omega_gate=args.use_omega_gate,
        use_momentum=args.use_momentum,
        poly_degree=args.poly_degree,
        poly_mode=args.poly_mode,
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Omega window: {args.omega_window}")
    print(f"  Use omega gate: {args.use_omega_gate}")
    print(f"  Use momentum: {args.use_momentum}")
    print(f"  Poly degree: {args.poly_degree}")
    print(f"  Poly mode: {args.poly_mode}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=0.01
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, args)
        
        # Validation
        model.eval()
        val_loss = 0
        val_tokens = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                loss, _, _ = model(x, return_loss=True, targets=y)
                val_loss += loss.item() * x.numel()
                val_tokens += x.numel()
        
        val_loss = val_loss / val_tokens
        val_ppl = math.exp(min(val_loss, 20))
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {math.exp(min(train_loss, 20)):.2f}")
        print(f"  Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = Path(args.save_path)
            save_path.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': args,
            }, save_path / 'best_model.pt')
            print(f"  Saved best model (val_loss: {val_loss:.4f})")
    
    print("\nTraining complete!")

if __name__ == '__main__':
    main()

