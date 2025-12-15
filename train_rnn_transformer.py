import argparse
import gzip
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from atlas_pytorch.rnn_transformer import RNNMemoryTransformer

# -----------------------------
# Data utilities
# -----------------------------

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
	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(f"Data file not found: {path}")
	with gzip.open(path, 'rb') as f:
		data = f.read(num_tokens)
	return torch.tensor(list(data), dtype=torch.long)

# -----------------------------
# Arg parsing
# -----------------------------

def parse_args():
	p = argparse.ArgumentParser(description='Train RNN-based MAG/MAL/LMM/MAC (stub)')
	# Architecture
	p.add_argument('--arch', type=str, default='mag', choices=['mag', 'mal', 'lmm', 'mac'], help='Architecture block')
	p.add_argument('--dim', type=int, default=256)
	p.add_argument('--depth', type=int, default=4)
	p.add_argument('--heads', type=int, default=4)
	p.add_argument('--dim-head', type=int, default=64)
	p.add_argument('--window-size', type=int, default=64, help='Sliding attention window (MAG/MAL)')
	p.add_argument('--persist-mem', type=int, default=0, help='Persistent memory tokens for attention')
	# Titans vs Atlas (Omega)
	p.add_argument('--model', type=str, default='titans', choices=['titans', 'omeganet'], help='Titans (e=1) or OmegaNet (e>=2)')
	p.add_argument('--omega-window', type=int, default=1, help='1 for Titans-RNN, >1 for OmegaNet-RNN')
	p.add_argument('--use-omega-gate', action='store_true', help='Enable U gates')
	p.add_argument('--poly-mode', type=str, default='off', choices=['off', 'elementwise', 'tensor'])
	p.add_argument('--poly-degree', type=int, default=1)
	# Training
	p.add_argument('--batch-size', type=int, default=16)
	p.add_argument('--seq-len', type=int, default=256)
	p.add_argument('--epochs', type=int, default=2)
	p.add_argument('--lr', type=float, default=1e-3)
	p.add_argument('--grad-clip', type=float, default=1.0)
	p.add_argument('--max-steps', type=int, default=None, help='Max train steps (batches) per run; helpful for smoke tests.')
	p.add_argument('--data-bytes', type=int, default=95_000_000, help='How many bytes to read from enwik8 (default full 95MB).')
	p.add_argument('--data-path', type=str, default='data/enwik8.gz')
	p.add_argument('--log-interval', type=int, default=1)
	p.add_argument('--save-path', type=str, default='checkpoints')
	return p.parse_args()

# -----------------------------
# Train / Val
# -----------------------------

def train_epoch(model, loader, opt, device, args):
	model.train()
	total_loss, total_tokens = 0.0, 0
	for step, (x, y) in enumerate(loader):
		if args.max_steps is not None and step >= args.max_steps:
			break
		x, y = x.to(device), y.to(device)
		opt.zero_grad()
		loss = model(x, return_loss=True)
		if isinstance(loss, tuple):
			# the wrapper may return (loss, logits, states)
			loss = loss[0]
		loss.backward()
		if args.grad_clip > 0:
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
		opt.step()
		total_loss += loss.item() * x.numel()
		total_tokens += x.numel()
		if (step + 1) % args.log_interval == 0:
			cur = loss.item()
			cur_ppl = math.exp(min(cur, 20))
			avg = total_loss / max(1, total_tokens)
			avg_ppl = math.exp(min(avg, 20))
			print(f"step {step+1} | loss {cur:.4f} (avg {avg:.4f}) | ppl {cur_ppl:.2f} (avg {avg_ppl:.2f})")
	return total_loss / max(1, total_tokens)


def evaluate(model, loader, device):
	model.eval()
	total_loss, total_tokens = 0.0, 0
	with torch.no_grad():
		for x, y in loader:
			x, y = x.to(device), y.to(device)
			loss = model(x, return_loss=True)
			if isinstance(loss, tuple):
				loss = loss[0]
			total_loss += loss.item() * x.numel()
			total_tokens += x.numel()
	avg = total_loss / max(1, total_tokens)
	return avg, math.exp(min(avg, 20))

# -----------------------------
# Main
# -----------------------------

def main():
	args = parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")

	# Data
	data = load_enwik8(args.data_path, num_tokens=args.data_bytes)
	n = len(data)
	train_data = data[: int(0.9 * n)]
	val_data = data[int(0.9 * n) : int(0.95 * n)]
	train_ds = TextDataset(train_data, args.seq_len)
	val_ds = TextDataset(val_data, args.seq_len)
	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

	# Arch selection
	block_type = args.arch

	# Model
	omega_window = args.omega_window
	if args.model == 'titans':
		omega_window = 1
	elif args.model == 'omeganet' and omega_window < 2:
		omega_window = 2

	model = RNNMemoryTransformer(
		num_tokens=256,
		dim=args.dim,
		depth=args.depth,
		block_type=block_type,
		window_size=args.window_size,
		dim_head=args.dim_head,
		heads=args.heads,
		num_persist_mem_tokens=args.persist_mem,
		omega_window=omega_window,
		use_omega_gate=args.use_omega_gate,
		use_momentum=True,
		poly_degree=args.poly_degree,
		poly_mode=args.poly_mode,
	).to(device)

	print('Model ready:')
	print(f"  arch={args.arch} block_type={block_type}")
	print(f"  model={args.model} omega_window={omega_window} gate={args.use_omega_gate}")
	print(f"  poly={args.poly_mode} degree={args.poly_degree}")

	opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)

	print(f"Start training for {args.epochs} epochs")
	best = float('inf')
	for epoch in range(args.epochs):
		print(f"\n== Epoch {epoch+1}/{args.epochs} ==")
		tr = train_epoch(model, train_loader, opt, device, args)
		vl, ppl = evaluate(model, val_loader, device)
		print(f"train {tr:.4f} | val {vl:.4f} | ppl {ppl:.2f}")

	print('Done')


if __name__ == '__main__':
	main()
