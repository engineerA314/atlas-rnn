"""
RNN Memory Transformers: MAG, MAL, LMM architectures using RNN-form memory.

These architectures use the explicit RNN update rules derived from Titans/Atlas,
replacing the per-sample gradient computation with closed-form matrix updates.
"""

from __future__ import annotations
from typing import Callable
from functools import partial
from collections import namedtuple

import tqdm
import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear

from einops import repeat, rearrange, pack, unpack
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding
from x_transformers.attend import Attend

from atlas_pytorch.rnn_memory import RNNMemoryCell, OmegaRNNMemoryCell, RNNMemState, state_detach

# ============================================================================
# Constants and Helpers
# ============================================================================

LinearNoBias = partial(Linear, bias=False)

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# Sampling helpers
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature=1.):
    if temperature > 0.:
        t = t / temperature + gumbel_noise(t)
    return t.argmax(dim=-1, keepdim=True)

def min_p_filter(logits, min_p=0.1):
    probs = logits.softmax(dim=-1)
    max_probs = probs.amax(dim=-1, keepdim=True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

# ============================================================================
# Components
# ============================================================================

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

def FeedForward(dim, mult=4):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Linear(dim_inner, dim)
    )

class SlidingWindowAttention(Module):
    """Sliding window attention for short-term memory (MAG/MAL)."""
    
    def __init__(
        self,
        dim,
        window_size,
        num_persist_mem_tokens=0,
        dim_head=64,
        heads=8,
        attend_kwargs: dict = dict(),
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.heads = heads
        self.window_size = window_size
        self.num_persist_mem_tokens = num_persist_mem_tokens
        
        dim_inner = dim_head * heads
        
        self.to_qkv = LinearNoBias(dim, dim_inner * 3)
        self.to_out = LinearNoBias(dim_inner, dim)
        
        self.split_heads = Rearrange('b n (h d) -> b h n d', h=heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        
        self.rotary_emb = RotaryEmbedding(dim_head)
        
        self.attend = Attend(
            causal=True,
            **attend_kwargs
        )
        
        if num_persist_mem_tokens > 0:
            self.persist_mem = nn.Parameter(torch.randn(heads, num_persist_mem_tokens, dim_head))
        else:
            self.persist_mem = None
    
    def forward(self, x, cache=None, return_cache=False, extra_kv=None):
        batch, seq_len, _ = x.shape
        
        x = self.norm(x)
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q, k, v = map(self.split_heads, (q, k, v))
        
        # Rotary embeddings
        if exists(cache):
            cached_k, cached_v = cache
            cache_len = cached_k.shape[2]
        else:
            cache_len = 0
        
        # Apply rotary embeddings
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        
        # Update cache
        if exists(cache):
            k = cat([cached_k, k], dim=2)
            v = cat([cached_v, v], dim=2)
        
        # Sliding window
        if k.shape[2] > self.window_size:
            k = k[:, :, -self.window_size:]
            v = v[:, :, -self.window_size:]
        
        # Persistent memory tokens
        if exists(self.persist_mem):
            pm = repeat(self.persist_mem, 'h n d -> b h n d', b=batch)
            k = cat([pm, k], dim=2)
            v = cat([pm, v], dim=2)

        # Optional extra_kv (e.g., memory context tokens)
        if exists(extra_kv):
            k_extra, v_extra = extra_kv
            k = cat([k_extra, k], dim=2)
            v = cat([v_extra, v], dim=2)
        
        # Attention
        out = self.attend(q, k, v)
        if isinstance(out, tuple):
            out = out[0]
        out = self.merge_heads(out)
        out = self.to_out(out)
        
        if return_cache:
            persist_len = self.persist_mem.shape[1] if exists(self.persist_mem) else 0
            extra_len = 0
            if exists(extra_kv):
                k_extra, _ = extra_kv
                extra_len = k_extra.shape[2]
            start = persist_len + extra_len
            k_base = k[:, :, start:]
            v_base = v[:, :, start:]
            if k_base.shape[2] > self.window_size:
                k_base = k_base[:, :, -self.window_size:]
                v_base = v_base[:, :, -self.window_size:]
            new_cache = (k_base, v_base)
            return out, new_cache
        
        return out


class SegmentedAttention(Module):
    """
    Segmented attention for MAC architecture.
    
    Full causal attention within each segment, no sliding window.
    Segment boundaries act as hard attention boundaries.
    """
    
    def __init__(
        self,
        dim,
        segment_len,
        num_persist_mem_tokens=0,
        num_longterm_mem_tokens=0,
        dim_head=64,
        heads=8,
        attend_kwargs: dict = dict(),
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.heads = heads
        self.segment_len = segment_len
        self.window_size = segment_len  # Alias for compatibility with MACBlock
        self.num_persist_mem_tokens = num_persist_mem_tokens
        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        self.total_segment_len = segment_len + num_longterm_mem_tokens
        
        dim_inner = dim_head * heads
        
        self.to_qkv = LinearNoBias(dim, dim_inner * 3)
        self.to_out = LinearNoBias(dim_inner, dim)
        
        self.split_heads = Rearrange('b n (h d) -> b h n d', h=heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        
        self.rotary_emb = RotaryEmbedding(dim_head)
        
        self.attend = Attend(
            causal=True,
            **attend_kwargs
        )
        
        if num_persist_mem_tokens > 0:
            self.persist_mem = nn.Parameter(torch.randn(heads, num_persist_mem_tokens, dim_head))
        else:
            self.persist_mem = None
    
    def forward(self, x, cache=None, return_cache=False, extra_kv=None):
        """
        Forward pass with segmented attention.
        
        For training: process full sequence with segment-wise attention mask.
        For inference (with cache): process single token with cached segment context.
        """
        batch, seq_len, _ = x.shape
        
        x = self.norm(x)
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q, k, v = map(self.split_heads, (q, k, v))
        
        # Rotary embeddings
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        
        # Handle cache for inference
        if exists(cache):
            cached_k, cached_v = cache
            k = cat([cached_k, k], dim=2)
            v = cat([cached_v, v], dim=2)
        
        # Persistent memory tokens
        if exists(self.persist_mem):
            pm = repeat(self.persist_mem, 'h n d -> b h n d', b=batch)
            k_with_pm = cat([pm, k], dim=2)
            v_with_pm = cat([pm, v], dim=2)
        else:
            k_with_pm = k
            v_with_pm = v
        
        # Optional extra_kv (memory context tokens)
        if exists(extra_kv):
            k_extra, v_extra = extra_kv
            k_with_pm = cat([k_extra, k_with_pm], dim=2)
            v_with_pm = cat([v_extra, v_with_pm], dim=2)
        
        # Create segmented attention mask (for training)
        attend_kwargs = dict()
        if not exists(cache) and seq_len > 1:
            device = x.device
            q_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len, 1)
            
            # Calculate kv length
            persist_len = self.persist_mem.shape[1] if exists(self.persist_mem) else 0
            extra_len = extra_kv[0].shape[2] if exists(extra_kv) else 0
            kv_len = k_with_pm.shape[2]
            k_idx = torch.arange(kv_len, device=device).view(1, 1, 1, -1)
            
            # Causal mask
            causal_mask = q_idx >= (k_idx - persist_len - extra_len)
            
            # Segment mask: tokens can only attend within their segment
            q_segment = q_idx // self.segment_len
            k_segment = (k_idx - persist_len - extra_len) // self.segment_len
            segment_mask = q_segment == k_segment
            
            # Persistent memory and extra_kv are always visible
            always_visible = k_idx < (persist_len + extra_len)
            
            mask = (causal_mask & segment_mask) | always_visible
            attend_kwargs['mask'] = mask
        
        # Attention
        out = self.attend(q, k_with_pm, v_with_pm, **attend_kwargs)
        if isinstance(out, tuple):
            out = out[0]
        out = self.merge_heads(out)
        out = self.to_out(out)
        
        if return_cache:
            # Keep only the last segment for cache
            persist_len = self.persist_mem.shape[1] if exists(self.persist_mem) else 0
            extra_len = extra_kv[0].shape[2] if exists(extra_kv) else 0
            start = persist_len + extra_len
            k_base = k_with_pm[:, :, start:]
            v_base = v_with_pm[:, :, start:]
            # Trim to segment length
            if k_base.shape[2] > self.total_segment_len:
                k_base = k_base[:, :, -self.total_segment_len:]
                v_base = v_base[:, :, -self.total_segment_len:]
            new_cache = (k_base, v_base)
            return out, new_cache
        
        return out
    
    def forward(self, x, cache=None, return_cache=False, extra_kv=None):
        batch, seq_len, _ = x.shape
        
        x = self.norm(x)
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q, k, v = map(self.split_heads, (q, k, v))
        
        # Rotary embeddings
        if exists(cache):
            cached_k, cached_v = cache
            cache_len = cached_k.shape[2]
        else:
            cache_len = 0
        
        # Apply rotary embeddings
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        
        # Update cache
        if exists(cache):
            k = cat([cached_k, k], dim=2)
            v = cat([cached_v, v], dim=2)
        
        # Sliding window
        if k.shape[2] > self.window_size:
            k = k[:, :, -self.window_size:]
            v = v[:, :, -self.window_size:]
        
        # Persistent memory tokens
        if exists(self.persist_mem):
            pm = repeat(self.persist_mem, 'h n d -> b h n d', b=batch)
            k = cat([pm, k], dim=2)
            v = cat([pm, v], dim=2)

        # Optional extra_kv (e.g., memory context tokens)
        if exists(extra_kv):
            k_extra, v_extra = extra_kv
            k = cat([k_extra, k], dim=2)
            v = cat([v_extra, v], dim=2)
        
        # Attention
        out = self.attend(q, k, v)
        if isinstance(out, tuple):
            out = out[0]
        out = self.merge_heads(out)
        out = self.to_out(out)
        
        if return_cache:
            persist_len = self.persist_mem.shape[1] if exists(self.persist_mem) else 0
            extra_len = 0
            if exists(extra_kv):
                k_extra, _ = extra_kv
                extra_len = k_extra.shape[2]
            start = persist_len + extra_len
            k_base = k[:, :, start:]
            v_base = v[:, :, start:]
            if k_base.shape[2] > self.window_size:
                k_base = k_base[:, :, -self.window_size:]
                v_base = v_base[:, :, -self.window_size:]
            new_cache = (k_base, v_base)
            return out, new_cache
        
        return out

# ============================================================================
# MAG: Memory As Gate
# ============================================================================

class MAGBlock(Module):
    """
    Memory-As-Gate block using RNN memory.
    """
    
    def __init__(
        self,
        dim,
        window_size=64,
        dim_head=64,
        heads=8,
        ff_mult=4,
        num_persist_mem_tokens=0,
        omega_window=1,
        use_omega_gate=False,
        use_momentum=True,
        poly_degree=1,
        poly_mode='off',
        qkv_conv_kernel=4,
    ):
        super().__init__()
        
        self.attention = SlidingWindowAttention(
            dim=dim,
            window_size=window_size,
            num_persist_mem_tokens=num_persist_mem_tokens,
            dim_head=dim_head,
            heads=heads,
        )
        
        if omega_window > 1 or use_omega_gate:
            self.memory = OmegaRNNMemoryCell(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                omega_window=omega_window,
                use_omega_gate=use_omega_gate,
                use_momentum=use_momentum,
                poly_degree=poly_degree,
                poly_mode=poly_mode,
                qkv_conv_kernel=qkv_conv_kernel,
            )
        else:
            self.memory = RNNMemoryCell(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                use_momentum=use_momentum,
                poly_degree=poly_degree,
                poly_mode=poly_mode,
                qkv_conv_kernel=qkv_conv_kernel,
            )
        
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        self.ff = FeedForward(dim, mult=ff_mult)
    
    def forward(self, x, mem_state=None, attn_cache=None, return_cache=False):
        if return_cache:
            attn_out, new_attn_cache = self.attention(x, cache=attn_cache, return_cache=True)
        else:
            attn_out = self.attention(x, cache=attn_cache)
            new_attn_cache = None
        
        mem_out, new_mem_state = self.memory(x, state=mem_state)
        
        g = self.gate(x)
        combined = g * attn_out + (1 - g) * mem_out
        
        x = x + combined
        x = x + self.ff(x)
        
        if return_cache:
            return x, new_mem_state, new_attn_cache
        return x, new_mem_state

# ============================================================================
# MAC: Memory As Context
# ============================================================================

class MACBlock(Module):
    """
    Memory-As-Context block using RNN memory.
    """
    def __init__(
        self,
        dim,
        window_size=64,
        dim_head=64,
        heads=8,
        ff_mult=4,
        num_persist_mem_tokens=0,
        num_longterm_mem_tokens=4,
        segment_len=16,
        neural_memory_segment_len=1,
        omega_window=1,
        use_omega_gate=False,
        use_momentum=True,
        poly_degree=1,
        poly_mode='off',
        qkv_conv_kernel=4,
    ):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        self.segment_len = segment_len
        self.neural_memory_segment_len = neural_memory_segment_len

        if omega_window > 1 or use_omega_gate:
            self.memory = OmegaRNNMemoryCell(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                omega_window=omega_window,
                use_omega_gate=use_omega_gate,
                use_momentum=use_momentum,
                poly_degree=poly_degree,
                poly_mode=poly_mode,
                qkv_conv_kernel=qkv_conv_kernel,
            )
        else:
            self.memory = RNNMemoryCell(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                use_momentum=use_momentum,
                poly_degree=poly_degree,
                poly_mode=poly_mode,
                qkv_conv_kernel=qkv_conv_kernel,
            )

        # MAC uses SegmentedAttention (full causal within segment), not sliding window
        self.attention = SegmentedAttention(
            dim=dim,
            segment_len=segment_len,
            num_persist_mem_tokens=num_persist_mem_tokens,
            num_longterm_mem_tokens=num_longterm_mem_tokens,
            dim_head=dim_head,
            heads=heads,
        )

        self.mem_pool_norm = nn.RMSNorm(dim)
        self.to_mem_tokens = LinearNoBias(dim, dim)

        self.ff = FeedForward(dim, mult=ff_mult)

        self._mac_committed_state = None
        self._mac_seq_index = 0

    def _build_mem_context(self, mem_out, b):
        pooled = mem_out.mean(dim=1)
        pooled = self.mem_pool_norm(pooled)
        mem_tokens = self.to_mem_tokens(pooled)
        mem_tokens = repeat(mem_tokens, 'b d -> b m d', m=self.num_longterm_mem_tokens)

        qkv_ctx = self.attention.to_qkv(mem_tokens)
        q_ctx, k_ctx, v_ctx = qkv_ctx.chunk(3, dim=-1)
        k_ctx = self.attention.split_heads(k_ctx)
        v_ctx = self.attention.split_heads(v_ctx)
        return k_ctx, v_ctx

    def forward(self, x, mem_state=None, attn_cache=None, return_cache=False):
        b = x.shape[0]
        n = x.shape[1]

        committed_state = self._mac_committed_state if exists(self._mac_committed_state) else mem_state

        mem_out, staged_state = self.memory(x, state=committed_state)

        k_extra, v_extra = self._build_mem_context(mem_out, b)

        if return_cache:
            attn_out, new_attn_cache = self.attention(x, cache=attn_cache, return_cache=True, extra_kv=(k_extra, v_extra))
        else:
            attn_out = self.attention(x, cache=attn_cache, return_cache=False, extra_kv=(k_extra, v_extra))
            new_attn_cache = None

        x = x + attn_out
        x = x + self.ff(x)

        self._mac_seq_index += n
        if self.segment_len > 0 and (self._mac_seq_index % self.segment_len) == 0:
            self._mac_committed_state = staged_state

        if return_cache:
            return x, committed_state, new_attn_cache
        return x, committed_state

# ============================================================================
# MAL: Memory As Layer
# ============================================================================

class MALBlock(Module):
    """
    Memory-As-Layer block using RNN memory.
    """
    
    def __init__(
        self,
        dim,
        window_size=64,
        dim_head=64,
        heads=8,
        ff_mult=4,
        num_persist_mem_tokens=0,
        omega_window=1,
        use_omega_gate=False,
        use_momentum=True,
        poly_degree=1,
        poly_mode='off',
        qkv_conv_kernel=4,
    ):
        super().__init__()
        
        if omega_window > 1 or use_omega_gate:
            self.memory = OmegaRNNMemoryCell(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                omega_window=omega_window,
                use_omega_gate=use_omega_gate,
                use_momentum=use_momentum,
                poly_degree=poly_degree,
                poly_mode=poly_mode,
                qkv_conv_kernel=qkv_conv_kernel,
            )
        else:
            self.memory = RNNMemoryCell(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                use_momentum=use_momentum,
                poly_degree=poly_degree,
                poly_mode=poly_mode,
                qkv_conv_kernel=qkv_conv_kernel,
            )
        
        self.attention = SlidingWindowAttention(
            dim=dim,
            window_size=window_size,
            num_persist_mem_tokens=num_persist_mem_tokens,
            dim_head=dim_head,
            heads=heads,
        )
        
        self.ff_pre = FeedForward(dim, mult=ff_mult)
        self.ff_post = FeedForward(dim, mult=ff_mult)
    
    def forward(self, x, mem_state=None, attn_cache=None, return_cache=False):
        mem_out, new_mem_state = self.memory(x, state=mem_state)
        x = x + mem_out
        x = x + self.ff_pre(x)
        
        if return_cache:
            attn_out, new_attn_cache = self.attention(x, cache=attn_cache, return_cache=True)
        else:
            attn_out = self.attention(x, cache=attn_cache)
            new_attn_cache = None
        x = x + attn_out
        
        x = x + self.ff_post(x)
        
        if return_cache:
            return x, new_mem_state, new_attn_cache
        return x, new_mem_state

# ============================================================================
# LMM: Long-term Memory Model (Pure RNN Memory)
# ============================================================================

class LMMBlock(Module):
    """
    Long-term Memory Model block (pure RNN memory, no attention).
    """
    
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        omega_window=1,
        use_omega_gate=False,
        use_momentum=True,
        poly_degree=1,
        poly_mode='off',
        qkv_conv_kernel=None,
    ):
        super().__init__()
        
        if omega_window > 1 or use_omega_gate:
            self.memory = OmegaRNNMemoryCell(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                omega_window=omega_window,
                use_omega_gate=use_omega_gate,
                use_momentum=use_momentum,
                poly_degree=poly_degree,
                poly_mode=poly_mode,
                qkv_conv_kernel=qkv_conv_kernel,
            )
        else:
            self.memory = RNNMemoryCell(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                use_momentum=use_momentum,
                poly_degree=poly_degree,
                poly_mode=poly_mode,
                qkv_conv_kernel=qkv_conv_kernel,
            )
        
        self.ff = FeedForward(dim, mult=ff_mult)
    
    def forward(self, x, mem_state=None):
        mem_out, new_mem_state = self.memory(x, state=mem_state)
        x = x + mem_out
        x = x + self.ff(x)
        return x, new_mem_state

# ============================================================================
# Full Transformers
# ============================================================================

class RNNMemoryTransformer(Module):
    """
    Base transformer class using RNN memory blocks.
    """
    
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        block_type: str = 'mag',
        window_size: int = 64,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: float = 4,
        num_persist_mem_tokens: int = 0,
        omega_window: int = 1,
        use_omega_gate: bool = False,
        use_momentum: bool = True,
        poly_degree: int = 1,
        poly_mode: str = 'off',
        qkv_conv_kernel: int | None = 4,
        # Legacy parameters (ignored for compatibility)
        mem_chunk_size: int = 1,
        use_accelerated_scan: bool = False,
    ):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.dim = dim
        self.depth = depth
        self.block_type = block_type
        
        self.embed = nn.Embedding(num_tokens, dim)
        self.embed_norm = nn.LayerNorm(dim)
        
        self.layers = ModuleList()
        
        block_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            omega_window=omega_window,
            use_omega_gate=use_omega_gate,
            use_momentum=use_momentum,
            poly_degree=poly_degree,
            poly_mode=poly_mode,
            qkv_conv_kernel=qkv_conv_kernel,
        )
        
        for _ in range(depth):
            if block_type == 'mag':
                block = MAGBlock(
                    window_size=window_size,
                    num_persist_mem_tokens=num_persist_mem_tokens,
                    **block_kwargs
                )
            elif block_type == 'mal':
                block = MALBlock(
                    window_size=window_size,
                    num_persist_mem_tokens=num_persist_mem_tokens,
                    **block_kwargs
                )
            elif block_type == 'mac':
                block = MACBlock(
                    window_size=window_size,
                    num_persist_mem_tokens=num_persist_mem_tokens,
                    num_longterm_mem_tokens=4,
                    **block_kwargs
                )
            elif block_type == 'lmm':
                block = LMMBlock(**block_kwargs)
            else:
                raise ValueError(f"Unknown block type: {block_type}")
            
            self.layers.append(block)
        
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.to_logits.weight, std=0.02)
    
    def forward(
        self,
        x,
        return_loss: bool = False,
        mem_states=None,
        attn_caches=None,
        return_cache: bool = False,
    ):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]
        
        batch, seq_len = x.shape
        
        h = self.embed(x)
        h = self.embed_norm(h)
        
        if mem_states is None:
            mem_states = [None] * self.depth
        if attn_caches is None:
            attn_caches = [None] * self.depth
        
        new_mem_states = []
        new_attn_caches = []
        
        for layer, mem_state, attn_cache in zip(self.layers, mem_states, attn_caches):
            if self.block_type == 'lmm':
                h, new_mem = layer(h, mem_state=mem_state)
                new_mem_states.append(new_mem)
                new_attn_caches.append(None)
            else:
                if return_cache:
                    h, new_mem, new_attn = layer(
                        h, mem_state=mem_state, attn_cache=attn_cache, return_cache=True
                    )
                else:
                    h, new_mem = layer(h, mem_state=mem_state, attn_cache=attn_cache)
                    new_attn = None
                new_mem_states.append(new_mem)
                new_attn_caches.append(new_attn)
        
        h = self.norm(h)
        logits = self.to_logits(h)
        
        if return_loss:
            loss = F.cross_entropy(
                logits.view(-1, self.num_tokens),
                labels.reshape(-1)
            )
            if return_cache:
                return loss, logits, new_mem_states, new_attn_caches
            return loss
        
        if return_cache:
            return logits, new_mem_states, new_attn_caches
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_length: int = 100,
        temperature: float = 1.0,
        min_p: float = 0.1,
        use_cache: bool = True,
    ):
        """Generate tokens autoregressively."""
        self.eval()
        
        full_seq = prompt.clone()
        x = prompt
        mem_states = None
        attn_caches = None
        
        for _ in tqdm.tqdm(range(max_length - prompt.shape[1]), desc='Generating'):
            if use_cache:
                logits, mem_states, attn_caches = self(
                    x, mem_states=mem_states, attn_caches=attn_caches, return_cache=True
                )
                logits = logits[:, -1:]
            else:
                logits = self(full_seq)
                logits = logits[:, -1:]
            
            logits = min_p_filter(logits, min_p=min_p)
            next_token = gumbel_sample(logits, temperature=temperature)
            
            full_seq = cat([full_seq, next_token.squeeze(-1)], dim=-1)
            
            if use_cache:
                x = next_token.squeeze(-1)
            else:
                x = full_seq
        
        return full_seq

# ============================================================================
# Convenience aliases
# ============================================================================

def MemoryAsGateTransformer(**kwargs):
    return RNNMemoryTransformer(block_type='mag', **kwargs)

def MemoryAsLayerTransformer(**kwargs):
    return RNNMemoryTransformer(block_type='mal', **kwargs)

def LongTermMemoryModel(**kwargs):
    return RNNMemoryTransformer(block_type='lmm', **kwargs)
