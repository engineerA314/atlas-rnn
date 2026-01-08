"""
RNN Memory: Explicit RNN-form implementation of Titans and OmegaNet memory.

This module derives closed-form RNN update rules from the implicit gradient descent
formulation in the Titans and Atlas papers.

Key equations (per-token semantics):
- Surprise: δ_t = S_{t-1} φ_t - v_t
- Gradient: g_t = δ_t φ_t^T
- Momentum: Z_t = β_t Z_{t-1} + g_t
- Update: S_t = α_t S_{t-1} - η_t Z_t
- Retrieval: y_t = S_{t-1} ψ_t

Parallelization via efficient scalar scan (using assoc_scan library):
- Compute gradients using fixed S_0 (chunk-start state) for all tokens in parallel
- Apply scalar-gated recurrence: S_t = α_t * S_{t-1} + δ_t
- Same algorithm as original Atlas paper
"""

from __future__ import annotations
from typing import NamedTuple
from functools import partial

import math
import warnings
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module, Parameter, Linear, Conv1d

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from assoc_scan import AssocScan


# ============================================================================
# Helper functions
# ============================================================================

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

LinearNoBias = partial(Linear, bias=False)


def _sliding_sum_along_time(x: Tensor, window: int) -> Tensor:
    """
    Sliding window sum along dim=1 (time), inclusive.
    x: [B, T, ...] -> y_t = sum_{k=max(0,t-window+1)}^t x_k
    """
    if window <= 1:
        return x
    T = x.shape[1]
    window = min(window, T)
    c = x.cumsum(dim=1)
    shifted = torch.cat([c.new_zeros((*c.shape[:1], window, *c.shape[2:])), c[:, :-window]], dim=1)
    return c - shifted


# ============================================================================
# RNN Memory State
# ============================================================================

class RNNMemState(NamedTuple):
    """State for RNN memory."""
    seq_index: int              # Current sequence position
    S: Tensor                   # Memory state [batch*heads, d_v, d_phi]
    Z: Tensor | None            # Momentum state (optional)
    omega_buffer: Tensor | None # Rolling buffer for Omega window (optional)


def state_detach(state: RNNMemState) -> RNNMemState:
    """Detach all tensors in state from computation graph."""
    return RNNMemState(
        seq_index=state.seq_index,
        S=state.S.detach() if exists(state.S) else None,
        Z=state.Z.detach() if exists(state.Z) else None,
        omega_buffer=state.omega_buffer.detach() if exists(state.omega_buffer) else None,
    )


# ============================================================================
# Polynomial Feature Maps
# ============================================================================

class PolynomialFeatureMap(Module):
    """
    Polynomial feature expansion for keys and queries.
    
    Modes:
    - 'off': φ(x) = x
    - 'elementwise': φ(x) = Σ_{i=1}^{g} x^i
    - 'tensor': φ(x) = RandomProj([x, vec(x ⊗ x)])
    """
    
    def __init__(self, dim: int, degree: int = 1, mode: str = 'off'):
        super().__init__()
        self.dim = dim
        self.degree = degree
        self.mode = mode
        
        if mode == 'tensor':
            feat_dim = dim + dim * dim
            self.register_buffer(
                'proj',
                torch.randn(feat_dim, dim) / math.sqrt(feat_dim)
            )
    
    def forward(self, x: Tensor) -> Tensor:
        if self.degree <= 1 or self.mode == 'off':
            return x
        
        if self.mode == 'elementwise':
            out = x.clone()
            power = x
            for _ in range(2, self.degree + 1):
                power = power * x
                out = out + power
            return out
        
        if self.mode == 'tensor':
            d = x.shape[-1]
            outer = torch.einsum('...i,...j->...ij', x, x)
            outer = outer.reshape(*x.shape[:-1], d * d)
            feats = torch.cat([x, outer], dim=-1)
            return feats @ self.proj.to(x.device, x.dtype)
        
        return x


# ============================================================================
# Multi-head RMS Norm
# ============================================================================

class MultiheadRMSNorm(Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** -0.5
        self.gamma = Parameter(torch.ones(heads, 1, dim))
    
    def forward(self, x: Tensor) -> Tensor:
        rms = x.norm(dim=-1, keepdim=True) * self.scale
        return (x / rms.clamp(min=1e-8)) * self.gamma


# ============================================================================
# RNN Memory Cell (Titans-RNN) - Per-token semantics with scalar scan
# ============================================================================

class RNNMemoryCell(Module):
    """
    Per-token RNN memory update with scalar scan parallelization.
    
    Implements (per-token):
        g_t = (S_0 φ_t - v_t) φ_t^T              (gradient using fixed S_0)
        Z_t = β_t Z_{t-1} + g_t                  (momentum)
        S_t = α_t S_{t-1} - η_t Z_t              (memory update)
        y_t = S_{t-1} ψ_t                        (retrieval)
    
    Parallelized via scalar scan using assoc_scan library.
    """
    
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 1,
        use_momentum: bool = True,
        poly_degree: int = 1,
        poly_mode: str = 'off',
        qk_norm: bool = True,
        qkv_conv_kernel: int | None = 4,
        use_accelerated_scan: bool = False,
        scan_chunk_len: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.use_momentum = use_momentum
        self.qkv_conv_kernel = qkv_conv_kernel
        self.scan_chunk_len = scan_chunk_len
        
        # Associative scan for parallel computation (use_accelerated=True requires Triton)
        self.assoc_scan = AssocScan(use_accelerated=use_accelerated_scan)
        
        dim_inner = dim_head * heads
        
        # Projections
        self.activation = nn.Identity()
        self.to_q = LinearNoBias(dim, dim_inner)
        self.to_k = LinearNoBias(dim, dim_inner)
        self.to_v = LinearNoBias(dim, dim_inner)
        self.to_out = LinearNoBias(dim_inner, dim)

        # Optional depthwise conv for Q/K/V
        if exists(qkv_conv_kernel) and qkv_conv_kernel > 1:
            padding = qkv_conv_kernel // 2
            self.q_conv = Conv1d(dim, dim, qkv_conv_kernel, padding=padding, groups=dim, bias=False)
            self.k_conv = Conv1d(dim, dim, qkv_conv_kernel, padding=padding, groups=dim, bias=False)
            self.v_conv = Conv1d(dim, dim, qkv_conv_kernel, padding=padding, groups=dim, bias=False)
        else:
            self.q_conv = None
            self.k_conv = None
            self.v_conv = None

        # Learned hyperparameters (per-head)
        self.to_lr = nn.Sequential(Linear(dim, heads), nn.Sigmoid())
        nn.init.zeros_(self.to_lr[0].weight)
        nn.init.constant_(self.to_lr[0].bias, -4.0)  # sigmoid(-4) ≈ 0.018
        
        self.to_decay = nn.Sequential(Linear(dim, heads), nn.Sigmoid())
        nn.init.zeros_(self.to_decay[0].weight)
        nn.init.constant_(self.to_decay[0].bias, 4.0)  # sigmoid(4) ≈ 0.982
        
        self.to_momentum = nn.Sequential(Linear(dim, heads), nn.Sigmoid()) if use_momentum else None
        if use_momentum:
            nn.init.zeros_(self.to_momentum[0].weight)
            nn.init.constant_(self.to_momentum[0].bias, 2.0)  # sigmoid(2) ≈ 0.88
        
        # Norms
        self.pre_norm = nn.RMSNorm(dim)
        self.q_norm = MultiheadRMSNorm(dim_head, heads) if qk_norm else nn.Identity()
        self.k_norm = MultiheadRMSNorm(dim_head, heads) if qk_norm else nn.Identity()
        
        # Polynomial feature map
        self.phi = PolynomialFeatureMap(dim_head, poly_degree, poly_mode)
        
        # Head operations
        self.split_heads = Rearrange('b n (h d) -> b h n d', h=heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        
        self.register_buffer('_dummy', torch.zeros(1))
    
    def init_state(self, batch: int, device=None, dtype=None) -> RNNMemState:
        """Initialize memory state."""
        device = default(device, self._dummy.device)
        dtype = default(dtype, self._dummy.dtype)
        
        S = torch.zeros(batch * self.heads, self.dim_head, self.dim_head, 
                       device=device, dtype=dtype)
        Z = torch.zeros_like(S) if self.use_momentum else None
        
        return RNNMemState(seq_index=0, S=S, Z=Z, omega_buffer=None)

    def forward(
        self,
        x: Tensor,
        state: RNNMemState | None = None,
    ) -> tuple[Tensor, RNNMemState]:
        """
        Forward pass with efficient scalar scan parallelization.
        
        Uses fixed S_0 (chunk-start state) for gradient computation,
        enabling 6x memory reduction compared to matrix affine scan.
        Same approximation as the original Atlas paper.
        """
        batch, seq_len, _ = x.shape

        if not exists(state):
            state = self.init_state(batch, x.device, x.dtype)

        # Pre-norm + activation (used for projections + scalar gates)
        x_act = self.activation(self.pre_norm(x))

        # Optional depthwise conv (compute on full seq once; safe to slice later)
        q_in = k_in = v_in = x_act
        if exists(self.q_conv):
            q_in = self.q_conv(q_in.transpose(1, 2)).transpose(1, 2)
        if exists(self.k_conv):
            k_in = self.k_conv(k_in.transpose(1, 2)).transpose(1, 2)
        if exists(self.v_conv):
            v_in = self.v_conv(v_in.transpose(1, 2)).transpose(1, 2)

        chunk_len = self.scan_chunk_len
        if exists(chunk_len) and seq_len > chunk_len:
            outs: list[Tensor] = []
            cur_state = state
            # Preserve current (unchunked) semantics: g is computed from the reference state at call start.
            S_ref = cur_state.S
            for start in range(0, seq_len, chunk_len):
                end = min(seq_len, start + chunk_len)
                out, cur_state = self._forward_impl(
                    x_act[:, start:end],
                    q_in[:, start:end],
                    k_in[:, start:end],
                    v_in[:, start:end],
                    cur_state,
                    S_ref=S_ref,
                )
                outs.append(out)
            return torch.cat(outs, dim=1), cur_state

        return self._forward_impl(x_act, q_in, k_in, v_in, state, S_ref=state.S)

    def _forward_impl(
        self,
        x_act: Tensor,
        q_in: Tensor,
        k_in: Tensor,
        v_in: Tensor,
        state: RNNMemState,
        *,
        S_ref: Tensor,
    ) -> tuple[Tensor, RNNMemState]:
        batch, seq_len, _ = x_act.shape

        d = self.dim_head

        # Project to Q, K, V and split heads: [batch, heads, seq, dim_head]
        q = self.split_heads(self.to_q(q_in))
        k = self.split_heads(self.to_k(k_in))
        v = self.split_heads(self.to_v(v_in))

        # Truncate to original seq_len in case conv changed length
        q = q[:, :, :seq_len]
        k = k[:, :, :seq_len]
        v = v[:, :, :seq_len]

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply polynomial feature map
        k_flat = rearrange(k, 'b h n d -> (b h n) d')
        q_flat = rearrange(q, 'b h n d -> (b h n) d')

        phi_k = rearrange(self.phi(k_flat), '(b h n) d -> (b h) n d', b=batch, h=self.heads, n=seq_len)
        phi_q = rearrange(self.phi(q_flat), '(b h n) d -> (b h) n d', b=batch, h=self.heads, n=seq_len)
        v_bh = rearrange(v, 'b h n d -> (b h) n d')

        # Learned hyperparameters: [BH, T]
        lr = rearrange(self.to_lr(x_act), 'b n h -> (b h) n')
        decay = rearrange(self.to_decay(x_act), 'b n h -> (b h) n')

        if self.use_momentum:
            momentum = rearrange(self.to_momentum(x_act), 'b n h -> (b h) n')

        # -----------------------------------------------------------------
        # Efficient Scalar Scan Implementation (Exact Refactor)
        # -----------------------------------------------------------------
        # Step 1: Compute per-token delta vector (avoids materializing G, B)
        # δ_t = S_ref @ φ_t - v_t (prediction error)
        pred = torch.einsum('bde,bte->btd', S_ref, phi_k)  # [BH, T, d]
        delta_vec = pred - v_bh                             # [BH, T, d]
        
        # Step 2: Compute gradient via outer product
        # g_t = δ_t ⊗ φ_t^T (exact same math as S_ref@(φφ^T) - v⊗φ^T)
        g = torch.einsum('bti,btj->btij', delta_vec, phi_k)  # [BH, T, d, d]

        # Get initial states (for recurrence)
        S0 = state.S  # [BH, d, d]
        Z0 = state.Z if exists(state.Z) else torch.zeros_like(S0)

        # Expand scalars for broadcasting: [BH, T, 1, 1]
        lr_e = lr[..., None, None]

        if self.use_momentum:
            # Step 3: Momentum via scalar scan: Z_t = β_t * Z_{t-1} + g_t
            Z_all = self.assoc_scan(momentum, g, prev=Z0)  # [BH, T, d, d]

            # Step 4: Compute delta: δ_t = -η_t * Z_t
            delta = -lr_e * Z_all  # [BH, T, d, d]

            # Step 5: State update via scalar scan: S_t = α_t * S_{t-1} + δ_t
            S_all = self.assoc_scan(decay, delta, prev=S0)  # [BH, T, d, d]

            # Final states
            S_end = S_all[:, -1].clamp(-100, 100)
            Z_end = Z_all[:, -1]
        else:
            # No momentum: δ_t = -η_t * g_t
            delta = -lr_e * g  # [BH, T, d, d]

            # State update via scalar scan: S_t = α_t * S_{t-1} + δ_t
            S_all = self.assoc_scan(decay, delta, prev=S0)  # [BH, T, d, d]

            S_end = S_all[:, -1].clamp(-100, 100)
            Z_end = None

        # -----------------------------------------------------------------
        # Retrieval: y_t = S_{t-1} @ ψ_t
        # -----------------------------------------------------------------
        # Avoid materializing S_start=[S0, S_all[:, :-1]] which is [BH, T, d, d].
        y0 = torch.einsum('bde,be->bd', S0, phi_q[:, 0])  # [BH, d]
        if seq_len > 1:
            y_rest = torch.einsum('btdp,btp->btd', S_all[:, :-1], phi_q[:, 1:])  # [BH, T-1, d]
            retrieved = torch.cat([y0.unsqueeze(1), y_rest], dim=1)  # [BH, T, d]
        else:
            retrieved = y0.unsqueeze(1)  # [BH, 1, d]

        # Reshape to [batch, heads, seq, dim_head] and merge
        retrieved = rearrange(retrieved, '(b h) t d -> b h t d', b=batch, h=self.heads)
        retrieved = self.merge_heads(retrieved)
        retrieved = self.to_out(retrieved)

        # Build next state
        next_state = RNNMemState(
            seq_index=state.seq_index + seq_len,
            S=S_end,
            Z=Z_end,
            omega_buffer=None
        )

        return retrieved, next_state


# ============================================================================
# Omega RNN Memory Cell (OmegaNet-RNN) - Sliding window context
# ============================================================================

class OmegaRNNMemoryCell(Module):
    """
    RNN memory with Omega rule (sliding window context).
    
    For omega_window > 1:
        G_t = Σ_{p∈W_t} U_t^p φ_p φ_p^T   (Gram matrix over window)
        B_t = Σ_{p∈W_t} U_t^p v_p φ_p^T   (Cross term)
    
    Parallelized via scalar scan using assoc_scan library.
    """
    
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 1,
        omega_window: int = 1,
        use_omega_gate: bool = True,
        use_momentum: bool = True,
        poly_degree: int = 1,
        poly_mode: str = 'off',
        qk_norm: bool = True,
        qkv_conv_kernel: int | None = 4,
        use_accelerated_scan: bool = False,
        scan_chunk_len: int | None = None,
        use_cuda: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.omega_window = omega_window
        self.use_omega_gate = use_omega_gate
        self.use_momentum = use_momentum
        self.qkv_conv_kernel = qkv_conv_kernel
        self.scan_chunk_len = scan_chunk_len
        self.use_cuda = use_cuda
        
        # CUDA backend setup (only if omega_window == 16 for now)
        self.cuda_available = False
        if use_cuda:
            if omega_window != 16:
                warnings.warn(f"CUDA backend only supports omega_window=16, got {omega_window}. Falling back to PyTorch.")
                self.use_cuda = False
            else:
                try:
                    from atlas_pytorch.cuda.atlas_omega_autograd import (
                        check_cuda_availability, AtlasOmegaFunction
                    )
                    available, msg = check_cuda_availability()
                    if not available:
                        warnings.warn(f"CUDA not available: {msg}. Falling back to PyTorch.")
                        self.use_cuda = False
                    else:
                        self.AtlasOmegaFunction = AtlasOmegaFunction
                        self.cuda_available = True
                except ImportError as e:
                    warnings.warn(f"CUDA extension not compiled: {e}. Falling back to PyTorch.")
                    self.use_cuda = False

        # Associative scan for parallel computation (use_accelerated=True requires Triton).
        # If CUDA backend is active, OmegaRNNMemoryCell does not use assoc_scan at all,
        # so we avoid initializing it (and avoid optional-triton warnings) in that case.
        self.assoc_scan = None
        if not (self.use_cuda and self.cuda_available):
            self.assoc_scan = AssocScan(use_accelerated=use_accelerated_scan)
        
        dim_inner = dim_head * heads
        
        # Projections
        self.activation = nn.Identity()
        self.to_q = LinearNoBias(dim, dim_inner)
        self.to_k = LinearNoBias(dim, dim_inner)
        self.to_v = LinearNoBias(dim, dim_inner)
        self.to_out = LinearNoBias(dim_inner, dim)

        # Optional depthwise conv
        if exists(qkv_conv_kernel) and qkv_conv_kernel > 1:
            padding = qkv_conv_kernel // 2
            self.q_conv = Conv1d(dim, dim, qkv_conv_kernel, padding=padding, groups=dim, bias=False)
            self.k_conv = Conv1d(dim, dim, qkv_conv_kernel, padding=padding, groups=dim, bias=False)
            self.v_conv = Conv1d(dim, dim, qkv_conv_kernel, padding=padding, groups=dim, bias=False)
        else:
            self.q_conv = None
            self.k_conv = None
            self.v_conv = None
        
        # Learned hyperparameters
        self.to_lr = nn.Sequential(Linear(dim, heads), nn.Sigmoid())
        nn.init.zeros_(self.to_lr[0].weight)
        nn.init.constant_(self.to_lr[0].bias, -4.0)
        
        self.to_decay = nn.Sequential(Linear(dim, heads), nn.Sigmoid())
        nn.init.zeros_(self.to_decay[0].weight)
        nn.init.constant_(self.to_decay[0].bias, 4.0)
        
        self.to_momentum = nn.Sequential(Linear(dim, heads), nn.Sigmoid()) if use_momentum else None
        if use_momentum:
            nn.init.zeros_(self.to_momentum[0].weight)
            nn.init.constant_(self.to_momentum[0].bias, 2.0)
        
        # Omega gate
        self.to_omega_gate = nn.Sequential(Linear(dim, heads), nn.Sigmoid()) if use_omega_gate else None
        if use_omega_gate:
            nn.init.zeros_(self.to_omega_gate[0].weight)
            nn.init.constant_(self.to_omega_gate[0].bias, 0.0)
        
        # Norms
        self.pre_norm = nn.RMSNorm(dim)
        self.q_norm = MultiheadRMSNorm(dim_head, heads) if qk_norm else nn.Identity()
        self.k_norm = MultiheadRMSNorm(dim_head, heads) if qk_norm else nn.Identity()
        
        # Polynomial feature map
        self.phi = PolynomialFeatureMap(dim_head, poly_degree, poly_mode)
        
        # Head operations
        self.split_heads = Rearrange('b n (h d) -> b h n d', h=heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        
        self.register_buffer('_dummy', torch.zeros(1))
    
    def init_state(self, batch: int, device=None, dtype=None) -> RNNMemState:
        """Initialize memory state with omega buffer."""
        device = default(device, self._dummy.device)
        dtype = default(dtype, self._dummy.dtype)
        
        S = torch.zeros(batch * self.heads, self.dim_head, self.dim_head,
                       device=device, dtype=dtype)
        Z = torch.zeros_like(S) if self.use_momentum else None
        
        # Buffer for sliding window: stores g for last (e-1) tokens (refactored)
        if self.omega_window > 1:
            omega_buffer = torch.zeros(
                batch * self.heads, self.omega_window - 1, self.dim_head, self.dim_head,
                device=device, dtype=dtype
            )
        else:
            omega_buffer = None
        
        return RNNMemState(seq_index=0, S=S, Z=Z, omega_buffer=omega_buffer)

    def forward(
        self,
        x: Tensor,
        state: RNNMemState | None = None,
    ) -> tuple[Tensor, RNNMemState]:
        """
        Forward pass with Omega rule (sliding window) and efficient scalar scan.
        
        Uses fixed S_0 (chunk-start state) for gradient computation,
        enabling 6x memory reduction compared to matrix affine scan.
        """
        batch, seq_len, _ = x.shape

        if not exists(state):
            state = self.init_state(batch, x.device, x.dtype)

        # Pre-norm and project (used for projections + scalar gates)
        x_normed = self.activation(self.pre_norm(x))

        # Optional depthwise conv (compute on full seq once; safe to slice later)
        q_in = k_in = v_in = x_normed
        if exists(self.q_conv):
            q_in = self.q_conv(q_in.transpose(1, 2)).transpose(1, 2)
        if exists(self.k_conv):
            k_in = self.k_conv(k_in.transpose(1, 2)).transpose(1, 2)
        if exists(self.v_conv):
            v_in = self.v_conv(v_in.transpose(1, 2)).transpose(1, 2)

        chunk_len = self.scan_chunk_len
        if exists(chunk_len) and seq_len > chunk_len:
            outs: list[Tensor] = []
            cur_state = state
            # Preserve current (unchunked) semantics: g is computed from the reference state at call start.
            S_ref = cur_state.S
            for start in range(0, seq_len, chunk_len):
                end = min(seq_len, start + chunk_len)
                out, cur_state = self._forward_impl(
                    x_normed[:, start:end],
                    q_in[:, start:end],
                    k_in[:, start:end],
                    v_in[:, start:end],
                    cur_state,
                    S_ref=S_ref,
                )
                outs.append(out)
            return torch.cat(outs, dim=1), cur_state

        return self._forward_impl(x_normed, q_in, k_in, v_in, state, S_ref=state.S)

    def _forward_impl(
        self,
        x_normed: Tensor,
        q_in: Tensor,
        k_in: Tensor,
        v_in: Tensor,
        state: RNNMemState,
        *,
        S_ref: Tensor,
    ) -> tuple[Tensor, RNNMemState]:
        batch, seq_len, _ = x_normed.shape
        e = self.omega_window
        d = self.dim_head
        BH = batch * self.heads

        # Use CUDA backend if available and enabled
        if self.use_cuda and self.cuda_available:
            return self._forward_cuda(x_normed, q_in, k_in, v_in, state, S_ref=S_ref)

        q = self.split_heads(self.to_q(q_in))
        k = self.split_heads(self.to_k(k_in))
        v = self.split_heads(self.to_v(v_in))

        # Truncate to original seq_len in case conv changed length
        q = q[:, :, :seq_len]
        k = k[:, :, :seq_len]
        v = v[:, :, :seq_len]

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply polynomial features
        k_flat = rearrange(k, 'b h n d -> (b h n) d')
        q_flat = rearrange(q, 'b h n d -> (b h n) d')

        phi_k = rearrange(self.phi(k_flat), '(b h n) d -> (b h) n d', b=batch, h=self.heads, n=seq_len)
        phi_q = rearrange(self.phi(q_flat), '(b h n) d -> (b h) n d', b=batch, h=self.heads, n=seq_len)
        v_bh = rearrange(v, 'b h n d -> (b h) n d')

        # Learned hyperparameters (use original x_normed which has original seq_len)
        lr = rearrange(self.to_lr(x_normed), 'b n h -> (b h) n')
        decay = rearrange(self.to_decay(x_normed), 'b n h -> (b h) n')

        if self.use_momentum:
            momentum = rearrange(self.to_momentum(x_normed), 'b n h -> (b h) n')

        omega_gate = None
        if self.use_omega_gate:
            omega_gate = rearrange(self.to_omega_gate(x_normed), 'b n h -> (b h) n')

        # -----------------------------------------------------------------
        # Per-token gradient (Exact Refactor: avoids G, B materialization)
        # -----------------------------------------------------------------
        # δ_t = S_ref @ φ_t - v_t (prediction error)
        pred = torch.einsum('bde,bte->btd', S_ref, phi_k)  # [BH, T, d]
        delta_vec = pred - v_bh                             # [BH, T, d]

        # g_raw_t = δ_t ⊗ φ_t^T (gradient before omega window)
        g_raw = torch.einsum('bti,btj->btij', delta_vec, phi_k)  # [BH, T, d, d]

        # Apply omega gate to gradient (not to G/B separately)
        if exists(omega_gate):
            gate_e = omega_gate[..., None, None]  # [BH, T, 1, 1]
            g_raw = g_raw * gate_e

        # -----------------------------------------------------------------
        # Omega window: sliding sum of g (single buffer, not G+B)
        # -----------------------------------------------------------------
        if e > 1:
            omega_buffer = state.omega_buffer
            if exists(omega_buffer):
                prev_g = omega_buffer  # [BH, e-1, d, d]
            else:
                prev_g = g_raw.new_zeros((BH, e - 1, d, d))

            g_ext = torch.cat([prev_g, g_raw], dim=1)  # [BH, e-1+T, d, d]
            g = _sliding_sum_along_time(g_ext, e)[:, -(seq_len):]  # [BH, T, d, d]

            # Store last e-1 gradients for next call
            new_omega_buffer = g_ext[:, -(e - 1):]  # [BH, e-1, d, d]
        else:
            g = g_raw
            new_omega_buffer = None

        # -----------------------------------------------------------------
        # Efficient Scalar Scan Implementation
        # -----------------------------------------------------------------
        S0 = state.S  # [BH, d, d]
        Z0 = state.Z if exists(state.Z) else torch.zeros_like(S0)

        lr_e = lr[..., None, None]

        if self.use_momentum:
            Z_all = self.assoc_scan(momentum, g, prev=Z0)  # [BH, T, d, d]
            delta = -lr_e * Z_all  # [BH, T, d, d]
            S_all = self.assoc_scan(decay, delta, prev=S0)  # [BH, T, d, d]
            S_end = S_all[:, -1].clamp(-100, 100)
            Z_end = Z_all[:, -1]
        else:
            delta = -lr_e * g  # [BH, T, d, d]
            S_all = self.assoc_scan(decay, delta, prev=S0)  # [BH, T, d, d]
            S_end = S_all[:, -1].clamp(-100, 100)
            Z_end = None

        # -----------------------------------------------------------------
        # Retrieval: y_t = S_{t-1} @ ψ_t
        # -----------------------------------------------------------------
        y0 = torch.einsum('bde,be->bd', S0, phi_q[:, 0])  # [BH, d]
        if seq_len > 1:
            y_rest = torch.einsum('btdp,btp->btd', S_all[:, :-1], phi_q[:, 1:])  # [BH, T-1, d]
            retrieved = torch.cat([y0.unsqueeze(1), y_rest], dim=1)  # [BH, T, d]
        else:
            retrieved = y0.unsqueeze(1)  # [BH, 1, d]

        retrieved = rearrange(retrieved, '(b h) t d -> b h t d', b=batch, h=self.heads)
        retrieved = self.merge_heads(retrieved)
        retrieved = self.to_out(retrieved)

        next_state = RNNMemState(
            seq_index=state.seq_index + seq_len,
            S=S_end,
            Z=Z_end,
            omega_buffer=new_omega_buffer
        )

        return retrieved, next_state

    def _forward_cuda(
        self,
        x_normed: Tensor,
        q_in: Tensor,
        k_in: Tensor,
        v_in: Tensor,
        state: RNNMemState,
        *,
        S_ref: Tensor,
    ) -> tuple[Tensor, RNNMemState]:
        """
        CUDA-accelerated forward pass for omega_window=16.
        
        This is ~50-100x faster than PyTorch and uses ~100x less memory.
        """
        batch, seq_len, _ = x_normed.shape
        d = self.dim_head
        BH = batch * self.heads

        # Q/K/V projections
        q = self.split_heads(self.to_q(q_in))
        k = self.split_heads(self.to_k(k_in))
        v = self.split_heads(self.to_v(v_in))

        # Truncate to original seq_len in case conv changed length
        q = q[:, :, :seq_len]
        k = k[:, :, :seq_len]
        v = v[:, :, :seq_len]

        # Normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply polynomial features
        k_flat = rearrange(k, 'b h n d -> (b h n) d')
        q_flat = rearrange(q, 'b h n d -> (b h n) d')

        phi_k = rearrange(self.phi(k_flat), '(b h n) d -> (b h) n d', b=batch, h=self.heads, n=seq_len)
        phi_q = rearrange(self.phi(q_flat), '(b h n) d -> (b h) n d', b=batch, h=self.heads, n=seq_len)
        v_bh = rearrange(v, 'b h n d -> (b h) n d')

        # Learned hyperparameters
        lr = rearrange(self.to_lr(x_normed), 'b n h -> (b h) n')
        decay = rearrange(self.to_decay(x_normed), 'b n h -> (b h) n')
        beta = rearrange(self.to_momentum(x_normed), 'b n h -> (b h) n') if self.use_momentum else torch.ones_like(lr)
        gate = rearrange(self.to_omega_gate(x_normed), 'b n h -> (b h) n') if self.use_omega_gate else torch.ones_like(lr)

        # Get initial states
        S0 = state.S if exists(state.S) else phi_k.new_zeros((BH, d, d))
        Z0 = state.Z if exists(state.Z) else phi_k.new_zeros((BH, d, d))

        # Ensure bfloat16 for CUDA kernel
        phi_k = phi_k.to(torch.bfloat16)
        phi_q = phi_q.to(torch.bfloat16)
        v_bh = v_bh.to(torch.bfloat16)
        S_ref = S_ref.to(torch.bfloat16)
        lr = lr.to(torch.bfloat16)
        decay = decay.to(torch.bfloat16)
        beta = beta.to(torch.bfloat16)
        gate = gate.to(torch.bfloat16)
        S0 = S0.to(torch.bfloat16)
        Z0 = Z0.to(torch.bfloat16)

        # Call CUDA kernel
        y_bh, S_T, Z_T = self.AtlasOmegaFunction.apply(
            phi_k, phi_q, v_bh, S_ref, lr, decay, beta, gate, S0, Z0
        )

        # Reshape output
        y_bh = y_bh.to(x_normed.dtype)  # Convert back to original dtype
        retrieved = rearrange(y_bh, '(b h) t d -> b h t d', b=batch, h=self.heads)
        retrieved = self.merge_heads(retrieved)
        retrieved = self.to_out(retrieved)

        # Update state (no omega_buffer needed for CUDA, it's managed internally)
        next_state = RNNMemState(
            seq_index=state.seq_index + seq_len,
            S=S_T.to(x_normed.dtype),
            Z=Z_T.to(x_normed.dtype) if self.use_momentum else None,
            omega_buffer=None  # CUDA manages ring buffer internally
        )

        return retrieved, next_state


# ============================================================================
# Wrapper for compatibility
# ============================================================================

class RNNMemory(Module):
    """
    Drop-in replacement for NeuralMemory using explicit RNN updates.
    """
    
    def __init__(
        self,
        dim: int,
        dim_head: int | None = None,
        heads: int = 1,
        use_momentum: bool = True,
        poly_degree: int = 1,
        poly_mode: str = 'off',
        omega_window: int = 1,
        use_omega_gate: bool = False,
        # Legacy parameters (ignored)
        chunk_size: int = 1,
        use_accelerated_scan: bool = False,
        scan_chunk_len: int | None = None,
        use_cuda: bool = False,
        **kwargs
    ):
        super().__init__()
        
        dim_head = default(dim_head, dim)
        
        if omega_window > 1 or use_omega_gate:
            self.cell = OmegaRNNMemoryCell(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                omega_window=omega_window,
                use_omega_gate=use_omega_gate,
                use_momentum=use_momentum,
                poly_degree=poly_degree,
                poly_mode=poly_mode,
                use_accelerated_scan=use_accelerated_scan,
                scan_chunk_len=scan_chunk_len,
                use_cuda=use_cuda,
                **kwargs
            )
        else:
            self.cell = RNNMemoryCell(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                use_momentum=use_momentum,
                poly_degree=poly_degree,
                poly_mode=poly_mode,
                use_accelerated_scan=use_accelerated_scan,
                scan_chunk_len=scan_chunk_len,
                **kwargs
            )
    
    def forward(
        self,
        seq: Tensor,
        state: RNNMemState | None = None,
        **kwargs
    ) -> tuple[Tensor, RNNMemState]:
        return self.cell(seq, state)


# Alias for OmegaNet-RNN
OmegaRNNMemory = partial(RNNMemory, omega_window=4, use_omega_gate=True)
