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

Parallelization via affine associative scan:
- Each token induces an affine transform on state H=[S,Z]
- H_t = H_{t-1} @ A_t + C_t
- All states computed in O(log T) depth via prefix scan
"""

from __future__ import annotations
from typing import Callable, NamedTuple
from functools import partial

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module, Parameter, Linear, Conv1d

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# ============================================================================
# Helper functions
# ============================================================================

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

LinearNoBias = partial(Linear, bias=False)


# ============================================================================
# Affine associative scan (for per-token parallelization)
# ============================================================================

def _interleave(a: Tensor, b: Tensor) -> Tensor:
    """Interleave two sequences along dim=1 (time)."""
    a_len, b_len = a.shape[1], b.shape[1]
    out_len = a_len + b_len

    if a_len == (b_len + 1):
        pad_shape = (*b.shape[:1], 1, *b.shape[2:])
        b = torch.cat([b, b.new_zeros(pad_shape)], dim=1)

    stacked = torch.stack([a, b], dim=2)
    interleaved = torch.flatten(stacked, start_dim=1, end_dim=2)
    return interleaved[:, :out_len]


def _associative_scan(
    operator: Callable,
    elems: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    """
    Pytorch implementation of JAX-style associative scan along dim=1.
    Returns inclusive scan (prefix) of `elems` under `operator`.
    """
    num_elems = int(elems[0].shape[1])
    if not all(int(e.shape[1]) == num_elems for e in elems[1:]):
        raise ValueError(f"associative_scan: all elems must share same time dim; saw {[e.shape for e in elems]}")

    def _scan(cur: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        n = int(cur[0].shape[1])
        if n < 2:
            return cur

        reduced = operator(
            [e[:, :-1:2] for e in cur],
            [e[:, 1::2] for e in cur],
        )
        reduced_tup = tuple(reduced)

        odd = _scan(reduced_tup)

        if n % 2 == 0:
            even = operator(
                [e[:, :-1] for e in odd],
                [e[:, 2::2] for e in cur],
            )
        else:
            even = operator(
                list(odd),
                [e[:, 2::2] for e in cur],
            )

        even_tup = tuple(even)
        even_tup = tuple(
            torch.cat([orig[:, :1], result], dim=1)
            for orig, result in zip(cur, even_tup)
        )

        return tuple(_interleave(e, o) for e, o in zip(even_tup, odd))

    return _scan(elems)


def _affine_pair_operator(a, b):
    """
    Compose affine transforms: H -> H @ A + C
    (A1, C1) then (A2, C2) = (A1@A2, C1@A2 + C2)
    """
    A1, C1 = a
    A2, C2 = b
    A = torch.matmul(A1, A2)
    C = torch.matmul(C1, A2) + C2
    return A, C


def _affine_scan_apply(H0: Tensor, A_seq: Tensor, C_seq: Tensor) -> Tensor:
    """
    Compute inclusive states for affine recurrence:
      H_t = H_{t-1} @ A_t + C_t
    using associative scan.

    H0:    [B, M, D]
    A_seq: [B, T, D, D]
    C_seq: [B, T, M, D]
    Returns:
      H_all: [B, T, M, D]  (H_1..H_T)
    """
    A_pref, C_pref = _associative_scan(_affine_pair_operator, (A_seq, C_seq))
    H0A = torch.einsum('bmd,btde->btme', H0, A_pref)
    return H0A + C_pref


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
# RNN Memory Cell (Titans-RNN) - Per-token semantics with affine scan
# ============================================================================

class RNNMemoryCell(Module):
    """
    Per-token RNN memory update with affine scan parallelization.
    
    Implements (per-token):
        g_t = (S_{t-1} φ_t - v_t) φ_t^T         (gradient)
        Z_t = β_t Z_{t-1} + g_t                  (momentum)
        S_t = α_t S_{t-1} - η_t Z_t              (memory update)
        y_t = S_{t-1} ψ_t                        (retrieval)
    
    Parallelized via affine scan over H=[S,Z].
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
        # Legacy parameters (ignored for compatibility)
        chunk_size: int = 1,
        use_accelerated_scan: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.use_momentum = use_momentum
        self.qkv_conv_kernel = qkv_conv_kernel
        
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
        Forward pass with per-token semantics.
        
        Uses affine scan for O(log T) parallelization while maintaining
        exact per-token update semantics.
        """
        batch, seq_len, _ = x.shape
        
        if not exists(state):
            state = self.init_state(batch, x.device, x.dtype)
        
        d = self.dim_head
        BH = batch * self.heads
        
        # Pre-norm + activation
        x = self.activation(self.pre_norm(x))

        q_in, k_in, v_in = x, x, x

        # Optional depthwise conv
        if exists(self.q_conv):
            q_in = self.q_conv(q_in.transpose(1, 2)).transpose(1, 2)
        if exists(self.k_conv):
            k_in = self.k_conv(k_in.transpose(1, 2)).transpose(1, 2)
        if exists(self.v_conv):
            v_in = self.v_conv(v_in.transpose(1, 2)).transpose(1, 2)
        
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
        lr = rearrange(self.to_lr(x), 'b n h -> (b h) n')
        decay = rearrange(self.to_decay(x), 'b n h -> (b h) n')
        
        if self.use_momentum:
            momentum = rearrange(self.to_momentum(x), 'b n h -> (b h) n')
        
        # -----------------------------------------------------------------
        # Build per-token affine transforms
        # -----------------------------------------------------------------
        # G_t = φ_t ⊗ φ_t^T: [BH, T, d, d]
        G = torch.einsum('bti,btj->btij', phi_k, phi_k)
        # B_t = v_t ⊗ φ_t^T: [BH, T, d, d]
        B = torch.einsum('bti,btj->btij', v_bh, phi_k)
        
        # Expand scalars for broadcasting: [BH, T, 1, 1]
        lr_e = lr[..., None, None]
        decay_e = decay[..., None, None]
        
        # Identity matrix: [BH, T, d, d]
        I = torch.eye(d, device=x.device, dtype=x.dtype).expand(BH, seq_len, d, d)
        
        if self.use_momentum:
            mom_e = momentum[..., None, None]
            
            # Build block affine: H=[S,Z], H_t = H_{t-1} @ A_t + C_t
            # A_t: [BH, T, 2d, 2d], C_t: [BH, T, d, 2d]
            A11 = decay_e * I - lr_e * G
            A12 = G
            A21 = -(lr_e * mom_e) * I
            A22 = mom_e * I
            
            A_top = torch.cat([A11, A12], dim=-1)
            A_bot = torch.cat([A21, A22], dim=-1)
            A_seq = torch.cat([A_top, A_bot], dim=-2)  # [BH, T, 2d, 2d]
            
            C_S = lr_e * B
            C_Z = -B
            C_seq = torch.cat([C_S, C_Z], dim=-1)      # [BH, T, d, 2d]
            
            # Initial state H0: [BH, d, 2d]
            S0 = state.S
            Z0 = state.Z if exists(state.Z) else torch.zeros_like(S0)
            H0 = torch.cat([S0, Z0], dim=-1)
            
            # Affine scan: compute H_1, ..., H_T
            H_all = _affine_scan_apply(H0, A_seq, C_seq)  # [BH, T, d, 2d]
            
            # For retrieval: y_t = S_{t-1} @ ψ_t
            # H_start = [H_0, H_1, ..., H_{T-1}]
            H_start = torch.cat([H0.unsqueeze(1), H_all[:, :-1]], dim=1)  # [BH, T, d, 2d]
            S_start = H_start[..., :d]  # [BH, T, d, d]
            
            # Final state
            H_end = H_all[:, -1]
            S_end = H_end[..., :d].clamp(-100, 100)
            Z_end = H_end[..., d:]
        else:
            # No momentum: simpler A_t = α_t I - η_t G_t, C_t = η_t B_t
            A_seq = decay_e * I - lr_e * G  # [BH, T, d, d]
            C_seq = lr_e * B                 # [BH, T, d, d]
            
            S0 = state.S
            
            # Affine scan
            S_all = _affine_scan_apply(S0, A_seq, C_seq)  # [BH, T, d, d]
            
            # For retrieval: S_start = [S_0, S_1, ..., S_{T-1}]
            S_start = torch.cat([S0.unsqueeze(1), S_all[:, :-1]], dim=1)
            
            S_end = S_all[:, -1].clamp(-100, 100)
            Z_end = None
        
        # -----------------------------------------------------------------
        # Retrieval: y_t = S_{t-1} @ ψ_t
        # -----------------------------------------------------------------
        # S_start: [BH, T, d, d], phi_q: [BH, T, d]
        retrieved = torch.einsum('btdp,btp->btd', S_start, phi_q)  # [BH, T, d]
        
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
    
    Then same affine scan as Titans-RNN.
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
        # Legacy parameters (ignored)
        chunk_size: int = 1,
        use_accelerated_scan: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.omega_window = omega_window
        self.use_omega_gate = use_omega_gate
        self.use_momentum = use_momentum
        self.qkv_conv_kernel = qkv_conv_kernel
        
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
        
        # Buffer for sliding window: stores (G, B) for last (e-1) tokens
        if self.omega_window > 1:
            omega_buffer = torch.zeros(
                batch * self.heads, self.omega_window - 1, self.dim_head, self.dim_head, 2,
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
        """Forward pass with Omega rule (sliding window)."""
        batch, seq_len, _ = x.shape
        e = self.omega_window
        
        if not exists(state):
            state = self.init_state(batch, x.device, x.dtype)
        
        d = self.dim_head
        BH = batch * self.heads
        
        # Pre-norm and project
        x_normed = self.activation(self.pre_norm(x))
        
        q_in, k_in, v_in = x_normed, x_normed, x_normed
        
        if exists(self.q_conv):
            q_in = self.q_conv(q_in.transpose(1, 2)).transpose(1, 2)
        if exists(self.k_conv):
            k_in = self.k_conv(k_in.transpose(1, 2)).transpose(1, 2)
        if exists(self.v_conv):
            v_in = self.v_conv(v_in.transpose(1, 2)).transpose(1, 2)
        
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
        # Per-token G_t and B_t (before windowing)
        # -----------------------------------------------------------------
        # Raw per-token outer products
        G_raw = torch.einsum('bti,btj->btij', phi_k, phi_k)  # [BH, T, d, d]
        B_raw = torch.einsum('bti,btj->btij', v_bh, phi_k)    # [BH, T, d, d]
        
        # Apply omega gate if enabled
        if exists(omega_gate):
            gate_e = omega_gate[..., None, None]  # [BH, T, 1, 1]
            G_raw = G_raw * gate_e
            B_raw = B_raw * gate_e
        
        # -----------------------------------------------------------------
        # Omega window: sliding sum of G and B
        # -----------------------------------------------------------------
        if e > 1:
            # Prepend buffer from state
            omega_buffer = state.omega_buffer
            if exists(omega_buffer):
                prev_G = omega_buffer[..., 0]  # [BH, e-1, d, d]
                prev_B = omega_buffer[..., 1]
            else:
                prev_G = G_raw.new_zeros((BH, e - 1, d, d))
                prev_B = B_raw.new_zeros((BH, e - 1, d, d))
            
            G_ext = torch.cat([prev_G, G_raw], dim=1)  # [BH, e-1+T, d, d]
            B_ext = torch.cat([prev_B, B_raw], dim=1)
            
            # Sliding sum over window
            G = _sliding_sum_along_time(G_ext, e)[:, -(seq_len):]  # [BH, T, d, d]
            B = _sliding_sum_along_time(B_ext, e)[:, -(seq_len):]
            
            # Update buffer for next call
            new_omega_buffer = torch.stack([
                G_ext[:, -(e - 1):],
                B_ext[:, -(e - 1):]
            ], dim=-1)  # [BH, e-1, d, d, 2]
        else:
            G = G_raw
            B = B_raw
            new_omega_buffer = None
        
        # -----------------------------------------------------------------
        # Build per-token affine transforms (same as Titans-RNN)
        # -----------------------------------------------------------------
        lr_e = lr[..., None, None]
        decay_e = decay[..., None, None]
        I = torch.eye(d, device=x.device, dtype=x.dtype).expand(BH, seq_len, d, d)
        
        if self.use_momentum:
            mom_e = momentum[..., None, None]
            
            A11 = decay_e * I - lr_e * G
            A12 = G
            A21 = -(lr_e * mom_e) * I
            A22 = mom_e * I
            
            A_top = torch.cat([A11, A12], dim=-1)
            A_bot = torch.cat([A21, A22], dim=-1)
            A_seq = torch.cat([A_top, A_bot], dim=-2)
            
            C_S = lr_e * B
            C_Z = -B
            C_seq = torch.cat([C_S, C_Z], dim=-1)
            
            S0 = state.S
            Z0 = state.Z if exists(state.Z) else torch.zeros_like(S0)
            H0 = torch.cat([S0, Z0], dim=-1)
            
            H_all = _affine_scan_apply(H0, A_seq, C_seq)
            
            H_start = torch.cat([H0.unsqueeze(1), H_all[:, :-1]], dim=1)
            S_start = H_start[..., :d]
            
            H_end = H_all[:, -1]
            S_end = H_end[..., :d].clamp(-100, 100)
            Z_end = H_end[..., d:]
        else:
            A_seq = decay_e * I - lr_e * G
            C_seq = lr_e * B
            
            S0 = state.S
            
            S_all = _affine_scan_apply(S0, A_seq, C_seq)
            
            S_start = torch.cat([S0.unsqueeze(1), S_all[:, :-1]], dim=1)
            
            S_end = S_all[:, -1].clamp(-100, 100)
            Z_end = None
        
        # -----------------------------------------------------------------
        # Retrieval: y_t = S_{t-1} @ ψ_t
        # -----------------------------------------------------------------
        retrieved = torch.einsum('btdp,btp->btd', S_start, phi_q)
        
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
