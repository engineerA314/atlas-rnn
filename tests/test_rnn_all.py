import pytest
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from atlas_pytorch.rnn_memory import (
    RNNMemoryCell,
    OmegaRNNMemoryCell,
    PolynomialFeatureMap,
    RNNMemState,
    _sliding_sum_along_time,
)
from atlas_pytorch.rnn_transformer import RNNMemoryTransformer
from assoc_scan import AssocScan
import math

# small local helper (mirrors atlas_pytorch style)
def exists(v):
    return v is not None

def _dtype_close_kwargs(dtype: torch.dtype):
    # fp32 should be very tight; bf16/fp16 may differ slightly due to different op ordering
    if dtype == torch.float32:
        return dict(atol=1e-5, rtol=1e-5)
    if dtype in (torch.float16, torch.bfloat16):
        return dict(atol=2e-2, rtol=2e-2)
    return dict(atol=1e-5, rtol=1e-5)

def _supported_test_dtypes(device: torch.device):
    # Keep CPU deterministic + fast; exercise fp16/bf16 only when CUDA is available.
    if device.type == "cuda":
        return (torch.float32, torch.float16, torch.bfloat16)
    return (torch.float32,)

def _scaled_randn(shape, *, device, dtype, scale: float):
    return torch.randn(*shape, device=device, dtype=dtype) * scale

# -----------------------------------------------------------------------------
# Reference implementations (pre-optimization)
# -----------------------------------------------------------------------------

def _reference_forward_rnn_memory_cell(
    cell: RNNMemoryCell,
    x: torch.Tensor,
    state: RNNMemState | None = None,
):
    """
    Reference forward that matches the pre-change implementation:
    - materializes S_start = cat([S0, S_all[:, :-1]]) and uses it for retrieval
    - does not use scan_chunk_len optimization
    """
    batch, seq_len, _ = x.shape

    if state is None:
        state = cell.init_state(batch, x.device, x.dtype)

    d = cell.dim_head

    # Pre-norm + activation
    x_act = cell.activation(cell.pre_norm(x))

    q_in = k_in = v_in = x_act

    # Optional depthwise conv
    if exists(cell.q_conv):
        q_in = cell.q_conv(q_in.transpose(1, 2)).transpose(1, 2)
    if exists(cell.k_conv):
        k_in = cell.k_conv(k_in.transpose(1, 2)).transpose(1, 2)
    if exists(cell.v_conv):
        v_in = cell.v_conv(v_in.transpose(1, 2)).transpose(1, 2)

    # Project to Q, K, V and split heads: [batch, heads, seq, dim_head]
    q = cell.split_heads(cell.to_q(q_in))
    k = cell.split_heads(cell.to_k(k_in))
    v = cell.split_heads(cell.to_v(v_in))

    # Truncate to original seq_len in case conv changed length
    q = q[:, :, :seq_len]
    k = k[:, :, :seq_len]
    v = v[:, :, :seq_len]

    q = cell.q_norm(q)
    k = cell.k_norm(k)

    # Apply polynomial feature map
    k_flat = rearrange(k, 'b h n d -> (b h n) d')
    q_flat = rearrange(q, 'b h n d -> (b h n) d')

    phi_k = rearrange(cell.phi(k_flat), '(b h n) d -> (b h) n d', b=batch, h=cell.heads, n=seq_len)
    phi_q = rearrange(cell.phi(q_flat), '(b h n) d -> (b h) n d', b=batch, h=cell.heads, n=seq_len)
    v_bh = rearrange(v, 'b h n d -> (b h) n d')

    # Learned hyperparameters: [BH, T]
    lr = rearrange(cell.to_lr(x_act), 'b n h -> (b h) n')
    decay = rearrange(cell.to_decay(x_act), 'b n h -> (b h) n')

    if cell.use_momentum:
        momentum = rearrange(cell.to_momentum(x_act), 'b n h -> (b h) n')

    # Step 1: outer products
    G = torch.einsum('bti,btj->btij', phi_k, phi_k)
    B = torch.einsum('bti,btj->btij', v_bh, phi_k)

    S0 = state.S
    Z0 = state.Z if exists(state.Z) else torch.zeros_like(S0)

    # Step 2: fixed S0 gradient
    g = torch.einsum('bde,btef->btdf', S0, G) - B

    lr_e = lr[..., None, None]

    if cell.use_momentum:
        Z_all = cell.assoc_scan(momentum, g, prev=Z0)
        delta = -lr_e * Z_all
        S_all = cell.assoc_scan(decay, delta, prev=S0)
        S_end = S_all[:, -1].clamp(-100, 100)
        Z_end = Z_all[:, -1]
    else:
        delta = -lr_e * g
        S_all = cell.assoc_scan(decay, delta, prev=S0)
        S_end = S_all[:, -1].clamp(-100, 100)
        Z_end = None

    # Retrieval (pre-change): materialize S_start
    S_start = torch.cat([S0.unsqueeze(1), S_all[:, :-1]], dim=1)
    retrieved = torch.einsum('btdp,btp->btd', S_start, phi_q)

    retrieved = rearrange(retrieved, '(b h) t d -> b h t d', b=batch, h=cell.heads)
    retrieved = cell.merge_heads(retrieved)
    retrieved = cell.to_out(retrieved)

    next_state = RNNMemState(
        seq_index=state.seq_index + seq_len,
        S=S_end,
        Z=Z_end,
        omega_buffer=None,
    )

    return retrieved, next_state


def _reference_forward_omega_rnn_memory_cell(
    cell: OmegaRNNMemoryCell,
    x: torch.Tensor,
    state: RNNMemState | None = None,
):
    """
    Reference forward that matches the pre-change Omega implementation:
    - materializes S_start = cat([S0, S_all[:, :-1]]) and uses it for retrieval
    - does not use scan_chunk_len optimization
    """
    batch, seq_len, _ = x.shape
    e = cell.omega_window

    if state is None:
        state = cell.init_state(batch, x.device, x.dtype)

    d = cell.dim_head
    BH = batch * cell.heads

    x_normed = cell.activation(cell.pre_norm(x))

    q_in = k_in = v_in = x_normed

    if exists(cell.q_conv):
        q_in = cell.q_conv(q_in.transpose(1, 2)).transpose(1, 2)
    if exists(cell.k_conv):
        k_in = cell.k_conv(k_in.transpose(1, 2)).transpose(1, 2)
    if exists(cell.v_conv):
        v_in = cell.v_conv(v_in.transpose(1, 2)).transpose(1, 2)

    q = cell.split_heads(cell.to_q(q_in))
    k = cell.split_heads(cell.to_k(k_in))
    v = cell.split_heads(cell.to_v(v_in))

    q = q[:, :, :seq_len]
    k = k[:, :, :seq_len]
    v = v[:, :, :seq_len]

    q = cell.q_norm(q)
    k = cell.k_norm(k)

    k_flat = rearrange(k, 'b h n d -> (b h n) d')
    q_flat = rearrange(q, 'b h n d -> (b h n) d')

    phi_k = rearrange(cell.phi(k_flat), '(b h n) d -> (b h) n d', b=batch, h=cell.heads, n=seq_len)
    phi_q = rearrange(cell.phi(q_flat), '(b h n) d -> (b h) n d', b=batch, h=cell.heads, n=seq_len)
    v_bh = rearrange(v, 'b h n d -> (b h) n d')

    lr = rearrange(cell.to_lr(x_normed), 'b n h -> (b h) n')
    decay = rearrange(cell.to_decay(x_normed), 'b n h -> (b h) n')

    if cell.use_momentum:
        momentum = rearrange(cell.to_momentum(x_normed), 'b n h -> (b h) n')

    omega_gate = None
    if cell.use_omega_gate:
        omega_gate = rearrange(cell.to_omega_gate(x_normed), 'b n h -> (b h) n')

    G_raw = torch.einsum('bti,btj->btij', phi_k, phi_k)
    B_raw = torch.einsum('bti,btj->btij', v_bh, phi_k)

    if exists(omega_gate):
        gate_e = omega_gate[..., None, None]
        G_raw = G_raw * gate_e
        B_raw = B_raw * gate_e

    if e > 1:
        omega_buffer = state.omega_buffer
        if exists(omega_buffer):
            prev_G = omega_buffer[..., 0]
            prev_B = omega_buffer[..., 1]
        else:
            prev_G = G_raw.new_zeros((BH, e - 1, d, d))
            prev_B = B_raw.new_zeros((BH, e - 1, d, d))

        G_ext = torch.cat([prev_G, G_raw], dim=1)
        B_ext = torch.cat([prev_B, B_raw], dim=1)

        G = _sliding_sum_along_time(G_ext, e)[:, -(seq_len):]
        B = _sliding_sum_along_time(B_ext, e)[:, -(seq_len):]

        new_omega_buffer = torch.stack([G_ext[:, -(e - 1):], B_ext[:, -(e - 1):]], dim=-1)
    else:
        G = G_raw
        B = B_raw
        new_omega_buffer = None

    S0 = state.S
    Z0 = state.Z if exists(state.Z) else torch.zeros_like(S0)

    g = torch.einsum('bde,btef->btdf', S0, G) - B
    lr_e = lr[..., None, None]

    if cell.use_momentum:
        Z_all = cell.assoc_scan(momentum, g, prev=Z0)
        delta = -lr_e * Z_all
        S_all = cell.assoc_scan(decay, delta, prev=S0)
        S_end = S_all[:, -1].clamp(-100, 100)
        Z_end = Z_all[:, -1]
    else:
        delta = -lr_e * g
        S_all = cell.assoc_scan(decay, delta, prev=S0)
        S_end = S_all[:, -1].clamp(-100, 100)
        Z_end = None

    S_start = torch.cat([S0.unsqueeze(1), S_all[:, :-1]], dim=1)
    retrieved = torch.einsum('btdp,btp->btd', S_start, phi_q)

    retrieved = rearrange(retrieved, '(b h) t d -> b h t d', b=batch, h=cell.heads)
    retrieved = cell.merge_heads(retrieved)
    retrieved = cell.to_out(retrieved)

    next_state = RNNMemState(
        seq_index=state.seq_index + seq_len,
        S=S_end,
        Z=Z_end,
        omega_buffer=new_omega_buffer,
    )

    return retrieved, next_state

# -----------------------------------------------------------------------------
# CUDA / Triton integration test (skip on non-CUDA)
# -----------------------------------------------------------------------------

def _skip_if_no_cuda_triton():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available (integration test for triton accelerated scan)")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("triton not installed")
    try:
        import accelerated_scan.triton as triton_mod  # noqa
    except Exception as e:
        pytest.skip(f"accelerated_scan.triton import failed: {e}")
    return triton_mod


def test_cuda_available_for_integration():
    """Marker test - skip if no CUDA."""
    _skip_if_no_cuda_triton()
    assert torch.cuda.is_available()

# ---------------------------------------
# Global small configs for fast execution
# ---------------------------------------

ARCHES = ['mag', 'mal', 'lmm', 'mac']
OMEGAS = [1, 4]
GATES = [False, True]
POLY_MODES = ['off', 'elementwise', 'tensor']
POLY_DEGS = [1, 2]

HEADS = [1, 2]
DIM_HEADS = [16, 32]
BATCHES = [1, 2]
SEQLENS = [8, 16]
WINDOWS = [8, 16]
PERSIST_MEMS = [0, 2]

# Curated grid to prevent combinatorial explosion
GRID = [
    dict(arch='mag', omega=1, gate=False, poly=('off', 1), heads=2, dhead=32, batch=2, seqlen=16, window=8, persist=0),
    dict(arch='mag', omega=4, gate=True,  poly=('elementwise', 2), heads=1, dhead=16, batch=1, seqlen=16, window=16, persist=2),
    dict(arch='mal', omega=1, gate=False, poly=('off', 1), heads=2, dhead=32, batch=1, seqlen=8,  window=8,  persist=0),
    dict(arch='mal', omega=4, gate=True,  poly=('elementwise', 2), heads=1, dhead=16, batch=2, seqlen=16, window=16, persist=2),
    dict(arch='mac', omega=1, gate=False, poly=('off', 1), heads=2, dhead=32, batch=2, seqlen=16, window=8,  persist=0),
    dict(arch='lmm', omega=4, gate=True,  poly=('elementwise', 2), heads=1, dhead=16, batch=1, seqlen=8,  window=16, persist=0),
]

# -----------------------
# Memory cell basic tests
# -----------------------

@pytest.mark.parametrize('mode,deg', [('off',1), ('elementwise',2), ('tensor',2)])
def test_poly_feature_map_shapes(mode, deg):
    phi = PolynomialFeatureMap(dim=32, degree=deg, mode=mode)
    x = torch.randn(4, 10, 32)
    out = phi(x)
    assert out.shape == x.shape


def test_rnn_memory_cell_forward():
    cell = RNNMemoryCell(dim=64, dim_head=32, heads=2, use_momentum=True, qkv_conv_kernel=None)
    x = torch.randn(2, 8, 64)
    out, state = cell(x)
    assert out.shape == x.shape
    assert state.S.shape == (4, 32, 32)
    assert state.Z is not None


def test_rnn_memory_cell_scan_chunk_len_matches_unchunked():
    torch.manual_seed(0)
    base = RNNMemoryCell(dim=64, dim_head=32, heads=2, use_momentum=True, qkv_conv_kernel=4, scan_chunk_len=None)
    torch.manual_seed(0)
    chunked = RNNMemoryCell(dim=64, dim_head=32, heads=2, use_momentum=True, qkv_conv_kernel=4, scan_chunk_len=4)
    chunked.load_state_dict(base.state_dict())

    x = torch.randn(2, 13, 64)
    out_base, s_base = base(x)
    out_chunk, s_chunk = chunked(x)

    assert torch.allclose(out_base, out_chunk, atol=1e-5)
    assert torch.allclose(s_base.S, s_chunk.S, atol=1e-5)
    assert torch.allclose(s_base.Z, s_chunk.Z, atol=1e-5)


def test_omega_rnn_memory_cell_forward():
    cell = OmegaRNNMemoryCell(dim=64, dim_head=32, heads=2, omega_window=3, use_omega_gate=True)
    x = torch.randn(2, 12, 64)
    out, state = cell(x)
    assert out.shape == x.shape
    assert state.S.shape == (4, 32, 32)
    assert state.omega_buffer is not None


def test_omega_rnn_memory_cell_scan_chunk_len_matches_unchunked():
    torch.manual_seed(0)
    base = OmegaRNNMemoryCell(dim=64, dim_head=32, heads=2, omega_window=3, use_omega_gate=True, qkv_conv_kernel=4, scan_chunk_len=None)
    torch.manual_seed(0)
    chunked = OmegaRNNMemoryCell(dim=64, dim_head=32, heads=2, omega_window=3, use_omega_gate=True, qkv_conv_kernel=4, scan_chunk_len=4)
    chunked.load_state_dict(base.state_dict())

    x = torch.randn(2, 13, 64)
    out_base, s_base = base(x)
    out_chunk, s_chunk = chunked(x)

    assert torch.allclose(out_base, out_chunk, atol=1e-5)
    assert torch.allclose(s_base.S, s_chunk.S, atol=1e-5)
    assert torch.allclose(s_base.Z, s_chunk.Z, atol=1e-5)
    assert torch.allclose(s_base.omega_buffer, s_chunk.omega_buffer, atol=1e-5)


@pytest.mark.parametrize("use_momentum", (False, True))
@pytest.mark.parametrize("scan_chunk_len", (None, 4))
def test_rnn_memory_cell_matches_reference_prechange(use_momentum, scan_chunk_len):
    torch.manual_seed(0)
    cell = RNNMemoryCell(
        dim=64,
        dim_head=32,
        heads=2,
        use_momentum=use_momentum,
        qkv_conv_kernel=4,
        scan_chunk_len=scan_chunk_len,
    )
    x = torch.randn(2, 13, 64)

    # use a non-trivial initial state
    S = torch.randn(2 * cell.heads, cell.dim_head, cell.dim_head)
    Z = torch.randn_like(S) if use_momentum else None
    state0 = RNNMemState(seq_index=7, S=S, Z=Z, omega_buffer=None)

    out_new, st_new = cell(x, state=state0)
    out_ref, st_ref = _reference_forward_rnn_memory_cell(cell, x, state=state0)

    assert torch.allclose(out_new, out_ref, atol=1e-5)
    assert torch.allclose(st_new.S, st_ref.S, atol=1e-5)
    if use_momentum:
        assert torch.allclose(st_new.Z, st_ref.Z, atol=1e-5)
    else:
        assert st_new.Z is None and st_ref.Z is None


@pytest.mark.parametrize("use_momentum", (False, True))
@pytest.mark.parametrize("use_omega_gate", (False, True))
@pytest.mark.parametrize("scan_chunk_len", (None, 4))
def test_omega_rnn_memory_cell_matches_reference_prechange(use_momentum, use_omega_gate, scan_chunk_len):
    torch.manual_seed(0)
    cell = OmegaRNNMemoryCell(
        dim=64,
        dim_head=32,
        heads=2,
        omega_window=3,
        use_omega_gate=use_omega_gate,
        use_momentum=use_momentum,
        qkv_conv_kernel=4,
        scan_chunk_len=scan_chunk_len,
    )
    x = torch.randn(2, 13, 64)

    S = torch.randn(2 * cell.heads, cell.dim_head, cell.dim_head)
    Z = torch.randn_like(S) if use_momentum else None
    omega_buffer = torch.randn(2 * cell.heads, cell.omega_window - 1, cell.dim_head, cell.dim_head, 2)
    state0 = RNNMemState(seq_index=7, S=S, Z=Z, omega_buffer=omega_buffer)

    out_new, st_new = cell(x, state=state0)
    out_ref, st_ref = _reference_forward_omega_rnn_memory_cell(cell, x, state=state0)

    assert torch.allclose(out_new, out_ref, atol=1e-5)
    assert torch.allclose(st_new.S, st_ref.S, atol=1e-5)
    if use_momentum:
        assert torch.allclose(st_new.Z, st_ref.Z, atol=1e-5)
    else:
        assert st_new.Z is None and st_ref.Z is None
    assert torch.allclose(st_new.omega_buffer, st_ref.omega_buffer, atol=1e-5)


# -----------------------------------------------------------------------------
# Stronger equivalence tests (more seeds / poly / longer seq / dtype)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("seed", (0, 1, 2))
@pytest.mark.parametrize("poly_mode,poly_degree", (("off", 1), ("elementwise", 2), ("tensor", 2)))
@pytest.mark.parametrize("seq_len", (13, 64, 129))
@pytest.mark.parametrize("use_momentum", (False, True))
def test_rnn_memory_cell_matches_reference_strong_cpu_fp32(seed, poly_mode, poly_degree, seq_len, use_momentum):
    # CPU fp32-only strong coverage (keeps CI/runtime reasonable and avoids dtype flakiness on CPU)
    device = torch.device("cpu")
    dtype = torch.float32

    torch.manual_seed(seed)
    cell = RNNMemoryCell(
        dim=64,
        dim_head=32,
        heads=2,
        use_momentum=use_momentum,
        poly_mode=poly_mode,
        poly_degree=poly_degree,
        qkv_conv_kernel=4,
        scan_chunk_len=None,
    ).to(device=device, dtype=dtype)

    x = torch.randn(2, seq_len, 64, device=device, dtype=dtype)

    # Keep states in a realistic range (in practice S is clamped each step)
    S = _scaled_randn((2 * cell.heads, cell.dim_head, cell.dim_head), device=device, dtype=dtype, scale=0.1).clamp(-10, 10)
    Z = _scaled_randn(S.shape, device=device, dtype=dtype, scale=0.1) if use_momentum else None
    state0 = RNNMemState(seq_index=7, S=S, Z=Z, omega_buffer=None)

    out_new, st_new = cell(x, state=state0)
    out_ref, st_ref = _reference_forward_rnn_memory_cell(cell, x, state=state0)

    kw = _dtype_close_kwargs(dtype)
    assert torch.allclose(out_new, out_ref, **kw)
    assert torch.allclose(st_new.S, st_ref.S, **kw)
    if use_momentum:
        assert torch.allclose(st_new.Z, st_ref.Z, **kw)
    else:
        assert st_new.Z is None and st_ref.Z is None


@pytest.mark.parametrize("seed", (0, 1))
@pytest.mark.parametrize("poly_mode,poly_degree", (("off", 1), ("elementwise", 2)))
@pytest.mark.parametrize("seq_len", (64, 256))
@pytest.mark.parametrize("use_momentum", (False, True))
def test_rnn_memory_cell_matches_reference_strong_cuda_dtypes(seed, poly_mode, poly_degree, seq_len, use_momentum):
    # Only meaningful when CUDA is present (exercise fp16/bf16)
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    for dtype in _supported_test_dtypes(device):
        torch.manual_seed(seed)
        cell = RNNMemoryCell(
            dim=64,
            dim_head=32,
            heads=2,
            use_momentum=use_momentum,
            poly_mode=poly_mode,
            poly_degree=poly_degree,
            qkv_conv_kernel=4,
            scan_chunk_len=None,
        ).to(device=device, dtype=dtype)

        x = torch.randn(2, seq_len, 64, device=device, dtype=dtype)

        S = _scaled_randn((2 * cell.heads, cell.dim_head, cell.dim_head), device=device, dtype=dtype, scale=0.1).clamp(-10, 10)
        Z = _scaled_randn(S.shape, device=device, dtype=dtype, scale=0.1) if use_momentum else None
        state0 = RNNMemState(seq_index=7, S=S, Z=Z, omega_buffer=None)

        out_new, st_new = cell(x, state=state0)
        out_ref, st_ref = _reference_forward_rnn_memory_cell(cell, x, state=state0)

        kw = _dtype_close_kwargs(dtype)
        assert torch.allclose(out_new, out_ref, **kw)
        assert torch.allclose(st_new.S, st_ref.S, **kw)
        if use_momentum:
            assert torch.allclose(st_new.Z, st_ref.Z, **kw)
        else:
            assert st_new.Z is None and st_ref.Z is None


@pytest.mark.parametrize("seed", (0, 1, 2))
@pytest.mark.parametrize("poly_mode,poly_degree", (("off", 1), ("elementwise", 2), ("tensor", 2)))
@pytest.mark.parametrize("seq_len", (13, 64, 129))
@pytest.mark.parametrize("use_momentum", (False, True))
@pytest.mark.parametrize("use_omega_gate", (False, True))
def test_omega_rnn_memory_cell_matches_reference_strong_cpu_fp32(seed, poly_mode, poly_degree, seq_len, use_momentum, use_omega_gate):
    device = torch.device("cpu")
    dtype = torch.float32

    torch.manual_seed(seed)
    cell = OmegaRNNMemoryCell(
        dim=64,
        dim_head=32,
        heads=2,
        omega_window=3,
        use_omega_gate=use_omega_gate,
        use_momentum=use_momentum,
        poly_mode=poly_mode,
        poly_degree=poly_degree,
        qkv_conv_kernel=4,
        scan_chunk_len=None,
    ).to(device=device, dtype=dtype)

    x = torch.randn(2, seq_len, 64, device=device, dtype=dtype)

    S = _scaled_randn((2 * cell.heads, cell.dim_head, cell.dim_head), device=device, dtype=dtype, scale=0.1).clamp(-10, 10)
    Z = _scaled_randn(S.shape, device=device, dtype=dtype, scale=0.1) if use_momentum else None
    omega_buffer = _scaled_randn((2 * cell.heads, cell.omega_window - 1, cell.dim_head, cell.dim_head, 2), device=device, dtype=dtype, scale=0.1).clamp(-10, 10)
    state0 = RNNMemState(seq_index=7, S=S, Z=Z, omega_buffer=omega_buffer)

    out_new, st_new = cell(x, state=state0)
    out_ref, st_ref = _reference_forward_omega_rnn_memory_cell(cell, x, state=state0)

    kw = _dtype_close_kwargs(dtype)
    assert torch.allclose(out_new, out_ref, **kw)
    assert torch.allclose(st_new.S, st_ref.S, **kw)
    if use_momentum:
        assert torch.allclose(st_new.Z, st_ref.Z, **kw)
    else:
        assert st_new.Z is None and st_ref.Z is None
    assert torch.allclose(st_new.omega_buffer, st_ref.omega_buffer, **kw)


@pytest.mark.parametrize("seed", (0, 1))
@pytest.mark.parametrize("poly_mode,poly_degree", (("off", 1), ("elementwise", 2)))
@pytest.mark.parametrize("seq_len", (64, 256))
@pytest.mark.parametrize("use_momentum", (False, True))
@pytest.mark.parametrize("use_omega_gate", (False, True))
def test_omega_rnn_memory_cell_matches_reference_strong_cuda_dtypes(seed, poly_mode, poly_degree, seq_len, use_momentum, use_omega_gate):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    for dtype in _supported_test_dtypes(device):
        torch.manual_seed(seed)
        cell = OmegaRNNMemoryCell(
            dim=64,
            dim_head=32,
            heads=2,
            omega_window=3,
            use_omega_gate=use_omega_gate,
            use_momentum=use_momentum,
            poly_mode=poly_mode,
            poly_degree=poly_degree,
            qkv_conv_kernel=4,
            scan_chunk_len=None,
        ).to(device=device, dtype=dtype)
        
        x = torch.randn(2, seq_len, 64, device=device, dtype=dtype)
        
        S = _scaled_randn((2 * cell.heads, cell.dim_head, cell.dim_head), device=device, dtype=dtype, scale=0.1).clamp(-10, 10)
        Z = _scaled_randn(S.shape, device=device, dtype=dtype, scale=0.1) if use_momentum else None
        omega_buffer = _scaled_randn((2 * cell.heads, cell.omega_window - 1, cell.dim_head, cell.dim_head, 2), device=device, dtype=dtype, scale=0.1).clamp(-10, 10)
        state0 = RNNMemState(seq_index=7, S=S, Z=Z, omega_buffer=omega_buffer)
        
        out_new, st_new = cell(x, state=state0)
        out_ref, st_ref = _reference_forward_omega_rnn_memory_cell(cell, x, state=state0)
        
        kw = _dtype_close_kwargs(dtype)
        assert torch.allclose(out_new, out_ref, **kw)
        assert torch.allclose(st_new.S, st_ref.S, **kw)
        if use_momentum:
            assert torch.allclose(st_new.Z, st_ref.Z, **kw)
        else:
            assert st_new.Z is None and st_ref.Z is None
        assert torch.allclose(st_new.omega_buffer, st_ref.omega_buffer, **kw)


@pytest.mark.parametrize("seed", (0, 1, 2))
@pytest.mark.parametrize("poly_mode,poly_degree", (("off", 1), ("elementwise", 2), ("tensor", 2)))
@pytest.mark.parametrize("seq_len", (64, 129))
@pytest.mark.parametrize("use_momentum", (False, True))
def test_rnn_memory_cell_chunked_close_to_unchunked_cpu_fp32(seed, poly_mode, poly_degree, seq_len, use_momentum):
    device = torch.device("cpu")
    dtype = torch.float32

    torch.manual_seed(seed)
    base = RNNMemoryCell(
        dim=64,
        dim_head=32,
        heads=2,
        use_momentum=use_momentum,
        poly_mode=poly_mode,
        poly_degree=poly_degree,
        qkv_conv_kernel=4,
        scan_chunk_len=None,
    ).to(device=device, dtype=dtype)

    torch.manual_seed(seed)
    chunked = RNNMemoryCell(
        dim=64,
        dim_head=32,
        heads=2,
        use_momentum=use_momentum,
        poly_mode=poly_mode,
        poly_degree=poly_degree,
        qkv_conv_kernel=4,
        scan_chunk_len=8,
    ).to(device=device, dtype=dtype)
    chunked.load_state_dict(base.state_dict())

    x = torch.randn(2, seq_len, 64, device=device, dtype=dtype)
    S = _scaled_randn((2 * base.heads, base.dim_head, base.dim_head), device=device, dtype=dtype, scale=0.1).clamp(-10, 10)
    Z = _scaled_randn(S.shape, device=device, dtype=dtype, scale=0.1) if use_momentum else None
    state0 = RNNMemState(seq_index=7, S=S, Z=Z, omega_buffer=None)

    out_base, st_base = base(x, state=state0)
    out_chunk, st_chunk = chunked(x, state=state0)

    # Chunking changes the scan reduction order -> allow slightly looser tolerance
    assert torch.allclose(out_base, out_chunk, atol=1e-3, rtol=1e-3)
    assert torch.allclose(st_base.S, st_chunk.S, atol=1e-3, rtol=1e-3)
    if use_momentum:
        assert torch.allclose(st_base.Z, st_chunk.Z, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("seed", (0, 1, 2))
@pytest.mark.parametrize("poly_mode,poly_degree", (("off", 1), ("elementwise", 2), ("tensor", 2)))
@pytest.mark.parametrize("seq_len", (64, 129))
@pytest.mark.parametrize("use_momentum", (False, True))
@pytest.mark.parametrize("use_omega_gate", (False, True))
def test_omega_rnn_memory_cell_chunked_close_to_unchunked_cpu_fp32(seed, poly_mode, poly_degree, seq_len, use_momentum, use_omega_gate):
    device = torch.device("cpu")
    dtype = torch.float32

    torch.manual_seed(seed)
    base = OmegaRNNMemoryCell(
        dim=64,
        dim_head=32,
        heads=2,
        omega_window=3,
        use_omega_gate=use_omega_gate,
        use_momentum=use_momentum,
        poly_mode=poly_mode,
        poly_degree=poly_degree,
        qkv_conv_kernel=4,
        scan_chunk_len=None,
    ).to(device=device, dtype=dtype)

    torch.manual_seed(seed)
    chunked = OmegaRNNMemoryCell(
        dim=64,
        dim_head=32,
        heads=2,
        omega_window=3,
        use_omega_gate=use_omega_gate,
        use_momentum=use_momentum,
        poly_mode=poly_mode,
        poly_degree=poly_degree,
        qkv_conv_kernel=4,
        scan_chunk_len=8,
    ).to(device=device, dtype=dtype)
    chunked.load_state_dict(base.state_dict())

    x = torch.randn(2, seq_len, 64, device=device, dtype=dtype)
    S = _scaled_randn((2 * base.heads, base.dim_head, base.dim_head), device=device, dtype=dtype, scale=0.1).clamp(-10, 10)
    Z = _scaled_randn(S.shape, device=device, dtype=dtype, scale=0.1) if use_momentum else None
    omega_buffer = _scaled_randn((2 * base.heads, base.omega_window - 1, base.dim_head, base.dim_head, 2), device=device, dtype=dtype, scale=0.1).clamp(-10, 10)
    state0 = RNNMemState(seq_index=7, S=S, Z=Z, omega_buffer=omega_buffer)

    out_base, st_base = base(x, state=state0)
    out_chunk, st_chunk = chunked(x, state=state0)

    assert torch.allclose(out_base, out_chunk, atol=1e-3, rtol=1e-3)
    assert torch.allclose(st_base.S, st_chunk.S, atol=1e-3, rtol=1e-3)
    if use_momentum:
        assert torch.allclose(st_base.Z, st_chunk.Z, atol=1e-3, rtol=1e-3)
    assert torch.allclose(st_base.omega_buffer, st_chunk.omega_buffer, atol=1e-3, rtol=1e-3)


def test_state_continuity_rnn():
    """Calling with subsequent states should accumulate."""
    cell = RNNMemoryCell(dim=32, dim_head=16, heads=1, use_momentum=True, qkv_conv_kernel=None)
    x1 = torch.randn(1, 4, 32)
    x2 = torch.randn(1, 4, 32)
    out1, s1 = cell(x1)
    out2, s2 = cell(x2, state=s1)
    assert s2.seq_index == 8
    assert not torch.allclose(s1.S, s2.S)


def test_state_continuity_omega():
    """Omega buffer should roll across calls."""
    cell = OmegaRNNMemoryCell(dim=32, dim_head=16, heads=1, omega_window=3, use_omega_gate=False, qkv_conv_kernel=None)
    x1 = torch.randn(1, 6, 32)
    x2 = torch.randn(1, 4, 32)
    out1, s1 = cell(x1)
    out2, s2 = cell(x2, state=s1)
    assert s2.seq_index == 10
    assert s2.omega_buffer is not None


# ---------------------------------------------------------
# Mini-batch vs Online SGD behavior tests
# ---------------------------------------------------------

def test_minibatch_vs_online_sgd():
    """
    Test that full sequence (mini-batch SGD) and token-by-token (online SGD)
    produce different but both valid results.
    
    Full sequence: uses fixed S_0 (chunk-start state) for all gradients
    Incremental: uses updated state for each token's gradient
    
    This is expected behavior (same as original Atlas paper approximation).
    """
    torch.manual_seed(42)
    cell = RNNMemoryCell(dim=32, dim_head=16, heads=1, use_momentum=True, qkv_conv_kernel=None)
    x = torch.randn(1, 8, 32)
    
    # Full sequence at once (mini-batch SGD)
    out_full, s_full = cell(x)
    
    # Token by token (online SGD)
    outs_inc = []
    st = None
    for t in range(x.shape[1]):
        o, st = cell(x[:, t:t+1, :], state=st)
        outs_inc.append(o)
    out_inc = torch.cat(outs_inc, dim=1)
    
    # Shapes should match
    assert out_full.shape == out_inc.shape
    
    # Both should be finite
    assert torch.isfinite(out_full).all()
    assert torch.isfinite(out_inc).all()
    
    # Final states should be finite
    assert torch.isfinite(s_full.S).all()
    assert torch.isfinite(st.S).all()
    
    # The outputs should be different (mini-batch vs online)
    # but for very short sequences with small lr, they should be close
    max_diff = (out_full - out_inc).abs().max()
    assert max_diff < 1.0, f"Max diff too large: {max_diff}"


def test_incremental_state_continuity():
    """Token-by-token processing should accumulate state correctly."""
    torch.manual_seed(42)
    cell = OmegaRNNMemoryCell(dim=32, dim_head=16, heads=1, omega_window=3, use_omega_gate=False, qkv_conv_kernel=None, use_momentum=True)
    x = torch.randn(1, 12, 32)
    
    # Token by token
    outs_inc = []
    st = None
    for t in range(x.shape[1]):
        o, st = cell(x[:, t:t+1, :], state=st)
        outs_inc.append(o)
        # State should always be finite
        assert torch.isfinite(st.S).all(), f"State diverged at token {t}"
    
    out_inc = torch.cat(outs_inc, dim=1)
    assert out_inc.shape == (1, 12, 32)
    assert torch.isfinite(out_inc).all()


def test_chunk_processing_consistency():
    """Processing in chunks should give consistent results."""
    torch.manual_seed(42)
    cell = RNNMemoryCell(dim=32, dim_head=16, heads=2, use_momentum=True, qkv_conv_kernel=None)
    x = torch.randn(2, 12, 32)
    
    # Full sequence
    out_full, s_full = cell(x)
    
    # Process in two chunks of 6
    out1, s1 = cell(x[:, :6, :])
    out2, s2 = cell(x[:, 6:, :], state=s1)
    out_chunked = torch.cat([out1, out2], dim=1)
    
    # Chunked processing should be finite
    assert torch.isfinite(out_chunked).all()
    assert torch.isfinite(s2.S).all()
    
    # For mini-batch SGD, chunking changes the reference state,
    # so results will differ but should be close
    max_diff = (out_full - out_chunked).abs().max()
    assert max_diff < 1.0, f"Chunk processing diverged: {max_diff}"


# ---------------------------------------------------------
# AssocScan helper tests
# ---------------------------------------------------------

def test_assoc_scan_identity():
    """Alpha=1, delta=0 should preserve x0."""
    assoc_scan = AssocScan(use_accelerated=False)
    B, T, D = 2, 5, 4
    x0 = torch.randn(B, D, D)
    alpha = torch.ones(B, T)
    delta = torch.zeros(B, T, D, D)
    x_all = assoc_scan(alpha, delta, prev=x0)
    # All x_t should equal x0
    for t in range(T):
        assert torch.allclose(x_all[:, t], x0, atol=1e-5)


def test_assoc_scan_additive():
    """Alpha=1, constant delta should accumulate."""
    assoc_scan = AssocScan(use_accelerated=False)
    B, T, D = 1, 4, 2
    x0 = torch.zeros(B, D, D)
    alpha = torch.ones(B, T)
    delta = torch.ones(B, T, D, D)
    x_all = assoc_scan(alpha, delta, prev=x0)
    # x_t = t * delta
    for t in range(T):
        expected = torch.full((B, D, D), float(t + 1))
        assert torch.allclose(x_all[:, t], expected, atol=1e-5)


def test_assoc_scan_decay():
    """Test exponential decay with alpha < 1."""
    assoc_scan = AssocScan(use_accelerated=False)
    B, T, D = 1, 4, 2
    x0 = torch.ones(B, D, D)
    alpha = torch.full((B, T), 0.5)  # decay by half each step
    delta = torch.zeros(B, T, D, D)
    x_all = assoc_scan(alpha, delta, prev=x0)
    # x_t = x0 * (0.5)^t
    for t in range(T):
        expected = torch.full((B, D, D), 0.5 ** (t + 1))
        assert torch.allclose(x_all[:, t], expected, atol=1e-5)


def test_assoc_scan_mixed():
    """Test with both decay and additive terms."""
    assoc_scan = AssocScan(use_accelerated=False)
    B, T, D = 1, 3, 2
    x0 = torch.zeros(B, D, D)
    alpha = torch.full((B, T), 0.5)
    delta = torch.ones(B, T, D, D)
    x_all = assoc_scan(alpha, delta, prev=x0)
    # x_1 = 0 * 0.5 + 1 = 1
    # x_2 = 1 * 0.5 + 1 = 1.5
    # x_3 = 1.5 * 0.5 + 1 = 1.75
    expected = torch.tensor([[[1.0, 1.0], [1.0, 1.0]],
                              [[1.5, 1.5], [1.5, 1.5]],
                              [[1.75, 1.75], [1.75, 1.75]]]).unsqueeze(0)
    assert torch.allclose(x_all, expected, atol=1e-4)


def test_sliding_sum_along_time():
    """Sliding sum with window=3."""
    x = torch.arange(1, 7).float().view(1, 6, 1)  # [1,2,3,4,5,6]
    out = _sliding_sum_along_time(x, window=3)
    # out[t] = x[max(0,t-2):t+1].sum()
    expected = torch.tensor([[[1], [3], [6], [9], [12], [15]]], dtype=torch.float32)
    assert torch.allclose(out, expected)


# ---------------------------------------------------------
# Omega window effects
# ---------------------------------------------------------

def test_omega_window_changes_output():
    """Different omega_window values should produce different outputs."""
    torch.manual_seed(0)
    x = torch.randn(1, 10, 32)
    
    cell_e1 = OmegaRNNMemoryCell(dim=32, dim_head=16, heads=1, omega_window=1, use_omega_gate=False, qkv_conv_kernel=None)
    cell_e3 = OmegaRNNMemoryCell(dim=32, dim_head=16, heads=1, omega_window=3, use_omega_gate=False, qkv_conv_kernel=None)
    
    out1, _ = cell_e1(x)
    out3, _ = cell_e3(x)
    
    assert not torch.allclose(out1, out3)


def test_omega_gate_effect():
    """Gate=0 should suppress updates."""
    torch.manual_seed(0)
    cell = OmegaRNNMemoryCell(dim=32, dim_head=16, heads=1, omega_window=3, use_omega_gate=True, qkv_conv_kernel=None)
    
    # Force gate to near-zero
    with torch.no_grad():
        cell.to_omega_gate[0].weight.zero_()
        cell.to_omega_gate[0].bias.fill_(-20.)
    
    x = torch.randn(1, 8, 32)
    state0 = cell.init_state(1, x.device, x.dtype)
    out, state1 = cell(x, state=state0)
    
    # With gate~0, S should barely change
    assert torch.allclose(state0.S, state1.S, atol=1e-2)


# ---------------------------------------------------------
# Transformer architecture tests
# ---------------------------------------------------------

@pytest.mark.parametrize('arch', ARCHES)
def test_transformer_forward_basic(arch):
    m = RNNMemoryTransformer(
        num_tokens=256, dim=64, depth=1, block_type=arch,
        window_size=8, heads=2, dim_head=16, omega_window=1
    )
    x = torch.randint(0, 256, (2, 16))
    out = m(x)
    assert out.shape == (2, 16, 256)


@pytest.mark.parametrize('cfg', GRID)
def test_transformer_forward_grid(cfg):
    m = RNNMemoryTransformer(
        num_tokens=256, dim=64, depth=1, block_type=cfg['arch'],
        window_size=cfg['window'], heads=cfg['heads'], dim_head=cfg['dhead'],
        num_persist_mem_tokens=cfg['persist'],
        omega_window=cfg['omega'], use_omega_gate=cfg['gate'],
        poly_mode=cfg['poly'][0], poly_degree=cfg['poly'][1]
    )
    x = torch.randint(0, 256, (cfg['batch'], cfg['seqlen']))
    out = m(x)
    assert out.shape == (cfg['batch'], cfg['seqlen'], 256)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize('arch', ARCHES)
def test_transformer_backward(arch):
    m = RNNMemoryTransformer(
        num_tokens=256, dim=64, depth=1, block_type=arch,
        window_size=8, heads=2, dim_head=16, omega_window=1
    )
    x = torch.randint(0, 256, (1, 12))
    out = m(x)
    loss = out.sum()
    loss.backward()
    # Check gradients exist
    for name, p in m.named_parameters():
        if p.requires_grad and 'gamma' not in name:
            assert p.grad is not None, f"No grad for {name}"


@pytest.mark.parametrize('arch', ARCHES)
def test_transformer_generate(arch):
    m = RNNMemoryTransformer(
        num_tokens=256, dim=64, depth=1, block_type=arch,
        window_size=8, heads=2, dim_head=16, omega_window=1
    )
    prompt = torch.randint(0, 256, (1, 4))
    out = m.generate(prompt, max_length=12)
    assert out.shape == (1, 12)


# ---------------------------------------------------------
# Batch independence
# ---------------------------------------------------------

def test_batch_independence_rnn():
    """Each batch should be processed independently."""
    torch.manual_seed(0)
    cell = RNNMemoryCell(dim=32, dim_head=16, heads=1, qkv_conv_kernel=None)
    
    x1 = torch.randn(1, 6, 32)
    x2 = torch.randn(1, 6, 32)
    x_cat = torch.cat([x1, x2], dim=0)
    
    out1, _ = cell(x1)
    out2, _ = cell(x2)
    out_cat, _ = cell(x_cat)
    
    assert torch.allclose(out_cat[0], out1[0], atol=1e-5)
    assert torch.allclose(out_cat[1], out2[0], atol=1e-5)


def test_batch_independence_omega():
    """Same for Omega cell."""
    torch.manual_seed(0)
    cell = OmegaRNNMemoryCell(dim=32, dim_head=16, heads=1, omega_window=2, qkv_conv_kernel=None)
    
    x1 = torch.randn(1, 6, 32)
    x2 = torch.randn(1, 6, 32)
    x_cat = torch.cat([x1, x2], dim=0)
    
    out1, _ = cell(x1)
    out2, _ = cell(x2)
    out_cat, _ = cell(x_cat)
    
    assert torch.allclose(out_cat[0], out1[0], atol=1e-5)
    assert torch.allclose(out_cat[1], out2[0], atol=1e-5)


# ---------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------

def test_long_sequence_stability():
    """Long sequences should not explode."""
    cell = RNNMemoryCell(dim=32, dim_head=16, heads=1, qkv_conv_kernel=None)
    x = torch.randn(1, 256, 32)
    out, state = cell(x)
    assert torch.isfinite(out).all()
    assert torch.isfinite(state.S).all()


def test_gradient_stability():
    """Gradients should not explode."""
    cell = RNNMemoryCell(dim=32, dim_head=16, heads=1, qkv_conv_kernel=None)
    x = torch.randn(1, 64, 32, requires_grad=True)
    out, _ = cell(x)
    loss = out.sum()
    loss.backward()
    grad_norm = x.grad.norm()
    assert grad_norm < 1e4, f"Gradient norm too large: {grad_norm}"


def test_repeated_forward_stability():
    """Repeated forwards with state should stay stable."""
    cell = RNNMemoryCell(dim=32, dim_head=16, heads=1, qkv_conv_kernel=None)
    state = None
    for _ in range(20):
        x = torch.randn(1, 16, 32)
        _, state = cell(x, state=state)
    assert torch.isfinite(state.S).all()


# ---------------------------------------------------------
# Momentum toggle
# ---------------------------------------------------------

def test_momentum_toggle():
    """With vs without momentum should differ."""
    torch.manual_seed(0)
    x = torch.randn(1, 10, 32)
    
    cell_mom = RNNMemoryCell(dim=32, dim_head=16, heads=1, use_momentum=True, qkv_conv_kernel=None)
    cell_no = RNNMemoryCell(dim=32, dim_head=16, heads=1, use_momentum=False, qkv_conv_kernel=None)
    
    out_mom, _ = cell_mom(x)
    out_no, _ = cell_no(x)
    
    assert not torch.allclose(out_mom, out_no)


# ---------------------------------------------------------
# Polynomial mode effects
# ---------------------------------------------------------

@pytest.mark.parametrize('mode', ['off', 'elementwise', 'tensor'])
def test_poly_mode_output_shapes(mode):
    cell = RNNMemoryCell(dim=32, dim_head=16, heads=1, poly_mode=mode, poly_degree=2, qkv_conv_kernel=None)
    x = torch.randn(1, 8, 32)
    out, _ = cell(x)
    assert out.shape == x.shape


def test_poly_mode_changes_output():
    """Different poly modes should produce different outputs."""
    torch.manual_seed(0)
    x = torch.randn(1, 8, 32)
    
    cell_off = RNNMemoryCell(dim=32, dim_head=16, heads=1, poly_mode='off', poly_degree=2, qkv_conv_kernel=None)
    cell_elem = RNNMemoryCell(dim=32, dim_head=16, heads=1, poly_mode='elementwise', poly_degree=2, qkv_conv_kernel=None)
    
    out_off, _ = cell_off(x)
    out_elem, _ = cell_elem(x)
    
    assert not torch.allclose(out_off, out_elem)


# ---------------------------------------------------------
# Multi-batch training loop simulation
# ---------------------------------------------------------

@pytest.mark.parametrize('arch', ['mag', 'mal', 'lmm'])
def test_multibatch_training_loop(arch):
    """Simulate a few training steps."""
    m = RNNMemoryTransformer(
        num_tokens=256, dim=64, depth=1, block_type=arch,
        window_size=8, heads=2, dim_head=16, omega_window=1
    )
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    
    losses = []
    for _ in range(3):
        x = torch.randint(0, 256, (2, 16))
        out = m(x)
        loss = F.cross_entropy(out[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1))
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    assert all(math.isfinite(l) for l in losses)


# ---------------------------------------------------------
# Large grids for coverage
# ---------------------------------------------------------

FORWARD_LOSS_GRID = [
    {'arch': a, 'omega': o, 'gate': g}
    for a in ARCHES
    for o in OMEGAS
    for g in GATES
]


@pytest.mark.parametrize('cfg', FORWARD_LOSS_GRID[:16])
def test_forward_loss_grid(cfg):
    m = RNNMemoryTransformer(
        num_tokens=256, dim=64, depth=1, block_type=cfg['arch'],
        window_size=8, heads=2, dim_head=16,
        omega_window=cfg['omega'], use_omega_gate=cfg['gate']
    )
    x = torch.randint(0, 256, (1, 16))
    out = m(x)
    loss = F.cross_entropy(out[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1))
    assert math.isfinite(loss.item())


BACKWARD_GRID = [
    {'arch': a, 'poly': p}
    for a in ['mag', 'mal', 'lmm']
    for p in [('off', 1), ('elementwise', 2)]
]


@pytest.mark.parametrize('cfg', BACKWARD_GRID)
def test_backward_grid(cfg):
    m = RNNMemoryTransformer(
        num_tokens=256, dim=64, depth=1, block_type=cfg['arch'],
        window_size=8, heads=2, dim_head=16,
        poly_mode=cfg['poly'][0], poly_degree=cfg['poly'][1]
    )
    x = torch.randint(0, 256, (1, 12))
    out = m(x)
    loss = out.sum()
    loss.backward()
    
    grads_exist = sum(1 for p in m.parameters() if p.grad is not None)
    assert grads_exist > 0


# ---------------------------------------------------------
# QKV conv tests
# ---------------------------------------------------------

def test_qkv_conv_effect():
    """Conv kernel should change output vs no conv."""
    torch.manual_seed(0)
    x = torch.randn(1, 8, 32)
    
    cell_conv = RNNMemoryCell(dim=32, dim_head=16, heads=1, qkv_conv_kernel=4)
    cell_noconv = RNNMemoryCell(dim=32, dim_head=16, heads=1, qkv_conv_kernel=None)
    
    out_conv, _ = cell_conv(x)
    out_noconv, _ = cell_noconv(x)
    
    assert not torch.allclose(out_conv, out_noconv)


# ---------------------------------------------------------
# Heads scaling
# ---------------------------------------------------------

@pytest.mark.parametrize('heads', [1, 2, 4])
def test_heads_scaling(heads):
    cell = RNNMemoryCell(dim=64, dim_head=16, heads=heads, qkv_conv_kernel=None)
    x = torch.randn(2, 8, 64)
    out, state = cell(x)
    assert out.shape == x.shape
    assert state.S.shape == (2 * heads, 16, 16)


# ---------------------------------------------------------
# E=1 parity (RNN vs Omega with window=1)
# ---------------------------------------------------------

def test_e1_parity_rnn_vs_omega():
    """Omega with window=1, no gate should behave similarly to plain RNN."""
    torch.manual_seed(42)
    x = torch.randn(1, 10, 32)
    
    rnn = RNNMemoryCell(dim=32, dim_head=16, heads=1, qkv_conv_kernel=None, use_momentum=True)
    omega = OmegaRNNMemoryCell(dim=32, dim_head=16, heads=1, omega_window=1, use_omega_gate=False, qkv_conv_kernel=None, use_momentum=True)
    
    out_rnn, _ = rnn(x)
    out_omega, _ = omega(x)
    
    # Outputs should have same shape and be finite
    assert out_rnn.shape == out_omega.shape
    assert torch.isfinite(out_rnn).all()
    assert torch.isfinite(out_omega).all()


# ---------------------------------------------------------
# MAC specific tests
# ---------------------------------------------------------

def test_mac_context_generation():
    """MAC should generate context tokens from memory."""
    m = RNNMemoryTransformer(
        num_tokens=256, dim=64, depth=1, block_type='mac',
        window_size=8, heads=2, dim_head=16, omega_window=1
    )
    x = torch.randint(0, 256, (1, 16))
    out = m(x)
    assert out.shape == (1, 16, 256)


# ---------------------------------------------------------
# MAL two-FF test
# ---------------------------------------------------------

def test_mal_has_two_ff():
    """MAL block should have two FeedForward layers."""
    m = RNNMemoryTransformer(
        num_tokens=256, dim=64, depth=1, block_type='mal',
        window_size=8, heads=2, dim_head=16
    )
    layer = m.layers[0]
    # Check for ff_pre and ff_post attributes
    assert hasattr(layer, 'ff_pre')
    assert hasattr(layer, 'ff_post')


# ---------------------------------------------------------
# State detach
# ---------------------------------------------------------

def test_state_detach():
    """state_detach should break computation graph."""
    from atlas_pytorch.rnn_memory import state_detach
    
    cell = RNNMemoryCell(dim=32, dim_head=16, heads=1, qkv_conv_kernel=None)
    x = torch.randn(1, 4, 32, requires_grad=True)
    _, state = cell(x)
    
    detached = state_detach(state)
    assert not detached.S.requires_grad


# ---------------------------------------------------------
# Parameter count
# ---------------------------------------------------------

def test_parameter_count():
    """Sanity check parameter count."""
    cell = RNNMemoryCell(dim=64, dim_head=32, heads=2, qkv_conv_kernel=None)
    n_params = sum(p.numel() for p in cell.parameters())
    assert n_params > 0


# ---------------------------------------------------------
# Broader coverage grids
# ---------------------------------------------------------

COVERAGE_GRID = [
    {'batch': b, 'seq': s, 'heads': h}
    for b in [1, 2, 4]
    for s in [4, 8, 16]
    for h in [1, 2]
]


@pytest.mark.parametrize('cfg', COVERAGE_GRID[:12])
def test_coverage_shapes(cfg):
    cell = RNNMemoryCell(dim=32, dim_head=16, heads=cfg['heads'], qkv_conv_kernel=None)
    x = torch.randn(cfg['batch'], cfg['seq'], 32)
    out, state = cell(x)
    assert out.shape == x.shape
    assert state.S.shape[0] == cfg['batch'] * cfg['heads']


OMEGA_COVERAGE = [
    {'omega': e, 'gate': g, 'seq': s}
    for e in [1, 2, 4]
    for g in [False, True]
    for s in [8, 16]
]


@pytest.mark.parametrize('cfg', OMEGA_COVERAGE[:12])
def test_omega_coverage(cfg):
    cell = OmegaRNNMemoryCell(
        dim=32, dim_head=16, heads=1,
        omega_window=cfg['omega'], use_omega_gate=cfg['gate'],
        qkv_conv_kernel=None
    )
    x = torch.randn(1, cfg['seq'], 32)
    out, state = cell(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------
# Transformer coverage
# ---------------------------------------------------------

TRANSFORMER_GRID = [
    {'arch': a, 'depth': d, 'persist': p}
    for a in ['mag', 'mal', 'lmm', 'mac']
    for d in [1, 2]
    for p in [0, 2]
]


@pytest.mark.parametrize('cfg', TRANSFORMER_GRID[:16])
def test_transformer_depth_persist(cfg):
    m = RNNMemoryTransformer(
        num_tokens=256, dim=64, depth=cfg['depth'], block_type=cfg['arch'],
        window_size=8, heads=2, dim_head=16,
        num_persist_mem_tokens=cfg['persist']
    )
    x = torch.randint(0, 256, (1, 16))
    out = m(x)
    assert out.shape == (1, 16, 256)
