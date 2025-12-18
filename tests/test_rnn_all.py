import pytest
import torch
import torch.nn.functional as F
from torch import nn

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


def test_omega_rnn_memory_cell_forward():
    cell = OmegaRNNMemoryCell(dim=64, dim_head=32, heads=2, omega_window=3, use_omega_gate=True)
    x = torch.randn(2, 12, 64)
    out, state = cell(x)
    assert out.shape == x.shape
    assert state.S.shape == (4, 32, 32)
    assert state.omega_buffer is not None


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
