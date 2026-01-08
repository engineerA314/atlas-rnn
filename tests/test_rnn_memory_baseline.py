"""
Baseline tests for RNN Memory before optimization.
Establishes ground truth for correctness and memory usage.
"""

import pytest
import torch
import torch.nn.functional as F
from atlas_pytorch.rnn_memory import RNNMemoryCell, OmegaRNNMemoryCell, RNNMemState


def get_memory_allocated():
    """Get current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def reset_memory():
    """Reset GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def dtype():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


class TestRNNMemoryCellBaseline:
    """Baseline tests for RNNMemoryCell (no omega)."""
    
    @pytest.mark.parametrize("batch,seq_len,dim,dim_head,heads", [
        (2, 64, 256, 64, 4),    # Small
        (2, 512, 512, 64, 8),   # Medium (Stage 1/2)
        (1, 1024, 512, 64, 8),  # Large sequence
    ])
    def test_forward_shape(self, batch, seq_len, dim, dim_head, heads, device, dtype):
        """Test output shape is correct."""
        cell = RNNMemoryCell(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            use_momentum=True,
            scan_chunk_len=None,  # No chunking for baseline
        ).to(device).to(dtype)
        
        x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
        state = cell.init_state(batch, device=device, dtype=dtype)
        
        out, next_state = cell(x, state)
        
        assert out.shape == (batch, seq_len, dim)
        assert next_state.S.shape == (batch * heads, dim_head, dim_head)
        assert next_state.Z.shape == (batch * heads, dim_head, dim_head)
        assert next_state.seq_index == seq_len
    
    def test_forward_backward_runs(self, device, dtype):
        """Test forward + backward completes without error."""
        batch, seq_len, dim, dim_head, heads = 2, 128, 256, 64, 4
        
        cell = RNNMemoryCell(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            use_momentum=True,
            scan_chunk_len=None,
        ).to(device)
        
        x = torch.randn(batch, seq_len, dim, device=device, requires_grad=True)
        state = cell.init_state(batch, device=device)
        
        out, next_state = cell(x, state)
        
        # Backward pass
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_state_propagation(self, device, dtype):
        """Test state correctly propagates across multiple forward calls."""
        batch, seq_len, dim, dim_head, heads = 2, 64, 256, 64, 4
        
        cell = RNNMemoryCell(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            use_momentum=True,
        ).to(device).to(dtype)
        
        # First sequence
        x1 = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
        state1 = cell.init_state(batch, device=device, dtype=dtype)
        out1, state2 = cell(x1, state1)
        
        # Second sequence (should use state from first)
        x2 = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
        out2, state3 = cell(x2, state2)
        
        assert state2.seq_index == seq_len
        assert state3.seq_index == seq_len * 2
        
        # States should have changed
        assert not torch.allclose(state1.S, state2.S, atol=1e-3)
        assert not torch.allclose(state2.S, state3.S, atol=1e-3)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for memory test")
    @pytest.mark.parametrize("seq_len", [512, 1024, 2048])
    def test_memory_usage_baseline(self, seq_len, device):
        """Measure peak memory usage (baseline before optimization)."""
        batch, dim, dim_head, heads = 2, 512, 64, 8
        
        reset_memory()
        
        cell = RNNMemoryCell(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            use_momentum=True,
            scan_chunk_len=None,  # No chunking
        ).to(device)
        
        x = torch.randn(batch, seq_len, dim, device=device, requires_grad=True)
        state = cell.init_state(batch, device=device)
        
        mem_before = get_memory_allocated()
        
        out, next_state = cell(x, state)
        loss = out.sum()
        loss.backward()
        
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        mem_after = get_memory_allocated()
        
        print(f"\n[Baseline] seq_len={seq_len}")
        print(f"  Memory before: {mem_before:.2f} MB")
        print(f"  Memory after: {mem_after:.2f} MB")
        print(f"  Peak memory: {peak_mem:.2f} MB")
        print(f"  Activation memory: {peak_mem - mem_before:.2f} MB")
        
        # Store baseline for comparison (will fail first time, update expected values)
        # This is just to track regression
        expected_peak_mb = {
            512: 1000,   # Placeholder - update after first run
            1024: 2000,  # Placeholder
            2048: 4000,  # Placeholder
        }
        
        # Assert memory is reasonable (will be improved in optimization)
        assert peak_mem < expected_peak_mb.get(seq_len, float('inf'))


class TestOmegaRNNMemoryCellBaseline:
    """Baseline tests for OmegaRNNMemoryCell (with omega window)."""
    
    @pytest.mark.parametrize("omega_window,use_omega_gate", [
        (1, False),   # No omega (should match RNNMemoryCell)
        (4, True),    # Standard omega
        (8, True),    # Larger window
    ])
    def test_forward_shape(self, omega_window, use_omega_gate, device, dtype):
        """Test output shape with different omega settings."""
        batch, seq_len, dim, dim_head, heads = 2, 128, 256, 64, 4
        
        cell = OmegaRNNMemoryCell(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            omega_window=omega_window,
            use_omega_gate=use_omega_gate,
            use_momentum=True,
            scan_chunk_len=None,
        ).to(device).to(dtype)
        
        x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
        state = cell.init_state(batch, device=device, dtype=dtype)
        
        out, next_state = cell(x, state)
        
        assert out.shape == (batch, seq_len, dim)
        assert next_state.S.shape == (batch * heads, dim_head, dim_head)
        
        if omega_window > 1:
            # After refactor: stores g only (no longer G+B), so no last dim
            expected_buf_shape = (batch * heads, omega_window - 1, dim_head, dim_head)
            assert next_state.omega_buffer.shape == expected_buf_shape
    
    def test_omega_buffer_propagation(self, device, dtype):
        """Test omega buffer correctly carries history across sequences."""
        batch, seq_len, dim, dim_head, heads = 2, 64, 256, 64, 4
        omega_window = 4
        
        cell = OmegaRNNMemoryCell(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            omega_window=omega_window,
            use_omega_gate=True,
            use_momentum=True,
        ).to(device).to(dtype)
        
        # First sequence
        x1 = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
        state1 = cell.init_state(batch, device=device, dtype=dtype)
        out1, state2 = cell(x1, state1)
        
        # Buffer should be populated
        assert state2.omega_buffer is not None
        assert not torch.allclose(state2.omega_buffer, torch.zeros_like(state2.omega_buffer))
        
        # Second sequence should use buffer
        x2 = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
        out2, state3 = cell(x2, state2)
        
        # Buffer should have changed
        assert not torch.allclose(state2.omega_buffer, state3.omega_buffer, atol=1e-3)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for memory test")
    @pytest.mark.parametrize("omega_window,seq_len", [
        (1, 512),   # No omega
        (4, 512),   # Standard
        (4, 1024),  # Longer sequence
    ])
    def test_memory_usage_baseline(self, omega_window, seq_len, device):
        """Measure peak memory with omega (expect higher than non-omega)."""
        batch, dim, dim_head, heads = 2, 512, 64, 8
        
        reset_memory()
        
        cell = OmegaRNNMemoryCell(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            omega_window=omega_window,
            use_omega_gate=(omega_window > 1),
            use_momentum=True,
            scan_chunk_len=None,
        ).to(device)
        
        x = torch.randn(batch, seq_len, dim, device=device, requires_grad=True)
        state = cell.init_state(batch, device=device)
        
        mem_before = get_memory_allocated()
        
        out, next_state = cell(x, state)
        loss = out.sum()
        loss.backward()
        
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"\n[Omega Baseline] omega={omega_window}, seq_len={seq_len}")
        print(f"  Peak memory: {peak_mem:.2f} MB")
        print(f"  Activation memory: {peak_mem - mem_before:.2f} MB")
        
        # Omega should use more memory than non-omega (due to G/B buffers)
        if omega_window > 1:
            # This will be improved in refactor
            pass


class TestMemoryComparison:
    """Direct comparison between RNN and Omega cells."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_omega1_vs_rnn_memory(self, device, dtype):
        """Omega(window=1, gate=False) should use similar memory to RNN."""
        batch, seq_len, dim, dim_head, heads = 2, 512, 512, 64, 8
        
        # RNN cell
        reset_memory()
        rnn_cell = RNNMemoryCell(
            dim=dim, dim_head=dim_head, heads=heads,
            use_momentum=True, scan_chunk_len=None,
        ).to(device)
        x = torch.randn(batch, seq_len, dim, device=device, requires_grad=True)
        state = rnn_cell.init_state(batch, device=device)
        out, _ = rnn_cell(x, state)
        loss = out.sum()
        loss.backward()
        rnn_peak = torch.cuda.max_memory_allocated() / 1024**2
        
        # Omega cell (omega=1, no gate)
        reset_memory()
        omega_cell = OmegaRNNMemoryCell(
            dim=dim, dim_head=dim_head, heads=heads,
            omega_window=1, use_omega_gate=False,
            use_momentum=True, scan_chunk_len=None,
        ).to(device)
        x = torch.randn(batch, seq_len, dim, device=device, requires_grad=True)
        state = omega_cell.init_state(batch, device=device)
        out, _ = omega_cell(x, state)
        loss = out.sum()
        loss.backward()
        omega_peak = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"\n[Memory Comparison]")
        print(f"  RNN peak: {rnn_peak:.2f} MB")
        print(f"  Omega(e=1) peak: {omega_peak:.2f} MB")
        print(f"  Difference: {omega_peak - rnn_peak:.2f} MB ({(omega_peak/rnn_peak - 1)*100:.1f}%)")
        
        # Should be very similar (within 20%)
        assert abs(omega_peak - rnn_peak) / rnn_peak < 0.20


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_rnn_memory_baseline.py -v -s
    pytest.main([__file__, "-v", "-s"])

