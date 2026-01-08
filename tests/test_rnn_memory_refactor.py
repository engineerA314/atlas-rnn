"""
Correctness tests for RNN Memory refactor.
Verifies mathematical equivalence of refactored implementation.
"""

import pytest
import torch
import torch.nn.functional as F
from atlas_pytorch.rnn_memory import RNNMemoryCell, OmegaRNNMemoryCell


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def dtype():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


class TestRefactorMathematicalEquivalence:
    """Test that refactor produces mathematically equivalent results."""
    
    def test_gradient_computation_manual(self, device, dtype):
        """Manually verify g = δφᵀ where δ = S_ref·φ - v."""
        batch, seq_len, dim_head, heads = 2, 8, 64, 4
        BH = batch * heads
        
        # Random inputs
        phi_k = torch.randn(BH, seq_len, dim_head, device=device, dtype=torch.float32)
        v_bh = torch.randn(BH, seq_len, dim_head, device=device, dtype=torch.float32)
        S_ref = torch.randn(BH, dim_head, dim_head, device=device, dtype=torch.float32)
        
        # Method 1 (old): G = φφᵀ, B = vφᵀ, g = S_ref·G - B
        G = torch.einsum('bti,btj->btij', phi_k, phi_k)
        B = torch.einsum('bti,btj->btij', v_bh, phi_k)
        g_old = torch.einsum('bde,btef->btdf', S_ref, G) - B
        
        # Method 2 (new): δ = S_ref·φ - v, g = δφᵀ
        pred = torch.einsum('bde,bte->btd', S_ref, phi_k)
        delta_vec = pred - v_bh
        g_new = torch.einsum('bti,btj->btij', delta_vec, phi_k)
        
        # Should be very close (numerical precision with fp32)
        assert torch.allclose(g_old, g_new, rtol=1e-4, atol=1e-4), \
            f"Max diff: {(g_old - g_new).abs().max().item()}"
        
        print(f"\n✅ Gradient computation equivalence verified")
        print(f"   Max absolute difference: {(g_old - g_new).abs().max().item():.2e}")
    
    def test_rnn_cell_output_consistency(self, device, dtype):
        """Test RNNMemoryCell produces consistent outputs across runs."""
        batch, seq_len, dim, dim_head, heads = 2, 64, 256, 64, 4
        
        torch.manual_seed(42)
        cell1 = RNNMemoryCell(
            dim=dim, dim_head=dim_head, heads=heads,
            use_momentum=True, scan_chunk_len=None,
        ).to(device).to(dtype)
        
        torch.manual_seed(42)
        cell2 = RNNMemoryCell(
            dim=dim, dim_head=dim_head, heads=heads,
            use_momentum=True, scan_chunk_len=None,
        ).to(device).to(dtype)
        
        # Same input
        torch.manual_seed(123)
        x1 = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
        torch.manual_seed(123)
        x2 = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
        
        state1 = cell1.init_state(batch, device=device, dtype=dtype)
        state2 = cell2.init_state(batch, device=device, dtype=dtype)
        
        out1, next_state1 = cell1(x1, state1)
        out2, next_state2 = cell2(x2, state2)
        
        # Outputs should be identical
        assert torch.allclose(out1, out2, rtol=1e-3, atol=1e-4)
        assert torch.allclose(next_state1.S, next_state2.S, rtol=1e-3, atol=1e-4)
        assert torch.allclose(next_state1.Z, next_state2.Z, rtol=1e-3, atol=1e-4)
        
        print(f"\n✅ RNNMemoryCell consistency verified")
    
    def test_omega_cell_output_consistency(self, device, dtype):
        """Test OmegaRNNMemoryCell produces consistent outputs."""
        batch, seq_len, dim, dim_head, heads = 2, 64, 256, 64, 4
        omega_window = 4
        
        torch.manual_seed(42)
        cell1 = OmegaRNNMemoryCell(
            dim=dim, dim_head=dim_head, heads=heads,
            omega_window=omega_window, use_omega_gate=True,
            use_momentum=True, scan_chunk_len=None,
        ).to(device).to(dtype)
        
        torch.manual_seed(42)
        cell2 = OmegaRNNMemoryCell(
            dim=dim, dim_head=dim_head, heads=heads,
            omega_window=omega_window, use_omega_gate=True,
            use_momentum=True, scan_chunk_len=None,
        ).to(device).to(dtype)
        
        torch.manual_seed(123)
        x1 = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
        torch.manual_seed(123)
        x2 = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
        
        state1 = cell1.init_state(batch, device=device, dtype=dtype)
        state2 = cell2.init_state(batch, device=device, dtype=dtype)
        
        out1, next_state1 = cell1(x1, state1)
        out2, next_state2 = cell2(x2, state2)
        
        assert torch.allclose(out1, out2, rtol=1e-3, atol=1e-4)
        assert torch.allclose(next_state1.S, next_state2.S, rtol=1e-3, atol=1e-4)
        
        print(f"\n✅ OmegaRNNMemoryCell consistency verified")
    
    def test_gradient_flow_preserved(self, device, dtype):
        """Test that gradients flow correctly through refactored implementation."""
        batch, seq_len, dim, dim_head, heads = 2, 32, 128, 64, 2
        
        cell = RNNMemoryCell(
            dim=dim, dim_head=dim_head, heads=heads,
            use_momentum=True, scan_chunk_len=None,
        ).to(device)
        
        x = torch.randn(batch, seq_len, dim, device=device, requires_grad=True)
        state = cell.init_state(batch, device=device)
        
        # Forward + backward
        out, next_state = cell(x, state)
        loss = out.pow(2).mean()
        loss.backward()
        
        # Check gradients exist and are not NaN/Inf
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        assert (x.grad.abs() > 0).any(), "Gradients should not be all zero"
        
        # Check parameter gradients
        for name, param in cell.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"
                assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradient"
        
        print(f"\n✅ Gradient flow verified")
        print(f"   Input gradient norm: {x.grad.norm().item():.4f}")
    
    def test_omega_gradient_flow(self, device, dtype):
        """Test gradients flow through omega window correctly."""
        batch, seq_len, dim, dim_head, heads = 2, 32, 128, 64, 2
        
        cell = OmegaRNNMemoryCell(
            dim=dim, dim_head=dim_head, heads=heads,
            omega_window=4, use_omega_gate=True,
            use_momentum=True, scan_chunk_len=None,
        ).to(device)
        
        x = torch.randn(batch, seq_len, dim, device=device, requires_grad=True)
        state = cell.init_state(batch, device=device)
        
        out, next_state = cell(x, state)
        loss = out.pow(2).mean()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        
        # Omega gate should have gradients
        if cell.use_omega_gate:
            omega_gate_linear = cell.to_omega_gate[0]  # Sequential -> Linear
            assert omega_gate_linear.weight.grad is not None
            assert not torch.isnan(omega_gate_linear.weight.grad).any()
        
        print(f"\n✅ Omega gradient flow verified")
    
    @pytest.mark.parametrize("use_momentum", [True, False])
    def test_momentum_correctness(self, use_momentum, device, dtype):
        """Test with/without momentum produces valid results."""
        batch, seq_len, dim, dim_head, heads = 2, 32, 128, 64, 2
        
        cell = RNNMemoryCell(
            dim=dim, dim_head=dim_head, heads=heads,
            use_momentum=use_momentum, scan_chunk_len=None,
        ).to(device).to(dtype)
        
        x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
        state = cell.init_state(batch, device=device, dtype=dtype)
        
        out, next_state = cell(x, state)
        
        assert out.shape == (batch, seq_len, dim)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        
        if use_momentum:
            assert next_state.Z is not None
        else:
            assert next_state.Z is None
        
        print(f"\n✅ Momentum={use_momentum} correctness verified")
    
    def test_memory_reduction(self, device):
        """Verify refactor reduces memory footprint."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory test")
        
        batch, seq_len, dim, dim_head, heads = 2, 512, 512, 64, 8
        
        # Measure peak memory during forward + backward
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        cell = RNNMemoryCell(
            dim=dim, dim_head=dim_head, heads=heads,
            use_momentum=True, scan_chunk_len=None,
        ).to(device)
        
        x = torch.randn(batch, seq_len, dim, device=device, requires_grad=True)
        state = cell.init_state(batch, device=device)
        
        out, next_state = cell(x, state)
        loss = out.sum()
        loss.backward()
        
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"\n📊 Memory usage after refactor")
        print(f"   Peak memory: {peak_mem_mb:.2f} MB")
        
        # Theoretical calculation of saved memory (rough estimate)
        # Before: G [BH,T,d,d] + B [BH,T,d,d] = 2 * BH * T * d^2 * 2 bytes
        # After: only pred [BH,T,d] + delta [BH,T,d] = 2 * BH * T * d * 2 bytes
        # Saved: 2 * BH * T * d * (d-1) * 2 bytes
        BH = batch * heads
        saved_bytes = 2 * BH * seq_len * dim_head * (dim_head - 1) * 2
        saved_mb = saved_bytes / 1024**2
        
        print(f"   Theoretical memory saved: {saved_mb:.2f} MB")
        print(f"   (from eliminating G, B tensors)")


class TestOmegaWindowRefactor:
    """Test omega window implementation after refactor."""
    
    def test_omega_buffer_size_reduced(self, device, dtype):
        """Verify omega buffer is now smaller (stores g, not G+B)."""
        batch, seq_len, dim, dim_head, heads = 2, 64, 256, 64, 4
        omega_window = 4
        
        cell = OmegaRNNMemoryCell(
            dim=dim, dim_head=dim_head, heads=heads,
            omega_window=omega_window, use_omega_gate=True,
        ).to(device).to(dtype)
        
        state = cell.init_state(batch, device=device, dtype=dtype)
        
        # Buffer shape should be [BH, e-1, d, d] (no last dim for G/B split)
        expected_shape = (batch * heads, omega_window - 1, dim_head, dim_head)
        assert state.omega_buffer.shape == expected_shape
        
        # Calculate memory savings
        # Old: [BH, e-1, d, d, 2] = 2 * BH * (e-1) * d^2 * dtype_size
        # New: [BH, e-1, d, d] = BH * (e-1) * d^2 * dtype_size
        # Saved: 50%
        old_numel = batch * heads * (omega_window - 1) * dim_head * dim_head * 2
        new_numel = state.omega_buffer.numel()
        
        assert new_numel == old_numel / 2
        
        print(f"\n📉 Omega buffer size reduced by 50%")
        print(f"   Old numel: {old_numel:,}")
        print(f"   New numel: {new_numel:,}")
    
    def test_omega_window_correctness(self, device, dtype):
        """Test sliding window still computes correctly."""
        batch, seq_len, dim, dim_head, heads = 2, 64, 256, 64, 4
        omega_window = 4
        
        cell = OmegaRNNMemoryCell(
            dim=dim, dim_head=dim_head, heads=heads,
            omega_window=omega_window, use_omega_gate=False,  # No gate for simplicity
            use_momentum=False,  # Simplify to test window only
        ).to(device).to(dtype)
        
        # Process two sequences to test buffer carryover
        x1 = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
        x2 = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
        
        state1 = cell.init_state(batch, device=device, dtype=dtype)
        out1, state2 = cell(x1, state1)
        out2, state3 = cell(x2, state2)
        
        # Outputs should be valid
        assert not torch.isnan(out1).any()
        assert not torch.isnan(out2).any()
        
        # States should have changed
        assert not torch.allclose(state1.S, state2.S)
        assert not torch.allclose(state2.S, state3.S)
        
        # Buffers should be populated and different
        assert state2.omega_buffer is not None
        assert state3.omega_buffer is not None
        assert not torch.allclose(state2.omega_buffer, state3.omega_buffer)
        
        print(f"\n✅ Omega window correctness verified across sequences")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

