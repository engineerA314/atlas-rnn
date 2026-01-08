"""
Test CUDA Backend Integration in OmegaRNNMemoryCell

Verifies that use_cuda flag works correctly and produces same results as PyTorch.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from atlas_pytorch.rnn_memory import OmegaRNNMemoryCell, RNNMemState


@pytest.fixture
def cuda_available():
    """Check if CUDA extension is available"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    try:
        from atlas_pytorch.cuda.atlas_omega_autograd import check_cuda_availability
        available, msg = check_cuda_availability()
        if not available:
            pytest.skip(f"CUDA extension not available: {msg}")
        return True
    except Exception as e:
        pytest.skip(f"Failed to check CUDA availability: {e}")


class TestCUDAIntegration:
    """Test CUDA backend integration"""
    
    def setup_method(self):
        """Setup test data"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        try:
            from atlas_pytorch.cuda.atlas_omega_autograd import check_cuda_availability
            available, msg = check_cuda_availability()
            if not available:
                pytest.skip(f"CUDA extension not available: {msg}")
        except Exception as e:
            pytest.skip(f"Failed to check CUDA availability: {e}")

        torch.manual_seed(42)
        self.device = 'cuda'
        # CUDA kernel runs in bf16; keep test inputs/states consistent with real usage.
        self.dtype = torch.bfloat16
        
        # Small test case
        self.batch = 2
        self.seq_len = 128
        self.dim = 256
        self.dim_head = 64
        self.heads = 4
        
        # Test input
        self.x = torch.randn(self.batch, self.seq_len, self.dim, device=self.device, dtype=self.dtype)
        
        # Initial state
        BH = self.batch * self.heads
        self.state = RNNMemState(
            seq_index=0,
            S=torch.zeros(BH, self.dim_head, self.dim_head, device=self.device, dtype=self.dtype),
            Z=torch.zeros(BH, self.dim_head, self.dim_head, device=self.device, dtype=self.dtype),
            omega_buffer=None
        )
    
    def test_cuda_flag_fallback(self, cuda_available):
        """Test that invalid configurations fall back to PyTorch"""
        # omega_window != 16 should fall back
        cell = OmegaRNNMemoryCell(
            dim=self.dim,
            dim_head=self.dim_head,
            heads=self.heads,
            omega_window=4,  # Not 16!
            use_cuda=True
        ).to(self.device, dtype=self.dtype)
        
        # Should have fallen back to PyTorch
        assert not cell.cuda_available
        
        # Should still work
        out, new_state = cell(self.x, self.state)
        assert out.shape == (self.batch, self.seq_len, self.dim)
    
    def test_pytorch_vs_cuda_forward(self, cuda_available):
        """Compare PyTorch and CUDA forward outputs"""
        # PyTorch version
        cell_pytorch = OmegaRNNMemoryCell(
            dim=self.dim,
            dim_head=self.dim_head,
            heads=self.heads,
            omega_window=16,
            use_omega_gate=True,
            use_momentum=True,
            poly_degree=2,
            poly_mode='elementwise',
            use_cuda=False
        ).to(self.device, dtype=self.dtype)
        
        # CUDA version (same weights)
        cell_cuda = OmegaRNNMemoryCell(
            dim=self.dim,
            dim_head=self.dim_head,
            heads=self.heads,
            omega_window=16,
            use_omega_gate=True,
            use_momentum=True,
            poly_degree=2,
            poly_mode='elementwise',
            use_cuda=True
        ).to(self.device, dtype=self.dtype)
        
        # Copy weights
        cell_cuda.load_state_dict(cell_pytorch.state_dict())
        
        # Forward pass
        with torch.no_grad():
            out_pytorch, state_pytorch = cell_pytorch(self.x, self.state)
            out_cuda, state_cuda = cell_cuda(self.x, self.state)
        
        # Compare outputs.
        # NOTE: PyTorch path uses assoc_scan which changes accumulation order,
        # and CUDA kernel uses explicit float accumulations. In bf16 this can
        # yield small-but-nontrivial numeric differences, especially after
        # depthwise conv + polynomial features.
        max_out = out_pytorch.abs().max()
        max_out_diff = (out_pytorch - out_cuda).abs().max()
        assert max_out_diff <= (0.05 * max_out + 0.5), \
            f"Output mismatch too large: max diff = {max_out_diff} (max abs = {max_out})"

        max_S = state_pytorch.S.abs().max()
        max_S_diff = (state_pytorch.S - state_cuda.S).abs().max()
        assert max_S_diff <= (0.05 * max_S + 1.0), \
            f"S state mismatch too large: max diff = {max_S_diff} (max abs = {max_S})"
        
        if state_pytorch.Z is not None and state_cuda.Z is not None:
            max_Z = state_pytorch.Z.abs().max()
            max_Z_diff = (state_pytorch.Z - state_cuda.Z).abs().max()
            assert max_Z_diff <= (0.05 * max_Z + 1.0), \
                f"Z state mismatch too large: max diff = {max_Z_diff} (max abs = {max_Z})"
        
        print(f"✓ PyTorch vs CUDA forward: max output diff = {(out_pytorch - out_cuda).abs().max():.6f}")
    
    def test_cuda_backward(self, cuda_available):
        """Test that CUDA backward works"""
        cell_cuda = OmegaRNNMemoryCell(
            dim=self.dim,
            dim_head=self.dim_head,
            heads=self.heads,
            omega_window=16,
            use_omega_gate=True,
            use_momentum=True,
            poly_degree=2,
            poly_mode='elementwise',
            use_cuda=True
        ).to(self.device, dtype=self.dtype)
        
        # Forward with gradients
        x = self.x.clone().requires_grad_(True)
        out, state = cell_cuda(x, self.state)
        
        # Backward
        loss = out.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        
        # Check cell parameters have gradients
        for name, param in cell_cuda.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"
        
        print("✓ CUDA backward produces valid gradients")
    
    def test_cuda_performance(self, cuda_available):
        """Benchmark CUDA vs PyTorch performance"""
        import time
        
        # Larger test for meaningful timing
        batch, seq_len, dim = 8, 512, 896
        dim_head, heads = 64, 14
        
        x = torch.randn(batch, seq_len, dim, device=self.device, dtype=self.dtype)
        BH = batch * heads
        state = RNNMemState(
            seq_index=0,
            S=torch.zeros(BH, dim_head, dim_head, device=self.device, dtype=self.dtype),
            Z=torch.zeros(BH, dim_head, dim_head, device=self.device, dtype=self.dtype),
            omega_buffer=None
        )
        
        # PyTorch version
        cell_pytorch = OmegaRNNMemoryCell(
            dim=dim, dim_head=dim_head, heads=heads,
            omega_window=16, use_cuda=False
        ).to(self.device, dtype=self.dtype)
        
        # CUDA version
        cell_cuda = OmegaRNNMemoryCell(
            dim=dim, dim_head=dim_head, heads=heads,
            omega_window=16, use_cuda=True
        ).to(self.device, dtype=self.dtype)
        
        cell_cuda.load_state_dict(cell_pytorch.state_dict())
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = cell_pytorch(x, state)
                _ = cell_cuda(x, state)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        n_iters = 20
        for _ in range(n_iters):
            with torch.no_grad():
                _ = cell_pytorch(x, state)
        torch.cuda.synchronize()
        time_pytorch = (time.time() - start) / n_iters * 1000  # ms
        
        # Benchmark CUDA
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iters):
            with torch.no_grad():
                _ = cell_cuda(x, state)
        torch.cuda.synchronize()
        time_cuda = (time.time() - start) / n_iters * 1000  # ms
        
        speedup = time_pytorch / time_cuda
        
        print(f"\n📊 Performance (batch={batch}, seq={seq_len}, dim={dim}):")
        print(f"  PyTorch: {time_pytorch:.2f} ms")
        print(f"  CUDA:    {time_cuda:.2f} ms")
        print(f"  Speedup: {speedup:.1f}x")
        
        # CUDA should be faster
        assert speedup > 1.0, f"CUDA not faster than PyTorch! Speedup: {speedup:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

