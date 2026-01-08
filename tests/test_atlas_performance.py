"""
Test Atlas Omega CUDA Performance vs PyTorch

This test measures actual memory usage and speed improvements
for realistic 0.5B model scenarios.

Key Metrics:
1. Memory usage: PyTorch refactored vs CUDA (512/16384 ctx len)
2. Speed: Forward/Backward/Full step timing
3. OOM prevention: Can 16k context fit in 80GB?

Run on A100: pytest test_atlas_performance.py -v -s
"""

import pytest
import torch
import torch.nn.functional as F
import sys
import os
import time
import gc

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from atlas_pytorch.rnn_memory import OmegaRNNMemoryCell


def get_memory_stats():
    """Get current GPU memory usage"""
    if not torch.cuda.is_available():
        return {}
    
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'max_allocated_gb': max_allocated
    }


def reset_memory():
    """Reset GPU memory tracking"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


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


class TestAtlasOmegaPerformance:
    """Performance tests for realistic 0.5B model scenarios"""
    
    def _create_test_data(self, batch_size, seq_len, d_model, n_heads):
        """Create test data matching 0.5B model dimensions"""
        device = 'cuda'
        dtype = torch.bfloat16
        
        # Per-head dimension
        d_head = d_model // n_heads
        BH = batch_size * n_heads
        
        # Input tensors
        phi_k = torch.randn(BH, seq_len, d_head, device=device, dtype=dtype)
        phi_q = torch.randn(BH, seq_len, d_head, device=device, dtype=dtype)
        v = torch.randn(BH, seq_len, d_head, device=device, dtype=dtype)
        S_ref = torch.randn(BH, d_head, d_head, device=device, dtype=dtype)
        
        lr = torch.full((BH, seq_len), 0.01, device=device, dtype=dtype)
        decay = torch.full((BH, seq_len), 0.99, device=device, dtype=dtype)
        beta = torch.full((BH, seq_len), 0.9, device=device, dtype=dtype)
        gate = torch.full((BH, seq_len), 1.0, device=device, dtype=dtype)
        
        S0 = torch.randn(BH, d_head, d_head, device=device, dtype=dtype)
        Z0 = torch.randn(BH, d_head, d_head, device=device, dtype=dtype)
        
        return phi_k, phi_q, v, S_ref, lr, decay, beta, gate, S0, Z0
    
    def test_memory_512_context(self, cuda_available):
        """Memory usage test: 512 context length (Stage 1 scenario)"""
        from atlas_pytorch.cuda.atlas_omega_autograd import AtlasOmegaFunction
        
        # 0.5B model config (approximate)
        batch_size = 8
        seq_len = 512
        d_model = 896  # Qwen2-0.5B hidden size
        n_heads = 14   # Qwen2-0.5B num heads
        
        print(f"\n{'='*60}")
        print(f"Memory Test: 512 Context Length")
        print(f"Config: B={batch_size}, T={seq_len}, d_model={d_model}, n_heads={n_heads}")
        print(f"{'='*60}\n")
        
        # Prepare data
        data = self._create_test_data(batch_size, seq_len, d_model, n_heads)
        phi_k, phi_q, v, S_ref, lr, decay, beta, gate, S0, Z0 = data
        
        # Test CUDA implementation
        reset_memory()
        mem_before = get_memory_stats()
        
        # Enable gradients for full memory test
        phi_k_cuda = phi_k.clone().requires_grad_(True)
        phi_q_cuda = phi_q.clone().requires_grad_(True)
        v_cuda = v.clone().requires_grad_(True)
        S_ref_cuda = S_ref.clone().requires_grad_(True)
        
        y_cuda, S_T_cuda, Z_T_cuda = AtlasOmegaFunction.apply(
            phi_k_cuda, phi_q_cuda, v_cuda, S_ref_cuda,
            lr, decay, beta, gate, S0, Z0
        )
        
        # Backward to measure full memory
        loss = (y_cuda ** 2).sum()
        loss.backward()
        
        mem_after = get_memory_stats()
        mem_cuda = mem_after['max_allocated_gb'] - mem_before['allocated_gb']
        
        print(f"CUDA Implementation:")
        print(f"  Peak memory: {mem_cuda:.3f} GB")
        print(f"  Allocated: {mem_after['allocated_gb']:.3f} GB")
        print(f"  Reserved: {mem_after['reserved_gb']:.3f} GB")
        
        # TODO: Compare with PyTorch refactored version
        # For now, just verify it fits in memory
        assert mem_cuda < 10.0, f"Memory usage too high: {mem_cuda:.3f} GB (expected < 10 GB)"
        
        print(f"\n✓ CUDA implementation fits in memory for 512 context")
    
    def test_memory_16k_context(self, cuda_available):
        """Memory usage test: 16384 context length (Stage 3 scenario) - OOM critical test"""
        from atlas_pytorch.cuda.atlas_omega_autograd import AtlasOmegaFunction
        
        # 0.5B model config
        batch_size = 4  # Reduced for 16k context
        seq_len = 16384
        d_model = 896
        n_heads = 14
        
        print(f"\n{'='*60}")
        print(f"Memory Test: 16K Context Length (OOM Critical)")
        print(f"Config: B={batch_size}, T={seq_len}, d_model={d_model}, n_heads={n_heads}")
        print(f"{'='*60}\n")
        
        try:
            # Prepare data
            data = self._create_test_data(batch_size, seq_len, d_model, n_heads)
            phi_k, phi_q, v, S_ref, lr, decay, beta, gate, S0, Z0 = data
            
            # Test CUDA implementation
            reset_memory()
            mem_before = get_memory_stats()
            
            phi_k_cuda = phi_k.clone().requires_grad_(True)
            phi_q_cuda = phi_q.clone().requires_grad_(True)
            v_cuda = v.clone().requires_grad_(True)
            S_ref_cuda = S_ref.clone().requires_grad_(True)
            
            y_cuda, S_T_cuda, Z_T_cuda = AtlasOmegaFunction.apply(
                phi_k_cuda, phi_q_cuda, v_cuda, S_ref_cuda,
                lr, decay, beta, gate, S0, Z0
            )
            
            # Backward to measure full memory
            loss = (y_cuda ** 2).sum()
            loss.backward()
            
            mem_after = get_memory_stats()
            mem_cuda = mem_after['max_allocated_gb'] - mem_before['allocated_gb']
            
            print(f"CUDA Implementation:")
            print(f"  Peak memory: {mem_cuda:.3f} GB")
            print(f"  Allocated: {mem_after['allocated_gb']:.3f} GB")
            print(f"  Reserved: {mem_after['reserved_gb']:.3f} GB")
            
            # For A100 80GB, should fit comfortably
            assert mem_cuda < 40.0, f"Memory usage too high: {mem_cuda:.3f} GB (expected < 40 GB)"
            
            print(f"\n✓ CUDA implementation handles 16K context without OOM!")
            print(f"  (This was the main problem - now solved)")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.fail(f"OOM at 16K context! Memory: {get_memory_stats()}")
            raise
    
    def test_speed_512_context(self, cuda_available):
        """Speed test: 512 context length"""
        from atlas_pytorch.cuda.atlas_omega_autograd import AtlasOmegaFunction
        
        batch_size = 8
        seq_len = 512
        d_model = 896
        n_heads = 14
        
        print(f"\n{'='*60}")
        print(f"Speed Test: 512 Context Length")
        print(f"Config: B={batch_size}, T={seq_len}, d_model={d_model}, n_heads={n_heads}")
        print(f"{'='*60}\n")
        
        data = self._create_test_data(batch_size, seq_len, d_model, n_heads)
        phi_k, phi_q, v, S_ref, lr, decay, beta, gate, S0, Z0 = data
        
        # Warmup
        for _ in range(5):
            phi_k_test = phi_k.clone().requires_grad_(True)
            phi_q_test = phi_q.clone().requires_grad_(True)
            v_test = v.clone().requires_grad_(True)
            S_ref_test = S_ref.clone().requires_grad_(True)
            
            y, S_T, Z_T = AtlasOmegaFunction.apply(
                phi_k_test, phi_q_test, v_test, S_ref_test,
                lr, decay, beta, gate, S0, Z0
            )
            loss = (y ** 2).sum()
            loss.backward()
        
        # Benchmark forward
        torch.cuda.synchronize()
        n_iters = 100
        start = time.time()
        
        for _ in range(n_iters):
            y, S_T, Z_T = AtlasOmegaFunction.apply(
                phi_k, phi_q, v, S_ref, lr, decay, beta, gate, S0, Z0
            )
        
        torch.cuda.synchronize()
        forward_time = (time.time() - start) / n_iters * 1000  # ms
        
        # Benchmark forward + backward
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(n_iters):
            phi_k_test = phi_k.clone().requires_grad_(True)
            phi_q_test = phi_q.clone().requires_grad_(True)
            v_test = v.clone().requires_grad_(True)
            S_ref_test = S_ref.clone().requires_grad_(True)
            
            y, S_T, Z_T = AtlasOmegaFunction.apply(
                phi_k_test, phi_q_test, v_test, S_ref_test,
                lr, decay, beta, gate, S0, Z0
            )
            loss = (y ** 2).sum()
            loss.backward()
        
        torch.cuda.synchronize()
        full_time = (time.time() - start) / n_iters * 1000  # ms
        backward_time = full_time - forward_time
        
        print(f"CUDA Implementation:")
        print(f"  Forward:  {forward_time:.2f} ms")
        print(f"  Backward: {backward_time:.2f} ms")
        print(f"  Total:    {full_time:.2f} ms")
        print(f"  Throughput: {batch_size * seq_len / full_time * 1000:.0f} tokens/sec")
        
        # TODO: Compare with PyTorch refactored version
        # Expected: 3-10x speedup
        
        print(f"\n✓ Speed measurement complete")
    
    def test_speed_16k_context(self, cuda_available):
        """Speed test: 16384 context length"""
        from atlas_pytorch.cuda.atlas_omega_autograd import AtlasOmegaFunction
        
        batch_size = 4
        seq_len = 16384
        d_model = 896
        n_heads = 14
        
        print(f"\n{'='*60}")
        print(f"Speed Test: 16K Context Length")
        print(f"Config: B={batch_size}, T={seq_len}, d_model={d_model}, n_heads={n_heads}")
        print(f"{'='*60}\n")
        
        data = self._create_test_data(batch_size, seq_len, d_model, n_heads)
        phi_k, phi_q, v, S_ref, lr, decay, beta, gate, S0, Z0 = data
        
        # Warmup
        for _ in range(3):
            phi_k_test = phi_k.clone().requires_grad_(True)
            phi_q_test = phi_q.clone().requires_grad_(True)
            v_test = v.clone().requires_grad_(True)
            S_ref_test = S_ref.clone().requires_grad_(True)
            
            y, S_T, Z_T = AtlasOmegaFunction.apply(
                phi_k_test, phi_q_test, v_test, S_ref_test,
                lr, decay, beta, gate, S0, Z0
            )
            loss = (y ** 2).sum()
            loss.backward()
        
        # Benchmark (fewer iterations for 16k)
        torch.cuda.synchronize()
        n_iters = 20
        start = time.time()
        
        for _ in range(n_iters):
            phi_k_test = phi_k.clone().requires_grad_(True)
            phi_q_test = phi_q.clone().requires_grad_(True)
            v_test = v.clone().requires_grad_(True)
            S_ref_test = S_ref.clone().requires_grad_(True)
            
            y, S_T, Z_T = AtlasOmegaFunction.apply(
                phi_k_test, phi_q_test, v_test, S_ref_test,
                lr, decay, beta, gate, S0, Z0
            )
            loss = (y ** 2).sum()
            loss.backward()
        
        torch.cuda.synchronize()
        full_time = (time.time() - start) / n_iters * 1000  # ms
        
        print(f"CUDA Implementation:")
        print(f"  Full step: {full_time:.2f} ms")
        print(f"  Throughput: {batch_size * seq_len / full_time * 1000:.0f} tokens/sec")
        
        # Estimate training time for 0.5B model
        # Assume ~24 layers, 100M tokens
        n_layers = 24
        n_tokens = 100_000_000
        n_steps = n_tokens // (batch_size * seq_len)
        
        time_per_step = full_time / 1000 * n_layers  # seconds
        total_time = time_per_step * n_steps / 3600  # hours
        
        print(f"\n📊 Training Estimate (0.5B model, 100M tokens, 16k context):")
        print(f"  Layers: {n_layers}")
        print(f"  Steps: {n_steps}")
        print(f"  Time per step: {time_per_step:.2f} sec")
        print(f"  Total time: {total_time:.1f} hours")
        print(f"  (Goal: ~10 hours, Current: {total_time:.1f} hours)")
        
        print(f"\n✓ Speed measurement complete for 16K context")


class TestPyTorchVsCUDAComparison:
    """Direct comparison: PyTorch refactored vs CUDA"""
    
    def test_compare_memory_and_speed(self, cuda_available):
        """Compare PyTorch refactored vs CUDA (512 context)"""
        # TODO: Implement side-by-side comparison
        # This requires running the same workload with:
        # 1. OmegaRNNMemoryCell (PyTorch refactored)
        # 2. AtlasOmegaFunction (CUDA)
        # And measuring both memory and speed
        
        pytest.skip("Side-by-side comparison not yet implemented")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

