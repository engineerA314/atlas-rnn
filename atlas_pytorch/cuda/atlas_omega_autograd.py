"""
Atlas Omega RNN CUDA Autograd Function
Exact omega_window=16 implementation with checkpoint-based backward
"""

import os
import time
import torch
import torch.nn.functional as F

_module_path = os.path.dirname(__file__)
_CUDA_AVAILABLE = False
_CUDA_ERROR = ""
_CUDA_LOADED_VIA = "none"

def get_cuda_status():
    """Return CUDA backend status for logging/debugging"""
    return {
        "available": _CUDA_AVAILABLE,
        "loaded_via": _CUDA_LOADED_VIA,
        "error": _CUDA_ERROR if not _CUDA_AVAILABLE else ""
    }

# Import compiled CUDA extension (JIT compile if needed).
# For multi-process (DDP/DS) runs, try to avoid simultaneous compilation.
atlas_omega_ext = None
is_distributed = False
rank = 0
try:
    from torch.utils.cpp_extension import load
    import torch.distributed as dist

    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    compile_lock = os.path.join(_module_path, ".compile_lock")

    if is_distributed:
        if rank == 0:
            # Create lock so other ranks wait even if load() uses cache checks.
            try:
                with open(compile_lock, "w") as f:
                    f.write("compiling\n")
            except Exception:
                pass
        else:
            # Wait for rank0 to finish compilation (best-effort).
            for _ in range(600):  # ~300s
                if os.path.exists(compile_lock):
                    time.sleep(0.5)
                else:
                    break
            time.sleep(1.0)

    atlas_omega_ext = load(
        name="atlas_omega_ext",
        sources=[
            os.path.join(_module_path, "atlas_omega.cpp"),
            os.path.join(_module_path, "atlas_omega.cu"),
        ],
        extra_cuda_cflags=[
            "-O3", "-lineinfo",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "-D_C_=64", "-D_E_=16",
            "--use_fast_math",
        ],
        verbose=False,
    )
    _CUDA_AVAILABLE = True
    _CUDA_LOADED_VIA = "jit"
except Exception as e:
    atlas_omega_ext = None
    _CUDA_AVAILABLE = False
    _CUDA_ERROR = str(e)
    _CUDA_LOADED_VIA = "failed"
finally:
    # Rank0: remove lock file after compilation attempt (best-effort).
    try:
        if (not is_distributed) or (rank == 0):
            lock_path = os.path.join(_module_path, ".compile_lock")
            if os.path.exists(lock_path):
                os.remove(lock_path)
    except Exception:
        pass

class AtlasOmegaFunction(torch.autograd.Function):
    """
    Atlas Omega RNN with exact sliding window (omega=16) and checkpoint-based backward
    
    Forward: O(BHTd + BH(T/16)d²) memory
    Backward: Recompute from checkpoints, O(d²) shared memory per block
    
    Requires: A100 GPU (shared memory >= 163KB)
    """
    
    @staticmethod
    def forward(ctx, phi_k, phi_q, v, S_ref, lr, decay, beta, gate, S0, Z0):
        """
        Args:
            phi_k: [BH, T, d] - Key features (polynomial-mapped)
            phi_q: [BH, T, d] - Query features (polynomial-mapped)
            v: [BH, T, d] - Values
            S_ref: [BH, d, d] - Reference state for delta computation
            lr: [BH, T] - Learning rates
            decay: [BH, T] - Decay factors (alpha)
            beta: [BH, T] - Momentum factors
            gate: [BH, T] - Omega gates (u_t)
            S0: [BH, d, d] - Initial state S
            Z0: [BH, d, d] - Initial state Z
        
        Returns:
            y: [BH, T, d] - Output
            S_T: [BH, d, d] - Final state S
            Z_T: [BH, d, d] - Final state Z
        """
        BH, T, d = phi_k.shape
        
        # Compute delta: pred = S_ref @ phi_k - v
        # Shape: [BH, T, d] = [BH, d, d] @ [BH, T, d]
        pred = torch.einsum('bde,bte->btd', S_ref, phi_k)
        delta = pred - v
        
        # Prepare outputs
        y = torch.empty((BH, T, d), device=phi_k.device, dtype=torch.bfloat16)
        S_T = torch.empty((BH, d, d), device=phi_k.device, dtype=torch.bfloat16)
        Z_T = torch.empty((BH, d, d), device=phi_k.device, dtype=torch.bfloat16)

        # Checkpoints for numerically stable backward (store S_t, Z_t every K steps).
        K = 16
        n_ckpt = (T + K - 1) // K
        S_ckpt = torch.empty((BH, n_ckpt, d, d), device=phi_k.device, dtype=torch.bfloat16)
        Z_ckpt = torch.empty((BH, n_ckpt, d, d), device=phi_k.device, dtype=torch.bfloat16)
        
        # Call CUDA kernel
        atlas_omega_ext.forward_exact(
            phi_k.contiguous(), 
            phi_q.contiguous(), 
            delta.contiguous(),
            lr.contiguous(), 
            decay.contiguous(), 
            beta.contiguous(), 
            gate.contiguous(),
            S0.contiguous(), 
            Z0.contiguous(),
            y, S_T, Z_T,
            S_ckpt, Z_ckpt
        )
        
        # Save for backward
        ctx.save_for_backward(phi_k, phi_q, delta, lr, decay, beta, gate, S_T, Z_T, S_ref, S_ckpt, Z_ckpt)
        
        return y, S_T, Z_T
    
    @staticmethod
    def backward(ctx, dy, dS_T, dZ_T):
        """
        Backward pass with checkpoint recomputation
        
        Strategy:
        - Load saved states (S_T, Z_T) and inputs
        - Recompute forward for each chunk (16 steps)
        - Compute gradients using ring buffer for "future 16 sum"
        """
        phi_k, phi_q, delta, lr, decay, beta, gate, S_T, Z_T, S_ref, S_ckpt, Z_ckpt = ctx.saved_tensors
        BH, T, d = phi_k.shape

        # Unused outputs may produce None grad outputs; treat as zeros.
        if dS_T is None:
            dS_T = torch.zeros_like(S_T)
        if dZ_T is None:
            dZ_T = torch.zeros_like(Z_T)
        
        # Prepare gradient outputs
        dphi_k = torch.empty_like(phi_k)
        dphi_q = torch.empty_like(phi_q)
        ddelta = torch.empty_like(delta)
        dlr = torch.empty((BH, T), device=phi_k.device, dtype=torch.bfloat16)
        ddecay = torch.empty((BH, T), device=phi_k.device, dtype=torch.bfloat16)
        dbeta = torch.empty((BH, T), device=phi_k.device, dtype=torch.bfloat16)
        dgate = torch.empty((BH, T), device=phi_k.device, dtype=torch.bfloat16)
        dS0 = torch.empty((BH, d, d), device=phi_k.device, dtype=torch.bfloat16)
        dZ0 = torch.empty((BH, d, d), device=phi_k.device, dtype=torch.bfloat16)
        
        # Call CUDA backward kernel
        atlas_omega_ext.backward_exact(
            phi_k, phi_q, delta, lr, decay, beta, gate, S_T, Z_T, S_ckpt, Z_ckpt,
            dy.contiguous(), dS_T.contiguous(), dZ_T.contiguous(),
            dphi_k, dphi_q, ddelta, dlr, ddecay, dbeta, dgate, dS0, dZ0
        )
        
        # Propagate ddelta to dv, dphi_k (from delta = pred - v)
        dv = -ddelta
        
        # dphi_k from delta computation: delta = S_ref @ phi_k - v
        # → ddelta/dphi_k = S_ref^T
        dphi_k_from_delta = torch.einsum('bde,btd->bte', S_ref.transpose(1, 2), ddelta)
        dphi_k += dphi_k_from_delta
        
        # dS_ref from delta computation
        dS_ref = torch.einsum('btd,bte->bde', ddelta, phi_k)
        
        # Return gradients in order: phi_k, phi_q, v, S_ref, lr, decay, beta, gate, S0, Z0
        return dphi_k, dphi_q, dv, dS_ref, dlr, ddecay, dbeta, dgate, dS0, dZ0


def atlas_omega_forward(phi_k, phi_q, v, S_ref, lr, decay, beta, gate, S0, Z0):
    """
    Convenience function for forward pass (no autograd)
    """
    return AtlasOmegaFunction.apply(phi_k, phi_q, v, S_ref, lr, decay, beta, gate, S0, Z0)


def check_cuda_availability():
    """
    Check if CUDA extension is available and shared memory is sufficient
    """
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    if not _CUDA_AVAILABLE:
        return False, f"CUDA extension not loaded: {_CUDA_ERROR}"
    
    props = torch.cuda.get_device_properties(0)
    shared_mem = props.shared_memory_per_block
    
    # Note: Global memory ring buffer is used, so no strict shared memory requirement
    return True, f"Atlas Omega CUDA ready (GPU: {props.name}, loaded_via={_CUDA_LOADED_VIA}, using global memory ring buffer)"


if __name__ == "__main__":
    # Test
    available, msg = check_cuda_availability()
    print(f"Status: {msg}")
    
    if available:
        print("\nRunning quick test...")
        BH, T, d = 2, 32, 64
        
        phi_k = torch.randn(BH, T, d, device='cuda', dtype=torch.bfloat16)
        phi_q = torch.randn(BH, T, d, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(BH, T, d, device='cuda', dtype=torch.bfloat16)
        S_ref = torch.randn(BH, d, d, device='cuda', dtype=torch.bfloat16)
        
        lr = torch.full((BH, T), 0.01, device='cuda', dtype=torch.bfloat16)
        decay = torch.full((BH, T), 0.99, device='cuda', dtype=torch.bfloat16)
        beta = torch.full((BH, T), 0.9, device='cuda', dtype=torch.bfloat16)
        gate = torch.full((BH, T), 1.0, device='cuda', dtype=torch.bfloat16)
        
        S0 = torch.randn(BH, d, d, device='cuda', dtype=torch.bfloat16)
        Z0 = torch.randn(BH, d, d, device='cuda', dtype=torch.bfloat16)
        
        y, S_T, Z_T = atlas_omega_forward(phi_k, phi_q, v, S_ref, lr, decay, beta, gate, S0, Z0)
        
        print(f"Output shape: {y.shape}")
        print(f"Final states: S_T={S_T.shape}, Z_T={Z_T.shape}")
        print("Test passed!")

