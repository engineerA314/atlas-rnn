"""
Compile Atlas Omega CUDA extension
Run this script on A100 instance to build the CUDA kernels
"""

import torch
from torch.utils.cpp_extension import load
import os

def compile_atlas_omega(verbose=True):
    """
    Compile CUDA extension with JIT
    
    Requirements:
    - CUDA 11.8+
    - PyTorch with CUDA support
    - ninja build system
    - A100 GPU (shared memory >= 150KB)
    """
    
    # Get current directory
    cuda_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Source files
    sources = [
        os.path.join(cuda_dir, "atlas_omega.cpp"),
        os.path.join(cuda_dir, "atlas_omega.cu"),
    ]
    
    # Check if files exist
    for src in sources:
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source file not found: {src}\nPlease ensure atlas_omega.cu is created.")
    
    # Compilation flags
    extra_cuda_cflags = [
        "-O3",
        "-lineinfo",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "-D_C_=64",  # d dimension
        "-D_E_=16",  # omega_window
        "--use_fast_math",
    ]
    
    print("=" * 60)
    print("Compiling Atlas Omega CUDA Extension")
    print("=" * 60)
    print(f"\nSources:")
    for src in sources:
        print(f"  - {src}")
    
    print(f"\nCUDA flags:")
    for flag in extra_cuda_cflags:
        print(f"  {flag}")
    
    print(f"\nCompiling...")
    print("(This may take 2-5 minutes on first run)\n")
    
    # Compile
    atlas_omega_ext = load(
        name="atlas_omega_ext",
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        verbose=verbose,
    )
    
    print("\n" + "=" * 60)
    print("Compilation successful!")
    print("=" * 60)
    print("\nVerifying extension...")
    
    # Verify ops are available (using PYBIND11_MODULE)
    try:
        assert hasattr(atlas_omega_ext, 'forward_exact'), "forward_exact not found"
        assert hasattr(atlas_omega_ext, 'backward_exact'), "backward_exact not found"
        print("✓ Forward op available")
        print("✓ Backward op available")
    except (AttributeError, AssertionError) as e:
        print(f"✗ Error: {e}")
        return None
    
    # Check GPU info
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        shared_mem = props.shared_memory_per_block
        print(f"\nGPU: {props.name}")
        print(f"Shared memory per block: {shared_mem} bytes ({shared_mem/1024:.1f} KB)")
        print(f"✓ Using global memory ring buffer (no strict shared memory requirement)")
    
    print("\n" + "=" * 60)
    print("Ready to use!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run tests: cd ../../tests && pytest test_atlas_cuda.py -v")
    print("2. Or test directly: python atlas_omega_autograd.py")
    
    return atlas_omega_ext


if __name__ == "__main__":
    compile_atlas_omega(verbose=True)

