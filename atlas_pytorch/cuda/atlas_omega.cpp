// atlas_omega.cpp - PyTorch C++ Extension for Atlas Omega RNN
// Exact omega_window=16 implementation with checkpoint-based backward
// Requires A100 (shared memory: 163KB)

#include <torch/extension.h>

// Forward declarations for CUDA kernels
void cuda_forward_exact(
    int BH, int T,
    int n_ckpt,
    torch::Tensor phi_k,
    torch::Tensor phi_q,
    torch::Tensor delta,
    torch::Tensor lr,
    torch::Tensor decay,
    torch::Tensor beta,
    torch::Tensor gate,
    torch::Tensor S0,
    torch::Tensor Z0,
    torch::Tensor y,
    torch::Tensor S_T,
    torch::Tensor Z_T,
    torch::Tensor S_ckpt,
    torch::Tensor Z_ckpt
);

void cuda_backward_exact(
    int BH, int T,
    int n_ckpt,
    torch::Tensor phi_k,
    torch::Tensor phi_q,
    torch::Tensor delta,
    torch::Tensor lr,
    torch::Tensor decay,
    torch::Tensor beta,
    torch::Tensor gate,
    torch::Tensor S_T,
    torch::Tensor Z_T,
    torch::Tensor S_ckpt,
    torch::Tensor Z_ckpt,
    torch::Tensor dy,
    torch::Tensor dS_T,
    torch::Tensor dZ_T,
    torch::Tensor dphi_k,
    torch::Tensor dphi_q,
    torch::Tensor ddelta,
    torch::Tensor dlr,
    torch::Tensor ddecay,
    torch::Tensor dbeta,
    torch::Tensor dgate,
    torch::Tensor dS0,
    torch::Tensor dZ0
);

// Python-facing forward function (with checkpoints for numerically stable backward)
static void forward_exact_ckpt(
    torch::Tensor phi_k,
    torch::Tensor phi_q,
    torch::Tensor delta,
    torch::Tensor lr,
    torch::Tensor decay,
    torch::Tensor beta,
    torch::Tensor gate,
    torch::Tensor S0,
    torch::Tensor Z0,
    torch::Tensor y,
    torch::Tensor S_T,
    torch::Tensor Z_T,
    torch::Tensor S_ckpt,
    torch::Tensor Z_ckpt
) {
    TORCH_CHECK(phi_k.is_cuda(), "phi_k must be CUDA");
    TORCH_CHECK(phi_k.dtype() == torch::kBFloat16, "phi_k must be bf16");
    TORCH_CHECK(phi_q.dtype() == torch::kBFloat16, "phi_q must be bf16");
    TORCH_CHECK(delta.dtype() == torch::kBFloat16, "delta must be bf16");
    TORCH_CHECK(lr.dtype() == torch::kBFloat16, "lr must be bf16");
    TORCH_CHECK(decay.dtype() == torch::kBFloat16, "decay must be bf16");
    TORCH_CHECK(beta.dtype() == torch::kBFloat16, "beta must be bf16");
    TORCH_CHECK(gate.dtype() == torch::kBFloat16, "gate must be bf16");
    TORCH_CHECK(S0.dtype() == torch::kBFloat16 && Z0.dtype() == torch::kBFloat16, "S0/Z0 must be bf16");
    
    TORCH_CHECK(phi_k.is_contiguous(), "phi_k must be contiguous");
    TORCH_CHECK(phi_q.is_contiguous(), "phi_q must be contiguous");
    TORCH_CHECK(delta.is_contiguous(), "delta must be contiguous");
    TORCH_CHECK(lr.is_contiguous() && decay.is_contiguous() && beta.is_contiguous() && gate.is_contiguous(), "scalars must be contiguous");
    TORCH_CHECK(S0.is_contiguous() && Z0.is_contiguous(), "S0/Z0 must be contiguous");
    TORCH_CHECK(y.is_contiguous() && S_T.is_contiguous() && Z_T.is_contiguous(), "outputs must be contiguous");
    TORCH_CHECK(S_ckpt.is_cuda() && Z_ckpt.is_cuda(), "checkpoints must be CUDA");
    TORCH_CHECK(S_ckpt.dtype() == torch::kBFloat16 && Z_ckpt.dtype() == torch::kBFloat16, "checkpoints must be bf16");
    TORCH_CHECK(S_ckpt.is_contiguous() && Z_ckpt.is_contiguous(), "checkpoints must be contiguous");
    
    int64_t BH = phi_k.size(0);
    int64_t T = phi_k.size(1);
    int64_t n_ckpt = S_ckpt.size(1);
    
    cuda_forward_exact((int)BH, (int)T, (int)n_ckpt,
                       phi_k, phi_q, delta, lr, decay, beta, gate, S0, Z0,
                       y, S_T, Z_T, S_ckpt, Z_ckpt);
}

// Python-facing backward function (with checkpoints)
static void backward_exact_ckpt(
    torch::Tensor phi_k,
    torch::Tensor phi_q,
    torch::Tensor delta,
    torch::Tensor lr,
    torch::Tensor decay,
    torch::Tensor beta,
    torch::Tensor gate,
    torch::Tensor S_T,
    torch::Tensor Z_T,
    torch::Tensor S_ckpt,
    torch::Tensor Z_ckpt,
    torch::Tensor dy,
    torch::Tensor dS_T,
    torch::Tensor dZ_T,
    torch::Tensor dphi_k,
    torch::Tensor dphi_q,
    torch::Tensor ddelta,
    torch::Tensor dlr,
    torch::Tensor ddecay,
    torch::Tensor dbeta,
    torch::Tensor dgate,
    torch::Tensor dS0,
    torch::Tensor dZ0
) {
    int64_t BH = phi_k.size(0);
    int64_t T = phi_k.size(1);
    int64_t n_ckpt = S_ckpt.size(1);
    
    cuda_backward_exact((int)BH, (int)T, (int)n_ckpt,
                        phi_k, phi_q, delta, lr, decay, beta, gate,
                        S_T, Z_T, S_ckpt, Z_ckpt,
                        dy, dS_T, dZ_T,
                        dphi_k, dphi_q, ddelta, dlr, ddecay, dbeta, dgate, dS0, dZ0);
}

// Register with PyTorch using pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_exact", &forward_exact_ckpt, "Atlas Omega forward (CUDA, with ckpt)");
    m.def("backward_exact", &backward_exact_ckpt, "Atlas Omega backward (CUDA, with ckpt)");
}

