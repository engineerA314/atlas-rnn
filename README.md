<img src="./fig1.png" width="400px"></img>

<img src="./fig2.png" width="400px"></img>

## Atlas-RNN - Pytorch

**Pure RNN-form implementation of Titans and Atlas (OmegaNet) memory architectures.**

This project derives the explicit RNN update rules from the implicit gradient descent formulation in Titans and Atlas papers, enabling efficient training without per-sample gradient computation via `torch.func`.

## Key Insight: From Implicit GD to Explicit RNN

The Titans and Atlas papers show that linear attention can be viewed as implicit gradient descent on a "surprise" loss. We reverse this insight: by explicitly computing the gradient of the MSE loss, we derive closed-form RNN update rules that are mathematically equivalent but computationally more efficient.

### Mathematical Derivation

#### 1. Internal Objective (Attentional Bias / Surprise)

For a memory state \( S \) and input key-value pair \( (k_t, v_t) \):

```
ℓ_t(S) = ½ ‖S φ_t − v_t‖²
```

where \( φ_t = φ(k_t) \) is the (optionally polynomial) feature map of the key.

#### 2. Gradient Computation

Taking the gradient with respect to \( S \):

```
∇_S ℓ_t(S) = (S φ_t − v_t) φ_t^T = S φ_t φ_t^T − v_t φ_t^T
```

Define:

- **Prediction error (surprise)**: `δ_t = S_{t-1} φ_t − v_t`
- **Gram term**: `G_t = φ_t φ_t^T`
- **Cross term**: `B_t = v_t φ_t^T`

#### 3. Titans-RNN (Online, e=1)

**Without momentum:**

```
S_t = α_t S_{t-1} − η_t δ_t φ_t^T
    = S_{t-1} (α_t I − η_t φ_t φ_t^T) + η_t v_t φ_t^T
```

**With momentum:**

```
Z_t = β_t Z_{t-1} + δ_t φ_t^T
S_t = α_t S_{t-1} − η_t Z_t
```

**Retrieval:**

```
y_t = S_{t-1} ψ_t    (where ψ_t = φ(q_t))
```

#### 4. OmegaNet-RNN (Sliding Window, e ≥ 1)

For context window \( W_t = \{t-e+1, ..., t\} \) with gates \( U_t^p \):

```
G_t = Σ_{p∈W_t} U_t^p φ_p φ_p^T    (Gram matrix)
B_t = Σ_{p∈W_t} U_t^p v_p φ_p^T    (Cross term)
Δ_t = S_{t-1} G_t − B_t            (Context surprise)
```

**Without momentum:**

```
S_t = S_{t-1} (α_t I − η_t G_t) + η_t B_t
```

**With momentum:**

```
Z_t = β_t Z_{t-1} + Δ_t
S_t = α_t S_{t-1} − η_t Z_t
```

#### 5. Polynomial Feature Map

```
poly_mode = 'off':        φ(x) = x
poly_mode = 'elementwise': φ(x) = Σ_{i=1}^{g} x^i
poly_mode = 'tensor':      φ(x) = RandomProj([x, vec(x ⊗ x)])
```

### Parallelization via Affine Prefix Scan

The naive RNN update has a sequential dependency on \( S\_{t-1} \), requiring \( O(T) \) sequential steps. Following the approach in the Atlas paper (Section 3.2), we reformulate the update as an **affine transformation** and use a **parallel prefix scan** (pure PyTorch implementation, no external dependencies) to compute all states in \( O(\log T) \) depth.

#### Key Insight: Affine Form

The RNN update can be rewritten as:

```
S_t = S_{t-1} @ A_t + C_t
```

where:

- **A_t** = α_t I − η_t G_t (transition matrix)
- **C_t** = η_t B_t (input term)
- **G_t** = φ_t φ_t^T (Gram matrix)
- **B_t** = v_t φ_t^T (cross term)

This is an **affine recurrence**: each step applies a matrix multiplication plus an addition.

#### With Momentum: Block Affine Form

When using momentum, we have two coupled recurrences:

```
Z_t = β_t Z_{t-1} + g_t
S_t = α_t S_{t-1} − η_t Z_t
```

We stack them into a joint state \( H_t = [S_t, Z_t] \) and build a block-affine system:

```
H_t = H_{t-1} @ A_t + C_t

where A_t = ┌ α_t I − η_t G_t    G_t    ┐    C_t = [ η_t B_t,  −B_t ]
            └ −η_t β_t I        β_t I   ┘
```

#### Associative Scan Algorithm

Since affine transforms compose associatively:

```
(A₁, C₁) ∘ (A₂, C₂) = (A₁ @ A₂, C₁ @ A₂ + C₂)
```

We can use **parallel prefix scan** to compute all states simultaneously:

```python
def _affine_pair_operator(a, b):
    """Compose affine transforms: H -> H @ A + C"""
    A1, C1 = a
    A2, C2 = b
    return (A1 @ A2, C1 @ A2 + C2)

def _affine_scan_apply(H0, A_seq, C_seq):
    """
    Compute H_t = H_{t-1} @ A_t + C_t for all t in O(log T) depth.

    H0:    [B, M, D]       - initial state
    A_seq: [B, T, D, D]    - per-token transition matrices
    C_seq: [B, T, M, D]    - per-token input terms
    Returns: H_all [B, T, M, D]
    """
    # Parallel prefix scan over (A, C) pairs
    A_pref, C_pref = associative_scan(_affine_pair_operator, (A_seq, C_seq))
    # Apply initial state: H_t = H_0 @ A_{1:t} + C_{1:t}
    return torch.einsum('bmd,btde->btme', H0, A_pref) + C_pref
```

#### Implementation in Code

From `rnn_memory.py`:

```python
# Build per-token affine transforms
# G_t = φ_t ⊗ φ_t^T: [BH, T, d, d]
G = torch.einsum('bti,btj->btij', phi_k, phi_k)
# B_t = v_t ⊗ φ_t^T: [BH, T, d, d]
B = torch.einsum('bti,btj->btij', v_bh, phi_k)

if self.use_momentum:
    # Block affine: H=[S,Z], H_t = H_{t-1} @ A_t + C_t
    A11 = decay_e * I - lr_e * G   # S transition
    A12 = G                         # Z -> S coupling
    A21 = -(lr_e * mom_e) * I       # Cross term
    A22 = mom_e * I                 # Z momentum

    A_seq = block_diag(A11, A12, A21, A22)  # [BH, T, 2d, 2d]
    C_seq = concat([lr_e * B, -B], dim=-1)  # [BH, T, d, 2d]

    # Parallel scan: O(log T) depth instead of O(T)
    H_all = _affine_scan_apply(H0, A_seq, C_seq)
else:
    # Simpler: A_t = α_t I - η_t G_t, C_t = η_t B_t
    A_seq = decay_e * I - lr_e * G
    C_seq = lr_e * B
    S_all = _affine_scan_apply(S0, A_seq, C_seq)
```

#### Retrieval Uses Previous State

Critical detail: retrieval at step \( t \) uses the **previous** state \( S\_{t-1} \):

```python
# S_start = [S_0, S_1, ..., S_{T-1}] (shifted by 1)
S_start = torch.cat([S0.unsqueeze(1), S_all[:, :-1]], dim=1)
# y_t = S_{t-1} @ ψ_t
retrieved = torch.einsum('btdp,btp->btd', S_start, phi_q)
```

This ensures the model doesn't "see into the future" during training.

### Comparison with Related Architectures

| Model            | State  | Update             | Decay          | Context        |
| ---------------- | ------ | ------------------ | -------------- | -------------- |
| **Titans-RNN**   | Matrix | GD-derived         | α_t            | e=1 (online)   |
| **OmegaNet-RNN** | Matrix | GD-derived         | α_t            | e≥1 (window)   |
| **RWKV-7**       | Matrix | Designed           | w_t (data-dep) | Token-shift    |
| **Mamba**        | Vector | SSM discretization | Ā_t            | Δ_t (data-dep) |

## Installation

```bash
pip install -e .
pip install -e ".[examples]"
```

## Usage

### RNN Memory (Titans-RNN)

```python
import torch
from atlas_pytorch import RNNMemory

mem = RNNMemory(
    dim = 384,
    heads = 8,
    dim_head = 64,
    use_momentum = True,
).cuda()

seq = torch.randn(2, 1024, 384).cuda()
retrieved, state = mem(seq)
assert seq.shape == retrieved.shape
```

### OmegaNet-RNN Memory

```python
import torch
from atlas_pytorch import OmegaRNNMemory

mem = OmegaRNNMemory(
    dim = 384,
    heads = 8,
    dim_head = 64,
    omega_window = 4,        # context window size
    use_omega_gate = True,   # learnable U gates
    poly_degree = 2,         # polynomial features
    poly_mode = 'elementwise',
    use_momentum = True,
).cuda()

seq = torch.randn(2, 1024, 384).cuda()
retrieved, state = mem(seq)
```

### MAG Transformer (Memory-As-Gate) with RNN Memory

```python
import torch
from atlas_pytorch import MemoryAsGateTransformer

transformer = MemoryAsGateTransformer(
    num_tokens = 256,
    dim = 256,
    depth = 4,
    window_size = 64,
    num_persist_mem_tokens = 4,
    neural_memory_layers = (1, 2, 3, 4),
    # RNN memory options
    use_rnn_memory = True,
    omega_window = 2,
    poly_degree = 2,
    poly_mode = 'elementwise',
)

ids = torch.randint(0, 256, (1, 512))
loss = transformer(ids, return_loss = True)
loss.backward()
```

### MAL Transformer (Memory-As-Layer)

```python
import torch
from atlas_pytorch import MemoryAsLayerTransformer

transformer = MemoryAsLayerTransformer(
    num_tokens = 256,
    dim = 256,
    depth = 4,
    window_size = 64,
    neural_memory_layers = (1, 2, 3, 4),
    use_rnn_memory = True,
    omega_window = 2,
)

ids = torch.randint(0, 256, (1, 512))
loss = transformer(ids, return_loss = True)
loss.backward()
```

### LMM (Long-term Memory Model - Pure RNN)

```python
import torch
from atlas_pytorch import LongTermMemoryModel

model = LongTermMemoryModel(
    num_tokens = 256,
    dim = 256,
    depth = 4,
    omega_window = 2,
    poly_degree = 2,
    poly_mode = 'elementwise',
)

ids = torch.randint(0, 256, (1, 512))
loss = model(ids, return_loss = True)
loss.backward()
```

## Training

### Unified Training Script

All architectures (MAG, MAL, MAC, LMM) with both Titans-RNN and OmegaNet-RNN:

```bash
# Titans-RNN MAG (omega_window=1, no gate)
python train_rnn_transformer.py --arch mag --model titans --omega-window 1

# OmegaNet-RNN MAL (omega_window>1, with gate)
python train_rnn_transformer.py --arch mal --model omeganet --omega-window 4 --use-omega-gate

# LMM (Pure RNN memory, no attention)
python train_rnn_transformer.py --arch lmm --model omeganet --omega-window 2 --poly-degree 2

# MAC (Memory-As-Context)
python train_rnn_transformer.py --arch mac --model titans
```

### Standalone RNN Memory Training

```bash
python train_rnn_memory.py --omega-window 4 --use-momentum --poly-mode elementwise
```

## Tests

```bash
# All tests
pytest -q tests/test_rnn_all.py

# Specific tests
pytest -q tests/test_rnn_all.py -k "affine_scan"
```

## Implementation Notes

### Memory State Structure

```python
RNNMemState = namedtuple('RNNMemState', [
    'seq_index',      # Current sequence position
    'S',              # Memory state matrix [batch*heads, d, d]
    'Z',              # Momentum state (if use_momentum=True)
    'omega_buffer',   # Rolling buffer for Omega window (if omega_window>1)
])
```

### Key Differences from Legacy (titans-pytorch, atlas-pytorch)

| Aspect               | Legacy                        | atlas-rnn                          |
| -------------------- | ----------------------------- | ---------------------------------- |
| Gradient computation | `torch.func.grad`             | Explicit closed-form formula       |
| Parallelization      | `assoc_scan` library (scalar) | Custom affine prefix scan (matrix) |
| Triton dependency    | Required for GPU acceleration | None (pure PyTorch)                |
| Memory type          | MLP parameters (W)            | State matrix (S)                   |

## Provenance / Attribution

This repository derives from:

1. **lucidrains/titans-pytorch**: Original Titans implementation
2. **atlas-pytorch** (by Junyoung Park): Paper-aligned MAC fixes, MAG/MAL/LMM architectures, Atlas/OmegaNet extensions
3. **RWKV-LM** (by BlinkDL): Training optimization techniques for RNN architectures

Key contributions in this fork:

- Derivation of explicit RNN update rules from implicit GD formulation
- Unified codebase for Titans-RNN and OmegaNet-RNN variants

## Citations

```bibtex
@inproceedings{Behrouz2024TitansLT,
    title   = {Titans: Learning to Memorize at Test Time},
    author  = {Ali Behrouz and Peilin Zhong and Vahab S. Mirrokni},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:275212078}
}
```

```bibtex
@article{Sun2024LearningT,
    title   = {Learning to (Learn at Test Time): RNNs with Expressive Hidden States},
    author  = {Yu Sun and Xinhao Li and Karan Dalal and Jiarui Xu and Arjun Vikram and Genghan Zhang and Yann Dubois and Xinlei Chen and Xiaolong Wang and Oluwasanmi Koyejo and Tatsunori Hashimoto and Carlos Guestrin},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2407.04620},
    url     = {https://api.semanticscholar.org/CorpusID:271039606}
}
```

```bibtex
@inproceedings{Yang2024GatedDN,
    title   = {Gated Delta Networks: Improving Mamba2 with Delta Rule},
    author  = {Songlin Yang and Jan Kautz and Ali Hatamizadeh},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:274598177}
}
```

```bibtex
@inproceedings{Nguyen2024TurningUT,
    title   = {Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs},
    author  = {Minh Nguyen and Andrew Baker and Clement Neo and Allen Roush and Andreas Kirsch and Ravid Shwartz-Ziv},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:270870613}
}
```

```bibtex
@article{Zhu2024HyperConnections,
    title   = {Hyper-Connections},
    author  = {Defa Zhu and Hongzhi Huang and Zihao Huang and Yutao Zeng and Yunyao Mao and Banggu Wu and Qiyang Min and Xun Zhou},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2409.19606},
    url     = {https://api.semanticscholar.org/CorpusID:272987528}
}
```

```bibtex
@article{Zhou2024ValueRL,
    title   = {Value Residual Learning For Alleviating Attention Concentration In Transformers},
    author  = {Zhanchao Zhou and Tianyi Wu and Zhiyun Jiang and Zhenzhong Lan},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2410.17897},
    url     = {https://api.semanticscholar.org/CorpusID:273532030}
}
```

```bibtex
@software{Kyrylov_Accelerated_Scan_2024,
    author  = {Kyrylov, Volodymyr},
    doi     = {10.5281/zenodo.10600962},
    title   = {Accelerated Scan},
    version = {0.1.2},
    year    = {2024}
}
```

```bibtex
@misc{wang2025testtimeregressionunifyingframework,
    title   = {Test-time regression: a unifying framework for designing sequence models with associative memory},
    author  = {Ke Alexander Wang and Jiaxin Shi and Emily B. Fox},
    year    = {2025},
    eprint  = {2501.12352},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2501.12352},
}
```

```bibtex
@misc{jordan2024muon,
    author  = {Keller Jordan and Yuchen Jin and Vlado Boza and Jiacheng You and
                    Franz Cesista and Laker Newhouse and Jeremy Bernstein},
    title   = {Muon: An optimizer for hidden layers in neural networks},
    year    = {2024},
    url     = {https://kellerjordan.github.io/posts/muon/}
}
```

```bibtex
@inproceedings{Zhang2025TestTimeTD,
    title   = {Test-Time Training Done Right},
    author  = {Tianyuan Zhang and Sai Bi and Yicong Hong and Kai Zhang and Fujun Luan and Songlin Yang and Kalyan Sunkavalli and William T. Freeman and Hao Tan},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:279071244}
}
```

```bibtex
@inproceedings{Behrouz2025ATLASLT,
    title  = {ATLAS: Learning to Optimally Memorize the Context at Test Time},
    author = {Ali Behrouz and Ze-Minghui Li and Praneeth Kacham and Majid Daliri and Yuan Deng and Peilin Zhong and Meisam Razaviyayn and Vahab S. Mirrokni},
    year   = {2025},
    url    = {https://api.semanticscholar.org/CorpusID:278996373}
}
```
