# Atlas-RNN (PyTorch)

<img alt="Figure 1" src="./fig1.png" width="400px"></img>

<img alt="Figure 2" src="./fig2.png" width="400px"></img>

## Atlas-RNN - PyTorch

**Pure RNN-form implementation of Titans and Atlas (OmegaNet) memory architectures.**

This project derives the explicit RNN update rules from the implicit gradient descent formulation in Titans and Atlas papers, enabling efficient training without per-sample gradient computation via `torch.func`.

## Key Insight: From Implicit GD to Explicit RNN

The Titans and Atlas papers show that linear attention can be viewed as implicit gradient descent on a "surprise" loss. We reverse this insight: by explicitly computing the gradient of the MSE loss, we derive closed-form RNN update rules that are mathematically equivalent but computationally more efficient.

### Mathematical Derivation

---

#### Step 1: Linear Attention as Gradient Descent (Titans)

The Titans paper establishes a fundamental equivalence: **linear attention is implicit gradient descent** on an associative memory objective.

Given a memory matrix $S$ and input key-value pair $(k_t, v_t)$, define the **surprise loss**:

$$\mathcal{L}_t(S) = \frac{1}{2} \| S k_t - v_t \|^2$$

This measures how well the memory $S$ can retrieve $v_t$ when queried with $k_t$.

**Gradient descent update:**

$$S_t = S_{t-1} - \eta_t \nabla_S \mathcal{L}_t(S_{t-1})$$

Computing the gradient:

$$\nabla_S \mathcal{L}_t(S) = (S k_t - v_t) k_t^\top$$

Substituting back:

$$S_t = S_{t-1} - \eta_t (S_{t-1} k_t - v_t) k_t^\top$$

This is exactly the **delta rule** from associative memory literature. The term $(S_{t-1} k_t - v_t)$ is the **prediction error** (or "surprise").

---

#### Step 2: Adding Weight Decay and Momentum (Titans)

Titans extends the basic update with two regularization terms:

**Weight decay** ($\alpha_t$): Prevents unbounded memory growth

$$S_t = \alpha_t S_{t-1} - \eta_t \nabla_S \mathcal{L}_t$$

**Momentum** ($\beta_t$): Accelerates convergence by accumulating gradients

$$Z_t = \beta_t Z_{t-1} + \nabla_S \mathcal{L}_t$$
$$S_t = \alpha_t S_{t-1} - \eta_t Z_t$$

where $Z_t$ is the momentum buffer.

**Expanding the gradient**, we get the full **Titans-RNN update**:

$$\boxed{S_t = \alpha_t S_{t-1} - \eta_t \left( S_{t-1} k_t k_t^\top - v_t k_t^\top \right)}$$

Or equivalently:

$$S_t = S_{t-1} \left( \alpha_t I - \eta_t k_t k_t^\top \right) + \eta_t v_t k_t^\top$$

**With momentum:**

$$Z_t = \beta_t Z_{t-1} + \left( S_{t-1} k_t - v_t \right) k_t^\top$$
$$S_t = \alpha_t S_{t-1} - \eta_t Z_t$$

**Retrieval** (using query $q_t$):

$$y_t = S_{t-1} \, q_t$$

Note: We use the **previous** state $S_{t-1}$ for retrieval to maintain causality.

---

#### Step 3: Polynomial Feature Maps (Atlas)

Atlas introduces **polynomial feature maps** to increase memory capacity. Instead of raw keys, we use:

$$\phi(k) = \text{FeatureMap}(k)$$

Three modes are supported:

| Mode          | Feature Map $\phi(x)$                                                 |
| ------------- | --------------------------------------------------------------------- |
| `off`         | $\phi(x) = x$                                                         |
| `elementwise` | $\phi(x) = \sum_{i=1}^{g} x^{\circ i}$ (element-wise powers)          |
| `tensor`      | $\phi(x) = \text{Proj}\left( [x; \, \text{vec}(x \otimes x)] \right)$ |

The update rules remain the same, but with $\phi(k_t)$ replacing $k_t$:

$$S_t = \alpha_t S_{t-1} - \eta_t \left( S_{t-1} \phi_t \phi_t^\top - v_t \phi_t^\top \right)$$

where $\phi_t = \phi(k_t)$.

---

#### Step 4: Sliding Window Update — The Omega Rule (Atlas)

The key innovation in Atlas is the **Omega rule**: instead of updating based on a single token, aggregate gradients over a sliding window of size $e$.

For window $W_t = \{t-e+1, \ldots, t\}$ with learnable gates $U_t^{(p)}$:

**Aggregated loss:**

$$\mathcal{L}_t^{\Omega}(S) = \sum_{p \in W_t} U_t^{(p)} \cdot \frac{1}{2} \| S \phi_p - v_p \|^2$$

**Aggregated gradient:**

$$\nabla_S \mathcal{L}_t^{\Omega} = \sum_{p \in W_t} U_t^{(p)} \left( S \phi_p - v_p \right) \phi_p^\top$$

$$= S \underbrace{\left( \sum_{p \in W_t} U_t^{(p)} \phi_p \phi_p^\top \right)}_{G_t} - \underbrace{\sum_{p \in W_t} U_t^{(p)} v_p \phi_p^\top}_{B_t}$$

where:

- $G_t = \sum_{p \in W_t} U_t^{(p)} \phi_p \phi_p^\top$ — weighted outer product sum
- $B_t = \sum_{p \in W_t} U_t^{(p)} v_p \phi_p^\top$ — weighted target-key product sum

**OmegaNet-RNN update (without momentum):**

$$\boxed{S_t = S_{t-1} \left( \alpha_t I - \eta_t G_t \right) + \eta_t B_t}$$

**With momentum:**

$$Z_t = \beta_t Z_{t-1} + \left( S_{t-1} G_t - B_t \right)$$
$$S_t = \alpha_t S_{t-1} - \eta_t Z_t$$

---

#### Summary: Unified RNN Form

| Variant          | Window     | Update Rule                                                          |
| ---------------- | ---------- | -------------------------------------------------------------------- |
| **Titans-RNN**   | $e=1$      | $S_t = \alpha_t S_{t-1} - \eta_t (S_{t-1} \phi_t - v_t) \phi_t^\top$ |
| **OmegaNet-RNN** | $e \geq 1$ | $S_t = \alpha_t S_{t-1} - \eta_t (S_{t-1} G_t - B_t)$                |

Both are **explicit RNN updates** derived from implicit gradient descent, enabling efficient parallel computation without `torch.func`.

---

## Parallelization via Efficient Scalar Scan

The naive RNN update has a sequential dependency on $S_{t-1}$, requiring $O(T)$ sequential steps. We use an **efficient scalar scan** approach inspired by the Atlas paper, achieving $O(\log T)$ parallel depth with memory-efficient implementation.

---

### Key Insight: Delta Decomposition

The Atlas paper computes gradients using a **fixed reference state** $S_0$ (chunk-start state):

$$\delta_t = -\eta_t \left( S_0 G_t - B_t \right)$$

This decouples the gradient computation from the state sequence, allowing **fully parallel** computation of all $\delta_t$.

---

### Memory Update as Scalar-Gated Recurrence

With the delta terms pre-computed, the state update becomes a simple **scalar-gated** recurrence:

$$S_t = \alpha_t S_{t-1} + \delta_t$$

where $\alpha_t$ is a **scalar** decay factor. This enables efficient parallel scan with minimal memory overhead.

---

### With Momentum

When using momentum, we have:

$$Z_t = \beta_t Z_{t-1} + g_t \quad \text{where } g_t = S_0 G_t - B_t$$
$$\delta_t = -\eta_t Z_t$$
$$S_t = \alpha_t S_{t-1} + \delta_t$$

The momentum buffer $Z_t$ is also updated via scalar scan.

---

### Algorithm (Memory-Efficient)

```python
def efficient_forward(phi_k, v_bh, S_ref, S0, Z0, alpha, beta, lr, omega_gate=None, omega_window=1, prev_g=None):

    BH, T, d = phi_k.shape

    # Step 1: Prediction error vector
    # δ_t = S_ref @ φ_t - v_t
    pred = torch.einsum('bde,bte->btd', S_ref, phi_k)    # [BH, T, d]
    delta_vec = pred - v_bh                               # [BH, T, d]

    # Step 2: Per-token gradient (rank-1, before omega window)
    # g_raw_t = δ_t ⊗ φ_t^T
    g_raw = torch.einsum('bti,btj->btij', delta_vec, phi_k)  # [BH, T, d, d]
    if omega_gate is not None:
        g_raw = g_raw * omega_gate[..., None, None]

    # Step 3: Omega window on g (single sliding-sum buffer)
    if omega_window > 1:
        if prev_g is None:
            prev_g = g_raw.new_zeros((BH, omega_window - 1, d, d))
        g_ext = torch.cat([prev_g, g_raw], dim=1)               # [BH, (e-1)+T, d, d]
        g = sliding_sum(g_ext, window=omega_window)[:, -T:]     # [BH, T, d, d]
    else:
        g = g_raw

    # Step 4: Momentum + state via scalar scan
    Z_all = scalar_scan(beta, g, Z0)                            # [BH, T, d, d]
    delta = -lr[..., None, None] * Z_all                        # [BH, T, d, d]
    S_all = scalar_scan(alpha, delta, S0)                       # [BH, T, d, d]

    return S_all
```

---

### Why This Works (Mathematical Justification)

This is the **exact same algorithm** used in the original Titans paper (Section 3.2):

1. **Per-sample gradient**: Gradients are computed w.r.t. a fixed "chunk-start" state $S_0$
2. **Parallel computation**: All $\delta_t$ terms can be computed independently
3. **State accumulation**: Final states computed via efficient scalar scan

---

### Implementation in Code

From `rnn_memory.py`:

```python
# Exact refactor: build gradient directly (no separate G/B tensors)
pred = torch.einsum('bde,bte->btd', S_ref, phi_k)          # [BH, T, d]
delta_vec = pred - v_bh                                     # [BH, T, d]
g_raw = torch.einsum('bti,btj->btij', delta_vec, phi_k)     # [BH, T, d, d]

# Apply omega gate to g (not to G/B separately)
if exists(omega_gate):
    g_raw = g_raw * omega_gate[..., None, None]

# Omega window: sliding sum of g (single buffer, not G+B)
if omega_window > 1:
    prev_g = omega_buffer if exists(omega_buffer) else g_raw.new_zeros((BH, omega_window - 1, d, d))
    g_ext = torch.cat([prev_g, g_raw], dim=1)
    g = _sliding_sum_along_time(g_ext, omega_window)[:, -seq_len:]
else:
    g = g_raw

# Momentum via scalar scan
Z_all = self.assoc_scan(momentum, g, prev=Z0)

# Delta and state via scalar scan
delta = -lr_e * Z_all
S_all = self.assoc_scan(decay, delta, prev=S0)
```

### Retrieval Uses Previous State

Critical detail: retrieval at step $t$ uses the **previous** state $S_{t-1}$:

$$y_t = S_{t-1} \, \psi_t \quad \text{where } \psi_t = \phi(q_t)$$

In code:

```python
# y_t = S_{t-1} @ ψ_t  (avoid concatenating full S_start)
y0 = torch.einsum('bde,be->bd', S0, phi_q[:, 0])  # [BH, d]
if seq_len > 1:
    y_rest = torch.einsum('btdp,btp->btd', S_all[:, :-1], phi_q[:, 1:])  # [BH, T-1, d]
    retrieved = torch.cat([y0.unsqueeze(1), y_rest], dim=1)              # [BH, T, d]
else:
    retrieved = y0.unsqueeze(1)
```

This ensures the model doesn't "see into the future" during training.

---

### Comparison with Related Architectures

| Model            | State  | Update             | Decay            | Context               |
| ---------------- | ------ | ------------------ | ---------------- | --------------------- |
| **Titans-RNN**   | Matrix | GD-derived         | $\alpha_t$       | $e=1$ (online)        |
| **OmegaNet-RNN** | Matrix | GD-derived         | $\alpha_t$       | $e \geq 1$ (window)   |
| **RWKV-7**       | Matrix | Designed           | $w_t$ (data-dep) | Token-shift           |
| **Mamba**        | Vector | SSM discretization | $\bar{A}_t$      | $\Delta_t$ (data-dep) |

## Installation

```bash
pip install -e .
pip install -e ".[examples]"
```

## CUDA Backend (Omega=16, optional)

Atlas-RNN includes an **optional CUDA backend** for the _exact_ Omega sliding-window update when:

- **`omega_window == 16`**, and
- you pass **`use_cuda=True`** to `OmegaRNNMemoryCell`, and
- you run on an NVIDIA GPU with CUDA toolkit available.

This CUDA path was made numerically stable for long backstepping by using **checkpoint anchoring (K=16)** inside the CUDA implementation.

### Compile the CUDA extension (A100 recommended)

On a CUDA machine with `nvcc`:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
pip install -U pip ninja

cd atlas_pytorch/cuda
python3 compile.py
```

If you see `CUDA_HOME environment variable is not set`, set `CUDA_HOME` as above.

### Use the CUDA backend

```python
import torch
from atlas_pytorch.rnn_memory import OmegaRNNMemoryCell

cell = OmegaRNNMemoryCell(
    dim=896,
    dim_head=64,
    heads=14,
    omega_window=16,     # required for CUDA backend
    use_omega_gate=True,
    use_momentum=True,
    poly_degree=2,
    poly_mode="elementwise",
    use_cuda=True,
).to("cuda", dtype=torch.bfloat16)
```

Notes:

- The CUDA kernel operates in **bf16**; keep module + inputs in bf16 for best behavior.
- `omega_window != 16` will **automatically fall back** to the PyTorch path.
- When the CUDA backend is active, `assoc_scan` / Triton acceleration is **not used** for Omega=16.

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
pytest -q tests/test_rnn_all.py -k "assoc_scan"
```

### GPU / CUDA tests

On a CUDA machine (after compiling the extension above):

```bash
pytest -q tests/test_atlas_cuda.py
pytest -q tests/test_cuda_integration.py
pytest -q tests/test_atlas_performance.py -k "memory_" -s
```

> PyTorch vs CUDA parity: the PyTorch path uses `assoc_scan` (different accumulation order),
> while the CUDA kernel uses explicit float accumulations. In bf16 this can produce small numeric differences.

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

| Aspect               | Legacy             | atlas-rnn                    |
| -------------------- | ------------------ | ---------------------------- |
| Gradient computation | `torch.func.grad`  | Explicit closed-form formula |
| Memory type          | MLP parameters (W) | State matrix (S)             |

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
