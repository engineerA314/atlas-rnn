from atlas_pytorch.neural_memory import (
    NeuralMemory,
    NeuralMemState,
    mem_state_detach
)

from atlas_pytorch.memory_models import (
    MemoryMLP,
    MemoryAttention,
    FactorizedMemoryMLP,
    MemorySwiGluMLP,
    GatedResidualMemoryMLP
)

from atlas_pytorch.mac_transformer import (
    MemoryAsContextTransformer
)

from atlas_pytorch.omega import (
    OmegaNeuralMemory
)

from atlas_pytorch.mag_transformer import (
    MemoryAsGateTransformer
)

from atlas_pytorch.mal_transformer import (
    MemoryAsLayerTransformer,
    AtlasLMM
)

# RNN-form memory implementations
from atlas_pytorch.rnn_memory import (
    RNNMemory,
    RNNMemoryCell,
    OmegaRNNMemory,
    OmegaRNNMemoryCell,
    RNNMemState,
    state_detach,
    PolynomialFeatureMap,
)

# RNN-form transformer architectures
from atlas_pytorch.rnn_transformer import (
    RNNMemoryTransformer,
    MAGBlock,
    MALBlock,
    LMMBlock,
    # Convenience aliases (note: these shadow the original implementations)
    # Use RNNMemoryTransformer with block_type for explicit control
)
