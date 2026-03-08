# YanAIEngine 🚀

`YanAIEngine` is a high-performance AI compute engine built natively for Apple Silicon. It bypasses high-level frameworks to demonstrate direct control over the GPU via Swift and the Metal Shading Language (MSL).

## Core Philosophy: The Native Advantage
Modern AI models live and die by their matrix operations. `YanAIEngine` focuses on the **Silicon & Kernel Layer**, exploiting the Apple Unified Memory Architecture (UMA) to achieve zero-copy data sharing between the CPU and GPU.

### Key Milestones Completed
- [x] **Zero-Copy Memory Bridge**: `Tensor` structures backed by `MTLBuffer` with shared storage mode.
- [x] **Deep Learning & Chain Rule**: Full backpropagation support across multiple layers.
- [x] **Sequential Model Abstraction**: Stackable `LinearLayer` components managed by a `Sequential` container.
- [x] **Goal #4: Non-Linear Solving**: Successfully trained to solve the XOR problem natively on the GPU.
- [x] **Goal #5: Distributed Interconnect**: Multi-node synchronization via **SwiftNIO** with All-Reduce gradient averaging.
- [x] **Goal #6: Transformer Attention**: Scaled Dot-Product Self-Attention with Softmax, Scaling, and Causal Masking.
- [x] **Goal #7: Full Transformer Block**: Multi-Head Attention, LayerNorm, GELU, Residual Connections — the exact architecture of GPT/Llama.
- [x] **Bare-Metal Kernels**: 14 hand-written MSL kernels: GEMM, Transpose, Softmax, Scale, Causal Mask, GELU, LayerNorm, Element-wise Add, and more.

## Architecture

| Component | Description |
|-----------|-------------|
| `Tensor.swift` | The foundation. Manages page-aligned CPU/GPU shared memory with serialization support. |
| `MetalEngine.swift` | The control plane. Handles device discovery, command queues, and kernel caching. |
| `TransformerBlock.swift` | The LLM core. Pre-Norm architecture with MHA, FFN(GELU), LayerNorm, and Residual Connections. |
| `MultiHeadAttention.swift` | Parallelized attention. Splits Q/K/V into h heads with batched computation. |
| `SelfAttention.swift` | Single-head attention. Q/K/V projections and Scaled Dot-Product Attention. |
| `Interconnect.swift` | The network layer. Asynchronous TCP bridge using **SwiftNIO**. |
| `Sequential.swift` | The orchestrator. Chains layers for multi-layer forward/backward flow. |
| `LinearLayer.swift` | The building block. Manages parameters, gradients, and SGD optimization. |
| `gemm.metal` | The math. 14 kernels: GEMM, Transpose, Softmax, Scale, GELU, LayerNorm, and more. |
| `yanaiengine.swift` | The entry point. Demos a full TransformerBlock with MHA verification. |

## Quick Start

### Prerequisites
- A Mac with **Apple Silicon** (M1, M2, M3 series).
- **Xcode 15+** or the **Swift 6.0+** toolchain.

### Running the Transformer Block Demo
```bash
cd yanaiengine
swift run
```
Runs a 4-token sequence through a full **Transformer Block** (LayerNorm → Multi-Head Attention → Residual → LayerNorm → FFN(GELU) → Residual), verifying shape preservation, transformation, and numerical stability.

### Running via Xcode
1. In Xcode, select **File > Open...** and select the `yanaiengine` folder (or `Package.swift`).
2. Ensure the `yanaiengine` target is selected.
3. Press `Cmd + R` to Build and Run.

## Performance & UMA
Unlike traditional discrete GPU setups, `YanAIEngine` uses `.storageModeShared`. This means the CPU writes inputs into a memory region that the GPU can see immediately without any PCIe transfer overhead. By chaining kernels in a single `MTLCommandBuffer`, we ensure the GPU remains fully utilized while the CPU stays asynchronous, proving that a true Deep Learning framework can be built from the silicon up.
