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
- [x] **Goal #8: RoPE & Autoregressive Generation**: Rotary Positional Embeddings, Embedding/LMHead layers, and token-by-token text generation.
- [x] **Goal #9: KV Cache Inference**: Prefill/Decode loop with KV-cached attention — zero redundant recomputation.
- [x] **Goal #10: INT8 Quantization**: 4x weight compression with on-the-fly GPU dequantization (0.37% error).
- [x] **Bare-Metal Kernels**: 17 hand-written MSL kernels: GEMM, Q8-GEMM, RoPE, Softmax, GELU, LayerNorm, and more.

## Architecture

| Component | Description |
|-----------|-------------|
| `Tensor.swift` | The foundation. Manages page-aligned CPU/GPU shared memory with serialization support. |
| `MetalEngine.swift` | The control plane. Handles device discovery, command queues, and kernel caching. |
| `TransformerBlock.swift` | The LLM core. Pre-Norm with MHA, FFN(GELU), LayerNorm, Residual Connections. Supports KV-cached decode. |
| `MultiHeadAttention.swift` | Parallelized attention with **RoPE** positional encoding and KV-cached single-token decode. |
| `KVCache.swift` | Inference optimizer. Per-head Key/Value buffers with position tracking for Prefill/Decode. |
| `EmbeddingLayer.swift` | Token ID → dense vector lookup via GPU kernel. |
| `LMHead.swift` | Final projection to vocabulary logits with greedy argmax decoding. |
| `SelfAttention.swift` | Single-head attention. Q/K/V projections and Scaled Dot-Product Attention. |
| `Interconnect.swift` | The network layer. Asynchronous TCP bridge using **SwiftNIO**. |
| `Sequential.swift` | The orchestrator. Chains layers for multi-layer forward/backward flow. |
| `LinearLayer.swift` | The building block. Manages parameters, gradients, and SGD optimization. |
| `QuantizedLinearLayer.swift` | INT8 inference. 4x weight compression with on-the-fly GPU dequantization. |
| `QuantizedTensor.swift` | Quantized storage. INT8 weights + FP32 per-row scale factors. |
| `gemm.metal` | The math. 17 kernels: GEMM, Q8-GEMM, RoPE, Softmax, GELU, LayerNorm, and more. |
| `yanaiengine.swift` | The entry point. Autoregressive text generation with full LLM pipeline. |

## Quick Start

### Prerequisites
- A Mac with **Apple Silicon** (M1, M2, M3 series).
- **Xcode 15+** or the **Swift 6.0+** toolchain.

### Running the KV-Cached LLM Demo
```bash
cd yanaiengine
swift run
```
Runs a **Prefill/Decode** inference loop: processes the prompt in parallel (Prefill), then generates tokens one at a time using a KV Cache (Decode) — the same architecture as vLLM and TensorRT-LLM.

### Running via Xcode
1. In Xcode, select **File > Open...** and select the `yanaiengine` folder (or `Package.swift`).
2. Ensure the `yanaiengine` target is selected.
3. Press `Cmd + R` to Build and Run.

## Performance & UMA
Unlike traditional discrete GPU setups, `YanAIEngine` uses `.storageModeShared`. This means the CPU writes inputs into a memory region that the GPU can see immediately without any PCIe transfer overhead. By chaining kernels in a single `MTLCommandBuffer`, we ensure the GPU remains fully utilized while the CPU stays asynchronous, proving that a true Deep Learning framework can be built from the silicon up.
