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
- [x] **Goal #11: Safetensors Loader**: Zero-copy `.safetensors` parser with POSIX `mmap` — ready for HuggingFace models.
- [x] **Goal #12: Llama 3 Architecture**: RMSNorm, SwiGLU FFN, Grouped Query Attention (GQA), and BPE Tokenizer.
- [x] **Goal #13: Full Model & Sampling**: Stacked LlamaModel, Temperature/Top-K/Top-P nucleus sampling, Llama 3 chat templates.
- [x] **Goal #14: FlashAttention (Kernel Fusion)**: Smashed the "Memory Wall" with a fused kernel using Online Softmax and tiling to bypass VRAM bottlenecks.
- [x] **Goal #15: Inference Server & Gemini API**: Turn the engine into a deployable microservice. Implements the Google Gemini API contract (`generateContent` + SSE Streaming) via Hummingbird 2.0.
- [x] **Goal #16: Google Gemma 2 Architecture**: Polymorphic support for Google's **Logit Soft-Capping**, **GeGLU Activation**, and **Alternating Sliding Window Attention (SWA)**.
- [x] **Bare-Metal Kernels**: 22 hand-written MSL kernels including `gemm`, `q8_gemm`, `rope`, `rmsnorm`, `gelu`, `logit_softcap_kernel`, and the enhanced `fused_attention_kernel`.

## Architecture

| Component | Description |
|-----------|-------------|
| `Tensor.swift` | The foundation. Manages page-aligned CPU/GPU shared memory with serialization support. |
| `MetalEngine.swift` | The control plane. Handles device discovery, command queues, and kernel caching. |
| `LlamaBlock.swift` | Llama 3 block. Optimized for $O(1)$ inference with `forwardCached` and fused attention. |
| `GemmaBlock.swift` | Google Gemma 2 block. Implements GeGLU, Logit Soft-Capping (50.0), and Sliding Window Attention. |
| `GemmaModel.swift` | Gemma orchestrator. Alternates between Global Attention and SWA across even/odd layers. |
| `LlamaModel.swift` | Llama orchestrator. Stacks N layers with per-layer `KVCache` and synchronized weight-sharing. |
| `Sampler.swift` | Probabilistic decoding. Temperature, Top-K, Top-P (Nucleus) sampling. |
| `MultiHeadAttention.swift` | **FlashAttention** powered. Single-pass GPU kernel for prefill with RoPE and Online Softmax. |
| `KVCache.swift` | Inference optimizer. Per-head Key/Value buffers with position tracking for true autoregressive generation. |
| `EmbeddingLayer.swift` | Token ID → dense vector lookup with optimized single-token `forwardDecode` path. |
| `Tokenizer.swift` | BPE tokenizer. Parses `tokenizer.json` with standard byte-pair encoding merge rules. |
| `SafetensorsReader.swift` | HuggingFace bridge. Parses `.safetensors` files via POSIX `mmap` (zero-copy). |
| `gemm.metal` | The math. 20 kernels implementing every layer natively in C++/MSL. |
| `Interconnect.swift` | The network layer. Asynchronous multi-node synchronization using **SwiftNIO**. |
| `InferenceServer.swift` | The API layer. Asynchronous Hummingbird server exposing the **Gemini API** via SSE streaming. |
| `GeminiSchema.swift` | The contract. `Codable` Swift structs mirroring Google's Gemini v1beta JSON schema. |
| `KVCache.swift` | Enhanced buffer. Supports standard and **Sliding Window** mode via circular indexing. |

## Performance & Infrastructure

### Bypassing the Memory Wall (FlashAttention)
In standard attention, computing scores for 4,000 tokens creates a massive $N \times N$ bottleneck. `YanAIEngine` implements **FlashAttention (Goal #14)**: a single fused kernel that computes dot-product scores, scaling, masking, and softmax as a tiled stream. Intermediate data stays in the GPU's ultra-fast L1 cache (Threadgroup Memory), reducing global VRAM traffic and enabling massive context windows.

### $O(1)$ Autoregressive Inference
By implementing a full **KV-Cache (Goal #9)**, the engine achieves true $O(1)$ complexity per token during generation. Instead of recomputing the past, each layer remembers its $K$ and $V$ states, allowing the decode pass to process only the *newest* token, just like production inference engines like vLLM.

## Quick Start

### Swift CLI Demo
```bash
# Processes a prompt and generates text token-by-token (KV Cache)
swift run yanaiengine
```

### Gemini-Compatible Server
```bash
# Boot the HTTP server on port 8080
swift run yanaiengine --server
```

### Querying the API
```bash
# In a separate terminal:
curl http://localhost:8080/v1beta/models/yanai-model:generateContent \
    -X POST -H "Content-Type: application/json" \
    -d '{"contents": [{"role": "user", "parts": [{"text": "Hello!"}]}]}'
```
