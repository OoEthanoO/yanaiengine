# YanAIEngine 🚀

`YanAIEngine` is an absolute titan of a multimodal inference engine built natively for Apple Silicon. It bypasses high-level frameworks to demonstrate direct control over the GPU via Swift and the Metal Shading Language (MSL), enabling state-of-the-art inference at scale.

## Core Philosophy: The Native Advantage
Modern AI models live and die by their memory bandwidth. `YanAIEngine` focuses on the **Silicon & Kernel Layer**, exploiting the Apple Unified Memory Architecture (UMA) to achieve zero-copy data sharing between the CPU and GPU, and implementing algorithmic breakthroughs to break the memory wall.

### Key Milestones Completed
- [x] **Zero-Copy Memory Bridge**: `Tensor` structures backed by `MTLBuffer` with shared storage mode.
- [x] **Transformer Attention (Goal #6)**: Scaled Dot-Product Self-Attention with Softmax, Scaling, and Causal Masking.
- [x] **Goal #12: Llama 3 Architecture**: RMSNorm, SwiGLU FFN, Grouped Query Attention (GQA), and BPE Tokenizer.
- [x] **Goal #14: FlashAttention (Kernel Fusion)**: Smashed the "Memory Wall" with a fused kernel using Online Softmax and tiling to bypass VRAM bottlenecks.
- [x] **Goal #15: Inference Server & Gemini API**: Hummingbird-powered microservice implementing the Google Gemini API contract (`generateContent` + SSE Streaming).
- [x] **Goal #16: Google Gemma 2 Architecture**: Polymorphic support for **Logit Soft-Capping**, **GeGLU Activation**, and **Alternating Sliding Window Attention (SWA)**.
- [x] **Goal #17: PagedAttention (Memory Virtualization)**: Virtualized KV Cache with a hand-written block allocator and page table system to eliminate VRAM fragmentation.
- [x] **Goal #18: Continuous Batching**: ORCA-style scheduling that interleaves prefill and decode phases for massive concurrent throughput.
- [x] **Goal #19: Mixture of Experts (MoE)**: Sparse routing for trillion-parameter scale models (Mixtral-style).
- [x] **Goal #20: Multimodal VLM (PaliGemma)**: Integrated **SigLIP Vision Encoder** for high-density image reasoning.
- [x] **Goal #21: Speculative Decoding (Draft-Verify)**: Breaking the **Memory Bandwidth Wall** using a draft-verify pipeline that doubles or triples generation speed.
- [x] **Bare-Metal Kernels**: 28 hand-written MSL kernels including `gemm`, `rope`, `rmsnorm`, `paged_fused_attention_kernel`, and `patch_embedding_kernel`.

## Architecture

| Component | Description |
|-----------|-------------|
| `Tensor.swift` | The foundation. Manages page-aligned CPU/GPU shared memory with zero-copy buffer sharing. |
| `MetalEngine.swift` | The control plane. Handles device discovery, command queues, and kernel caching. |
| `Scheduler.swift` | The brain. Manages continuous batching and the **Speculative Decoding** draft-verify loop. |
| `LlamaModel.swift` | Llama 3/4 orchestrator. Optimized for $O(1)$ inference with `forwardStep` and batch verification. |
| `SpeculativeSampler.swift` | The verify engine. Implements rejection sampling to validate draft tokens against the target model. |
| `PagedKVCache.swift` | Virtual mapping. Uses Page Tables to map logical sequences to physical blocks in the pool. |
| `BlockAllocator.swift` | Physical pool. Pre-allocates VRAM blocks (16 tokens) to eliminate fragmentation. |
| `SigLIPEncoder.swift` | Vision Transformer (ViT). Encodes raw pixels into high-density visual embeddings. |
| `MultimodalProjector.swift` | The bridge. Aligns visual latent spaces with the LLM's dimensional space. |
| `MoERouter.swift` | Gating network. Sparsely dispatches tokens to expert-partitioned Feed-Forward Networks. |
| `InferenceServer.swift` | The API layer. Asynchronous server exposing a unified Gemini/OpenAI-compatible interface. |
| `gemm.metal` | The math. Hand-optimized C++/MSL kernels for maximum compute utilization. |

## Performance & Infrastructure

### Breaking the Bandwidth Wall (Speculative Decoding)
During autoregressive decoding, GPU compute cores often sit idle while waiting for massive weight matrices to be fetched from memory. `YanAIEngine` implements **Speculative Decoding (Goal #21)**: a technique where a tiny, lightning-fast "draft" model guesses the next several tokens, and the massive "target" model verifies them in parallel. This converts a memory-bound sequential problem into a compute-bound parallel one, often **doubling or tripling generation speed** on local hardware.

### $O(1)$ Throughput (PagedAttention & Continuous Batching)
By virtualizing the KV Cache, we eliminate the need for contiguous VRAM. The engine chops memory into small "Pages" managed by a **Block Allocator**, allowing the **Scheduler** to interleave processing for many users simultaneously. This solves the "Memory Wall" (fragmentation) and enables high-concurrency serving without performance degradation.

### Multimodal Reasoning (Vision-Language Fusion)
`YanAIEngine` is fully multimodal. It uses a **SigLIP Vision Encoder** to process image patches, which are then fused with text tokens via a **Multimodal Projector**. This allows the engine to "see" and "read" simultaneously, enabling visual question answering and complex scene reasoning.

## Quick Start

### Swift CLI Demo
```bash
# Processes a prompt and generates text natively on the GPU
swift run yanaiengine
```

### Gemini-Compatible Server
```bash
# Boot the HTTP server on port 8080
swift run yanaiengine --server
```

### Querying the Multimodal API
```bash
# Example: Ask about an image using the Gemini API schema
curl http://localhost:8080/v1beta/models/yanai-model:generateContent \
    -X POST -H "Content-Type: application/json" \
    -d '{
      "contents": [{
        "parts": [
          {"text": "What is in this image?"},
          {"inline_data": {"mime_type": "image/png", "data": "BASE64_DATA"}}
        ]
      }]
    }'
```

