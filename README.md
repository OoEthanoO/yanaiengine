# YanAIEngine 🚀

`YanAIEngine` is a high-performance AI compute engine built natively for Apple Silicon. It bypasses high-level frameworks to demonstrate direct control over the GPU via Swift and the Metal Shading Language (MSL).

## Core Philosophy: The Native Advantage
Modern AI models live and die by their matrix operations. `YanAIEngine` focuses on the **Silicon & Kernel Layer**, exploiting the Apple Unified Memory Architecture (UMA) to achieve zero-copy data sharing between the CPU and GPU.

### Key Milestones Completed
- [x] **Zero-Copy Memory Bridge**: `Tensor` structures backed by `MTLBuffer` with shared storage mode.
- [x] **Chained Forward Pass**: Executing complex layers (Linear + ReLU) in a single GPU command buffer without CPU intervention.
- [x] **Bare-Metal Kernels**: Hand-written MSL for GEMM, Bias Addition (broadcasting), and ReLU activation.
- [x] **Modular Abstractions**: Higher-level Swift wrappers like `LinearLayer` that encapsulate low-level dispatch logic.

## Architecture

| Component | Description |
|-----------|-------------|
| `Tensor.swift` | The foundation. Manages page-aligned CPU/GPU shared memory. |
| `MetalEngine.swift` | The control plane. Handles device discovery, command queues, and caching multiple compute pipelines. |
| `LinearLayer.swift` | The building block. Manages weights/biases and chains kernels for a complete forward pass. |
| `gemm.metal` | The math. optimized kernels for GEMM, Bias Addition, and ReLU. |
| `yanaiengine.swift` | The entry point. Demonstrates the computational graph execution. |

## Quick Start

### Prerequisites
- A Mac with **Apple Silicon** (M1, M2, M3 series).
- **Xcode 15+** or the **Swift 6.0+** toolchain.

### Running via Terminal
```bash
cd yanaiengine
swift run
```

### Running via Xcode
1. In Xcode, select **File > Open...** and select the `yanaiengine` folder.
2. Ensure the `yanaiengine` target is selected.
3. Press `Cmd + R` to Build and Run.

## Performance & UMA
Unlike traditional discrete GPU setups, `YanAIEngine` uses `.storageModeShared`. This means the CPU writes the input matrices into a memory region that the GPU can see immediately without any PCIe transfer overhead. By chaining kernels in a single `MTLCommandBuffer`, we ensure the GPU remains fully utilized while the CPU stays asynchronous.

---
*Part of the YanAIEngine Project.*
