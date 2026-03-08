# YanAIEngine 🚀

`YanAIEngine` is a high-performance AI compute engine built natively for Apple Silicon. It bypasses high-level frameworks to demonstrate direct control over the GPU via Swift and the Metal Shading Language (MSL).

## Core Philosophy: The Native Advantage
Modern AI models live and die by their matrix operations. `YanAIEngine` focuses on the **Silicon & Kernel Layer**, exploiting the Apple Unified Memory Architecture (UMA) to achieve zero-copy data sharing between the CPU and GPU.

### Key Milestones Completed
- [x] **Zero-Copy Memory Bridge**: `Tensor` structures backed by `MTLBuffer` with shared storage mode.
- [x] **Deep Learning & Chain Rule**: Full backpropagation support across multiple layers.
- [x] **Sequential Model Abstraction**: Stackable `LinearLayer` components managed by a `Sequential` container.
- [x] **Non-Linear Solving**: Successfully trained to solve the XOR problem natively on the GPU.
- [x] **Bare-Metal Kernels**: Hand-written MSL for GEMM, Transposition, Bias Addition, Row Summing, MSE Derivative, ReLU Derivative, and SGD Optimization.

## Architecture

| Component | Description |
|-----------|-------------|
| `Tensor.swift` | The foundation. Manages page-aligned CPU/GPU shared memory. |
| `MetalEngine.swift` | The control plane. Handles device discovery, command queues, and kernel caching. |
| `Sequential.swift` | The orchestrator. Chains layers and manages the multi-layer forward/backward flow. |
| `LinearLayer.swift` | The building block. Manages parameters, calculates gradients (dW, db, dX), and applies optimization. |
| `gemm.metal` | The math. Optimized kernels for linear algebra and deep learning operations. |
| `yanaiengine.swift` | The entry point. Demonstrates a multi-layer training loop solving the XOR problem. |

## Quick Start

### Prerequisites
- A Mac with **Apple Silicon** (M1, M2, M3 series).
- **Xcode 15+** or the **Swift 6.0+** toolchain.

### Running the Training Demo
```bash
cd yanaiengine
swift run
```
The demo will train a 2-layer neural network (2 -> 16 -> 1) to solve the **XOR Problem**, demonstrating non-linear learning convergence over 2000 epochs.

### Running via Xcode
1. In Xcode, select **File > Open...** and select the `yanaiengine` folder (or `Package.swift`).
2. Ensure the `yanaiengine` target is selected.
3. Press `Cmd + R` to Build and Run.

## Performance & UMA
Unlike traditional discrete GPU setups, `YanAIEngine` uses `.storageModeShared`. This means the CPU writes inputs into a memory region that the GPU can see immediately without any PCIe transfer overhead. By chaining kernels in a single `MTLCommandBuffer`, we ensure the GPU remains fully utilized while the CPU stays asynchronous, proving that a true Deep Learning framework can be built from the silicon up.
