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
- [x] **Bare-Metal Kernels**: Hand-written MSL for GEMM, Transposition, Bias Addition, Row Summing, MSE Derivative, ReLU Derivative, and SGD Optimization.

## Architecture

| Component | Description |
|-----------|-------------|
| `Tensor.swift` | The foundation. Manages page-aligned CPU/GPU shared memory with serialization support. |
| `MetalEngine.swift` | The control plane. Handles device discovery, command queues, and kernel caching. |
| `Interconnect.swift` | The network layer. High-performance, asynchronous TCP bridge using **SwiftNIO**. |
| `Sequential.swift` | The orchestrator. Chains layers and manages the multi-layer forward/backward flow. |
| `LinearLayer.swift` | The building block. Manages parameters, calculates gradients (dW, db, dX), and applies optimization. |
| `gemm.metal` | The math. Optimized kernels for linear algebra and deep learning operations. |
| `yanaiengine.swift` | The entry point. Supports standalone, Master, and Worker roles for distributed training. |

## Quick Start

### Prerequisites
- A Mac with **Apple Silicon** (M1, M2, M3 series).
- **Xcode 15+** or the **Swift 6.0+** toolchain.

### Running the Distributed Demo
To test the interconnect on a single machine, open two terminals:

**Terminal 1 (Master):**
```bash
swift run yanaiengine --master
```

**Terminal 2 (Worker):**
```bash
swift run yanaiengine --worker
```

The nodes will automatically discover each other, split the XOR dataset, and synchronize gradients using the **All-Reduce** protocol over TCP.

### Running via Xcode
1. In Xcode, select **File > Open...** and select the `yanaiengine` folder (or `Package.swift`).
2. Ensure the `yanaiengine` target is selected.
3. Press `Cmd + R` to Build and Run.

## Performance & UMA
Unlike traditional discrete GPU setups, `YanAIEngine` uses `.storageModeShared`. This means the CPU writes inputs into a memory region that the GPU can see immediately without any PCIe transfer overhead. By chaining kernels in a single `MTLCommandBuffer`, we ensure the GPU remains fully utilized while the CPU stays asynchronous, proving that a true Deep Learning framework can be built from the silicon up.
