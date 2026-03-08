# YanAIEngine 🚀

`YanAIEngine` is a high-performance AI compute engine built natively for Apple Silicon. It bypasses high-level frameworks to demonstrate direct control over the GPU via Swift and the Metal Shading Language (MSL).

## Core Philosophy: The Native Advantage
Modern AI models live and die by their matrix operations. `YanAIEngine` focuses on the **Silicon & Kernel Layer**, exploiting the Apple Unified Memory Architecture (UMA) to achieve zero-copy data sharing between the CPU and GPU.

### Key Milestones Completed
- [x] **Zero-Copy Memory Bridge**: `Tensor` structures backed by `MTLBuffer` with shared storage mode.
- [x] **Bare-Metal GEMM**: Hand-written Metal compute kernels for General Matrix Multiplication.
- [x] **Integrated Swift Launcher**: A unified executable that handles GPU handshakes, threadgroup dispatching, and synchronization.

## Architecture

| Component | Description |
|-----------|-------------|
| `Tensor.swift` | The foundation. Manages page-aligned CPU/GPU shared memory. |
| `MetalEngine.swift` | The control plane. Handles device discovery, command queues, and resilient shader loading. |
| `gemm.metal` | The compute kernel. High-performance C++ based MSL for matrix math. |
| `yanaiengine.swift` | The entry point. Coordinates the end-to-end pipeline. |

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
Unlike traditional discrete GPU setups, `YanAIEngine` uses `.storageModeShared`. This means the CPU writes the input matrices into a memory region that the GPU can see immediately without any PCIe transfer overhead. This "Engine Block" is the foundation for future layers, backpropagation, and distributed networking.
