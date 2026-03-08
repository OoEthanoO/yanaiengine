import Foundation
import Metal

@main
struct yanaiengine {
    static func main() {
        print("==============================================")
        print("Starting Native Swift-to-Metal GEMM Pipeline")
        print("==============================================\n")
        
        // Initialize Step 2/3/5: Metal Engine
        guard let engine = MetalEngine() else {
            fatalError("Failed to initialize Metal Engine")
        }
        
        // Use the robust shader loader (works in Xcode and CLI)
        engine.loadLibrary(resourceName: "gemm", kernelName: "gemm_kernel")
        
        // Define dimensions for the matrices
        var M: UInt32 = 2 // Rows of A/C
        var K: UInt32 = 3 // Cols of A / Rows of B
        var N: UInt32 = 2 // Cols of B/C
        
        // Step 1: Initialize tensors using UMA zero-copy memory
        print("\nAllocating shared memory tensors...")
        let tensorA = Tensor(device: engine.device, rows: Int(M), cols: Int(K))
        let tensorB = Tensor(device: engine.device, rows: Int(K), cols: Int(N))
        let tensorC = Tensor(device: engine.device, rows: Int(M), cols: Int(N))
        
        // Seed tensor A with deterministic values
        let ptrA = tensorA.pointer()
        ptrA[0] = 1; ptrA[1] = 2; ptrA[2] = 3
        ptrA[3] = 4; ptrA[4] = 5; ptrA[5] = 6
        
        // Seed tensor B with deterministic values
        let ptrB = tensorB.pointer()
        ptrB[0] = 7; ptrB[1] = 8
        ptrB[2] = 9; ptrB[3] = 10
        ptrB[4] = 11; ptrB[5] = 12
        
        print("Matrix A (\(M)x\(K)):")
        tensorA.printMatrix()
        print("Matrix B (\(K)x\(N)):")
        tensorB.printMatrix()
        
        // Step 4: Dispatch and Synchronization
        print("Encoding compute commands...")
        guard let commandBuffer = engine.commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder(),
              let pipelineState = engine.pipelineState else {
            fatalError("Failed to create command buffer or encoder")
        }
        
        computeEncoder.setComputePipelineState(pipelineState)
        
        // Set buffers so the GPU can access them directly without data copies
        computeEncoder.setBuffer(tensorA.buffer, offset: 0, index: 0)
        computeEncoder.setBuffer(tensorB.buffer, offset: 0, index: 1)
        computeEncoder.setBuffer(tensorC.buffer, offset: 0, index: 2)
        
        // Set the dimension variables
        computeEncoder.setBytes(&M, length: MemoryLayout<UInt32>.size, index: 3)
        computeEncoder.setBytes(&K, length: MemoryLayout<UInt32>.size, index: 4)
        computeEncoder.setBytes(&N, length: MemoryLayout<UInt32>.size, index: 5)
        
        // Calculate the thread groupings
        // A 2D grid covering the entire output matrix (rows=M, cols=N)
        let threadsPerGrid = MTLSize(width: Int(N), height: Int(M), depth: 1)
        
        // Use optimal threadgroup size bounded by hardware constraints and dimensions
        let w = pipelineState.threadExecutionWidth
        let h = pipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerThreadgroup = MTLSize(width: min(w, Int(N)), height: min(h, Int(M)), depth: 1)
        
        computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        // Synchronize: Submit instructions to the GPU and wait for completion
        print("\nDispatching computation threads to Apple GPU...")
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted() // Blocks the CPU until the GPU math finishes
        
        print("\nGPU execution completed. Printing final shared memory results directly from CPU pointer:")
        print("Matrix C (\(M)x\(N)):")
        tensorC.printMatrix()
        
        print("Expected C:\n[58.0000, 64.0000]\n[139.0000, 154.0000]\n---")
    }
}
