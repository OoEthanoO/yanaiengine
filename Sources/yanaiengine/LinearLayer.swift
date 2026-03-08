import Metal
import Foundation

/// A Linear (Dense) Layer that encapsulates Y = ReLU(X * W + b)
/// Execution is chained on the GPU within a single command buffer.
public class LinearLayer {
    public let weights: Tensor
    public let bias: Tensor
    public let output: Tensor
    
    private let engine: MetalEngine
    
    /// Initializes a Linear Layer with the given dimensions.
    ///
    /// - Parameters:
    ///   - engine: The Metal engine to use for computation.
    ///   - inputDim: The size of the input vector (cols of input / rows of weights).
    ///   - outputDim: The size of the output vector (cols of weights / cols of bias).
    ///   - batchSize: The number of rows in the input matrix.
    public init(engine: MetalEngine, inputDim: Int, outputDim: Int, batchSize: Int) {
        self.engine = engine
        
        // Weights: [inputDim x outputDim]
        self.weights = Tensor(device: engine.device, rows: inputDim, cols: outputDim)
        
        // Bias: [1 x outputDim] (broadcast across batchSize)
        self.bias = Tensor(device: engine.device, rows: 1, cols: outputDim)
        
        // Output: [batchSize x outputDim]
        self.output = Tensor(device: engine.device, rows: batchSize, cols: outputDim)
        
        // Initialize weights/bias with some values
        weights.fillRandom()
        bias.fill(with: 0.1)
    }
    
    /// Performs the forward pass: ReLU(input * weights + bias)
    /// All operations are encoded into a single command buffer for GPU efficiency.
    public func forward(input: Tensor) {
        guard let commandBuffer = engine.commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create command buffer or encoder")
        }
        
        // Internal variables for dimensions
        var M = UInt32(input.rows)
        var K = UInt32(input.cols) // Should match weights.rows
        var N = UInt32(weights.cols)
        
        // 1. Dispatch GEMM: output = input * weights
        let gemmPSO = engine.getPipelineState(name: "gemm_kernel")
        encoder.setComputePipelineState(gemmPSO)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(weights.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        encoder.setBytes(&M, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&K, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&N, length: MemoryLayout<UInt32>.size, index: 5)
        
        let gemmThreadsPerGrid = MTLSize(width: Int(N), height: Int(M), depth: 1)
        let gemmThreadsPerTG = MTLSize(width: min(gemmPSO.threadExecutionWidth, Int(N)), 
                                      height: min(gemmPSO.maxTotalThreadsPerThreadgroup / gemmPSO.threadExecutionWidth, Int(M)), 
                                      depth: 1)
        encoder.dispatchThreads(gemmThreadsPerGrid, threadsPerThreadgroup: gemmThreadsPerTG)
        
        // 2. Dispatch Bias Add: output = output + bias
        // We use the same M and N for the grid
        let biasPSO = engine.getPipelineState(name: "bias_add_kernel")
        encoder.setComputePipelineState(biasPSO)
        encoder.setBuffer(output.buffer, offset: 0, index: 0)
        encoder.setBuffer(bias.buffer, offset: 0, index: 1)
        encoder.setBytes(&M, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&N, length: MemoryLayout<UInt32>.size, index: 3)
        
        encoder.dispatchThreads(gemmThreadsPerGrid, threadsPerThreadgroup: gemmThreadsPerTG)
        
        // 3. Dispatch ReLU: output = max(0, output)
        let reluPSO = engine.getPipelineState(name: "relu_kernel")
        encoder.setComputePipelineState(reluPSO)
        encoder.setBuffer(output.buffer, offset: 0, index: 0)
        var totalLength = UInt32(output.rows * output.cols)
        encoder.setBytes(&totalLength, length: MemoryLayout<UInt32>.size, index: 1)
        
        let reluThreadsPerGrid = MTLSize(width: Int(totalLength), height: 1, depth: 1)
        let reluThreadsPerTG = MTLSize(width: min(reluPSO.maxTotalThreadsPerThreadgroup, Int(totalLength)), height: 1, depth: 1)
        encoder.dispatchThreads(reluThreadsPerGrid, threadsPerThreadgroup: reluThreadsPerTG)
        
        encoder.endEncoding()
        
        // Submit everything to the GPU
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}
