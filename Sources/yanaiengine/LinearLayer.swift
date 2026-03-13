import Metal
import Foundation

/// A Linear (Dense) Layer that encapsulates Y = ReLU(X * W + b)
/// Execution is chained on the GPU within a single command buffer.
public class LinearLayer {
    public let weights: Tensor
    public let bias: Tensor
    public let output: Tensor
    
    // Gradient and intermediate storage for training
    public let weightGradients: Tensor
    public let biasGradients: Tensor
    public let inputGradients: Tensor
    private var lastInput: Tensor?
    private let transposedInput: Tensor
    private let transposedWeights: Tensor
    
    private let engine: MetalEngine
    private let useReLU: Bool
    
    /// Initializes a Linear Layer with the given dimensions.
    ///
    /// - Parameters:
    ///   - engine: The Metal engine to use for computation.
    ///   - inputDim: The size of the input vector (cols of input / rows of weights).
    ///   - outputDim: The size of the output vector (cols of weights / cols of bias).
    ///   - batchSize: The number of rows in the input matrix.
    public init(engine: MetalEngine, inputDim: Int, outputDim: Int, batchSize: Int, existingBuffer: MTLBuffer? = nil, useReLU: Bool = true) {
        self.engine = engine
        self.useReLU = useReLU
        
        // Weights: [inputDim x outputDim]
        if let buffer = existingBuffer {
             self.weights = Tensor(device: engine.device, rows: inputDim, cols: outputDim, existingBuffer: buffer)
        } else {
             self.weights = Tensor(device: engine.device, rows: inputDim, cols: outputDim)
        }
        
        // Bias: [1 x outputDim] (broadcast across batchSize)
        self.bias = Tensor(device: engine.device, rows: 1, cols: outputDim)
        
        // Output: [batchSize x outputDim]
        self.output = Tensor(device: engine.device, rows: batchSize, cols: outputDim)
        
        // Gradient buffers match parameter dimensions
        self.weightGradients = Tensor(device: engine.device, rows: inputDim, cols: outputDim)
        self.biasGradients = Tensor(device: engine.device, rows: 1, cols: outputDim)
        self.inputGradients = Tensor(device: engine.device, rows: batchSize, cols: inputDim)
        
        // Scratch buffers for matrix transpose
        // X^T is [inputDim x batchSize]
        self.transposedInput = Tensor(device: engine.device, rows: inputDim, cols: batchSize)
        // W^T is [outputDim x inputDim]
        self.transposedWeights = Tensor(device: engine.device, rows: outputDim, cols: inputDim)
        
        // Initialize weights/bias with some values
        if existingBuffer == nil {
            weights.fillRandom()
        }
        bias.fill(with: 0.1)
    }
    
    /// Convenience initializer using rows/cols naming
    public convenience init(engine: MetalEngine, rows: Int, cols: Int, batchSize: Int = 1, useReLU: Bool = true) {
        self.init(engine: engine, inputDim: rows, outputDim: cols, batchSize: batchSize, useReLU: useReLU)
    }
    
    /// Performs the forward pass: ReLU(input * weights + bias)
    /// All operations are encoded into a single command buffer for GPU efficiency.
    public func forward(input: Tensor) {
        self.lastInput = input // Save for backward pass
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
        if useReLU {
            let reluPSO = engine.getPipelineState(name: "relu_kernel")
            encoder.setComputePipelineState(reluPSO)
            encoder.setBuffer(output.buffer, offset: 0, index: 0)
            var totalLength = UInt32(output.rows * output.cols)
            encoder.setBytes(&totalLength, length: MemoryLayout<UInt32>.size, index: 1)
            
            let reluThreadsPerGrid = MTLSize(width: Int(totalLength), height: 1, depth: 1)
            let reluThreadsPerTG = MTLSize(width: min(reluPSO.maxTotalThreadsPerThreadgroup, Int(totalLength)), height: 1, depth: 1)
            encoder.dispatchThreads(reluThreadsPerGrid, threadsPerThreadgroup: reluThreadsPerTG)
        }
        
        encoder.endEncoding()
        
        // Submit everything to the GPU
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    /// Computes gradients (dW, db, dX) but DOES NOT update weights/biases.
    /// Used in distributed training where gradients must be averaged before updating.
    @discardableResult
    public func computeGradients(upstreamGradient: Tensor) -> Tensor {
        guard let input = lastInput else {
            fatalError("forward() must be called before backward()")
        }
        
        guard let commandBuffer = engine.commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create command buffer or encoder")
        }
        
        // 0. Apply ReLU Derivative
        if useReLU {
            let reluDerivPSO = engine.getPipelineState(name: "relu_derivative_kernel")
            encoder.setComputePipelineState(reluDerivPSO)
            encoder.setBuffer(upstreamGradient.buffer, offset: 0, index: 0)
            encoder.setBuffer(output.buffer, offset: 0, index: 1)
            var totalLength = UInt32(output.rows * output.cols)
            encoder.setBytes(&totalLength, length: MemoryLayout<UInt32>.size, index: 2)
            
            let reluThreadsPerGrid = MTLSize(width: Int(totalLength), height: 1, depth: 1)
            let reluThreadsPerTG = MTLSize(width: min(reluDerivPSO.maxTotalThreadsPerThreadgroup, Int(totalLength)), height: 1, depth: 1)
            encoder.dispatchThreads(reluThreadsPerGrid, threadsPerThreadgroup: reluThreadsPerTG)
        }
        
        // 1. Transpose Input: X -> X^T
        let transPSO = engine.getPipelineState(name: "transpose_kernel")
        encoder.setComputePipelineState(transPSO)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(transposedInput.buffer, offset: 0, index: 1)
        var x_rows = UInt32(input.rows)
        var x_cols = UInt32(input.cols)
        encoder.setBytes(&x_rows, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&x_cols, length: MemoryLayout<UInt32>.size, index: 3)
        
        let transXThreadsPerGrid = MTLSize(width: Int(x_cols), height: Int(x_rows), depth: 1)
        encoder.dispatchThreads(transXThreadsPerGrid, threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        
        // 2. Calculate Weight Gradient: dW = X^T * dY
        let gemmPSO = engine.getPipelineState(name: "gemm_kernel")
        encoder.setComputePipelineState(gemmPSO)
        encoder.setBuffer(transposedInput.buffer, offset: 0, index: 0)
        encoder.setBuffer(upstreamGradient.buffer, offset: 0, index: 1)
        encoder.setBuffer(weightGradients.buffer, offset: 0, index: 2)
        var M_dW = UInt32(transposedInput.rows)
        var K_dW = UInt32(transposedInput.cols)
        var N_dW = UInt32(upstreamGradient.cols)
        encoder.setBytes(&M_dW, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&K_dW, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&N_dW, length: MemoryLayout<UInt32>.size, index: 5)
        
        encoder.dispatchThreads(MTLSize(width: Int(N_dW), height: Int(M_dW), depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        
        // 3. Calculate Bias Gradient: db = sum(dY) over batch
        let sumPSO = engine.getPipelineState(name: "sum_rows_kernel")
        encoder.setComputePipelineState(sumPSO)
        encoder.setBuffer(upstreamGradient.buffer, offset: 0, index: 0)
        encoder.setBuffer(biasGradients.buffer, offset: 0, index: 1)
        var s_rows = UInt32(upstreamGradient.rows)
        var s_cols = UInt32(upstreamGradient.cols)
        encoder.setBytes(&s_rows, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&s_cols, length: MemoryLayout<UInt32>.size, index: 3)
        
        let sumThreadsPerGrid = MTLSize(width: Int(s_cols), height: 1, depth: 1)
        encoder.dispatchThreads(sumThreadsPerGrid, threadsPerThreadgroup: MTLSize(width: min(sumPSO.maxTotalThreadsPerThreadgroup, Int(s_cols)), height: 1, depth: 1))
        
        // 4. Calculate Input Gradient: dX = dY * W^T
        // A. Transpose Weights: W -> W^T [outputDim x inputDim]
        encoder.setComputePipelineState(transPSO)
        encoder.setBuffer(weights.buffer, offset: 0, index: 0)
        encoder.setBuffer(transposedWeights.buffer, offset: 0, index: 1)
        var w_rows = UInt32(weights.rows)
        var w_cols = UInt32(weights.cols)
        encoder.setBytes(&w_rows, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&w_cols, length: MemoryLayout<UInt32>.size, index: 3)
        
        let transWThreadsPerGrid = MTLSize(width: Int(w_cols), height: Int(w_rows), depth: 1)
        encoder.dispatchThreads(transWThreadsPerGrid, threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        
        // B. dX = dY * W^T
        encoder.setComputePipelineState(gemmPSO)
        encoder.setBuffer(upstreamGradient.buffer, offset: 0, index: 0)
        encoder.setBuffer(transposedWeights.buffer, offset: 0, index: 1)
        encoder.setBuffer(inputGradients.buffer, offset: 0, index: 2)
        var M_dX = UInt32(upstreamGradient.rows)
        var K_dX = UInt32(upstreamGradient.cols)
        var N_dX = UInt32(transposedWeights.cols)
        encoder.setBytes(&M_dX, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&K_dX, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&N_dX, length: MemoryLayout<UInt32>.size, index: 5)
        
        encoder.dispatchThreads(MTLSize(width: Int(N_dX), height: Int(M_dX), depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return inputGradients
    }
    
    /// Applies SGD updates to weights and biases.
    public func applyUpdates(learningRate: Float) {
        guard let commandBuffer = engine.commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create command buffer or encoder")
        }
        
        let updatePSO = engine.getPipelineState(name: "sgd_update_kernel")
        encoder.setComputePipelineState(updatePSO)
        encoder.setBuffer(weights.buffer, offset: 0, index: 0)
        encoder.setBuffer(weightGradients.buffer, offset: 0, index: 1)
        var lr = learningRate
        encoder.setBytes(&lr, length: MemoryLayout<Float>.size, index: 2)
        var wLength = UInt32(weights.rows * weights.cols)
        encoder.setBytes(&wLength, length: MemoryLayout<UInt32>.size, index: 3)
        
        encoder.dispatchThreads(MTLSize(width: Int(wLength), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        
        // Update Biases: b = b - lr * db
        encoder.setComputePipelineState(updatePSO)
        encoder.setBuffer(bias.buffer, offset: 0, index: 0)
        encoder.setBuffer(biasGradients.buffer, offset: 0, index: 1)
        encoder.setBytes(&lr, length: MemoryLayout<Float>.size, index: 2)
        var bLength = UInt32(bias.rows * bias.cols)
        encoder.setBytes(&bLength, length: MemoryLayout<UInt32>.size, index: 3)
        
        encoder.dispatchThreads(MTLSize(width: Int(bLength), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(updatePSO.maxTotalThreadsPerThreadgroup, Int(bLength)), height: 1, depth: 1))
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    /// Compatibility method for existing training loops.
    @discardableResult
    public func backward(upstreamGradient: Tensor, learningRate: Float) -> Tensor {
        let dX = computeGradients(upstreamGradient: upstreamGradient)
        applyUpdates(learningRate: learningRate)
        return dX
    }
}
