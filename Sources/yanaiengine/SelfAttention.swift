import Metal
import Foundation

/// Scaled Dot-Product Self-Attention Layer
/// Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
///
/// Contains three internal LinearLayers for Q, K, V projections.
public class SelfAttention {
    private let engine: MetalEngine
    private let seqLen: Int
    private let dModel: Int
    
    // Projection layers (no activation — raw linear transforms)
    public let queryProj: LinearLayer
    public let keyProj: LinearLayer
    public let valueProj: LinearLayer
    
    // Scratch tensors for intermediate results
    private let keyTransposed: Tensor     // K^T: [dModel x seqLen]
    public let scores: Tensor             // QK^T: [seqLen x seqLen]
    public let output: Tensor             // Final attention output: [seqLen x dModel]
    
    private let useCausalMask: Bool
    
    /// Initialize a Self-Attention layer.
    /// - Parameters:
    ///   - engine: The Metal engine.
    ///   - seqLen: Sequence length (number of tokens).
    ///   - dModel: Model dimension (embedding size per token).
    ///   - useCausalMask: Whether to apply causal (autoregressive) masking.
    public init(engine: MetalEngine, seqLen: Int, dModel: Int, useCausalMask: Bool = true) {
        self.engine = engine
        self.seqLen = seqLen
        self.dModel = dModel
        self.useCausalMask = useCausalMask
        
        // Q, K, V projections: [seqLen x dModel] -> [seqLen x dModel]
        self.queryProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: dModel, batchSize: seqLen, useReLU: false)
        self.keyProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: dModel, batchSize: seqLen, useReLU: false)
        self.valueProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: dModel, batchSize: seqLen, useReLU: false)
        
        // Scratch space
        self.keyTransposed = Tensor(device: engine.device, rows: dModel, cols: seqLen)
        self.scores = Tensor(device: engine.device, rows: seqLen, cols: seqLen)
        self.output = Tensor(device: engine.device, rows: seqLen, cols: dModel)
    }
    
    /// Forward pass: computes full Scaled Dot-Product Attention.
    /// Input: [seqLen x dModel] — the token embeddings.
    /// Output: [seqLen x dModel] — the attended representations.
    public func forward(input: Tensor) {
        // ---- Phase 1: Project into Q, K, V ----
        queryProj.forward(input: input)  // Q = input * Wq + bq
        keyProj.forward(input: input)    // K = input * Wk + bk
        valueProj.forward(input: input)  // V = input * Wv + bv
        
        // ---- Phase 2: Attention math in a single command buffer ----
        guard let commandBuffer = engine.commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create command buffer")
        }
        
        // 2a. Transpose K: K^T [dModel x seqLen]
        let transPSO = engine.getPipelineState(name: "transpose_kernel")
        encoder.setComputePipelineState(transPSO)
        encoder.setBuffer(keyProj.output.buffer, offset: 0, index: 0)
        encoder.setBuffer(keyTransposed.buffer, offset: 0, index: 1)
        var kRows = UInt32(seqLen)
        var kCols = UInt32(dModel)
        encoder.setBytes(&kRows, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&kCols, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: dModel, height: seqLen, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1)
        )
        
        // 2b. Scores = Q * K^T: [seqLen x dModel] * [dModel x seqLen] = [seqLen x seqLen]
        let gemmPSO = engine.getPipelineState(name: "gemm_kernel")
        encoder.setComputePipelineState(gemmPSO)
        encoder.setBuffer(queryProj.output.buffer, offset: 0, index: 0)
        encoder.setBuffer(keyTransposed.buffer, offset: 0, index: 1)
        encoder.setBuffer(scores.buffer, offset: 0, index: 2)
        var M = UInt32(seqLen)
        var K = UInt32(dModel)
        var N = UInt32(seqLen)
        encoder.setBytes(&M, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&K, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&N, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.dispatchThreads(
            MTLSize(width: seqLen, height: seqLen, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1)
        )
        
        // 2c. Scale: scores = scores / sqrt(d_k)
        let scalePSO = engine.getPipelineState(name: "scale_kernel")
        encoder.setComputePipelineState(scalePSO)
        encoder.setBuffer(scores.buffer, offset: 0, index: 0)
        var scaleFactor = 1.0 / sqrt(Float(dModel))
        encoder.setBytes(&scaleFactor, length: MemoryLayout<Float>.size, index: 1)
        var scaleLen = UInt32(seqLen * seqLen)
        encoder.setBytes(&scaleLen, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: seqLen * seqLen, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(scalePSO.maxTotalThreadsPerThreadgroup, seqLen * seqLen), height: 1, depth: 1)
        )
        
        // 2d. Causal Mask (optional): set upper triangle to -inf
        if useCausalMask {
            let maskPSO = engine.getPipelineState(name: "causal_mask_kernel")
            encoder.setComputePipelineState(maskPSO)
            encoder.setBuffer(scores.buffer, offset: 0, index: 0)
            var dim = UInt32(seqLen)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(
                MTLSize(width: seqLen, height: seqLen, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1)
            )
        }
        
        // 2e. Softmax (row-wise): normalize each row of scores
        let softmaxPSO = engine.getPipelineState(name: "softmax_kernel")
        encoder.setComputePipelineState(softmaxPSO)
        encoder.setBuffer(scores.buffer, offset: 0, index: 0)
        var sRows = UInt32(seqLen)
        var sCols = UInt32(seqLen)
        encoder.setBytes(&sRows, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.setBytes(&sCols, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: seqLen, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(softmaxPSO.maxTotalThreadsPerThreadgroup, seqLen), height: 1, depth: 1)
        )
        
        // 2f. Output = scores * V: [seqLen x seqLen] * [seqLen x dModel] = [seqLen x dModel]
        encoder.setComputePipelineState(gemmPSO)
        encoder.setBuffer(scores.buffer, offset: 0, index: 0)
        encoder.setBuffer(valueProj.output.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        var M2 = UInt32(seqLen)
        var K2 = UInt32(seqLen)
        var N2 = UInt32(dModel)
        encoder.setBytes(&M2, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&K2, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&N2, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.dispatchThreads(
            MTLSize(width: dModel, height: seqLen, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1)
        )
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}
