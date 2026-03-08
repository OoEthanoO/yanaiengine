import Metal
import Foundation

/// A complete Transformer Block using Pre-Norm architecture.
///
/// Architecture:
///   x₁ = x + MHA(LayerNorm(x))         — Residual #1
///   x₂ = x₁ + FFN(LayerNorm(x₁))      — Residual #2
///
/// FFN = Linear(dModel → 4*dModel) → GELU → Linear(4*dModel → dModel)
public class TransformerBlock {
    private let engine: MetalEngine
    public let seqLen: Int
    public let dModel: Int
    
    // Sub-modules
    public let mha: MultiHeadAttention
    private let ffnUp: LinearLayer        // dModel → 4*dModel
    private let ffnDown: LinearLayer      // 4*dModel → dModel
    
    // LayerNorm parameters (learnable γ and β)
    private let lnGamma1: Tensor
    private let lnBeta1: Tensor
    private let lnGamma2: Tensor
    private let lnBeta2: Tensor
    
    // Scratch tensors
    private let lnOutput1: Tensor         // after first LayerNorm
    private let residual1: Tensor         // after first residual add
    private let lnOutput2: Tensor         // after second LayerNorm
    public let output: Tensor             // final output
    
    public init(engine: MetalEngine, seqLen: Int, dModel: Int, numHeads: Int) {
        self.engine = engine
        self.seqLen = seqLen
        self.dModel = dModel
        
        // Multi-Head Attention
        self.mha = MultiHeadAttention(engine: engine, seqLen: seqLen, dModel: dModel, numHeads: numHeads)
        
        // Feed-Forward Network (4x expansion, standard Transformer)
        let ffnDim = 4 * dModel
        self.ffnUp = LinearLayer(engine: engine, inputDim: dModel, outputDim: ffnDim, batchSize: seqLen, useReLU: false)
        self.ffnDown = LinearLayer(engine: engine, inputDim: ffnDim, outputDim: dModel, batchSize: seqLen, useReLU: false)
        
        // LayerNorm parameters: gamma=1, beta=0 (initialized)
        self.lnGamma1 = Tensor(device: engine.device, rows: 1, cols: dModel)
        self.lnBeta1 = Tensor(device: engine.device, rows: 1, cols: dModel)
        self.lnGamma2 = Tensor(device: engine.device, rows: 1, cols: dModel)
        self.lnBeta2 = Tensor(device: engine.device, rows: 1, cols: dModel)
        
        lnGamma1.fill(with: 1.0)
        lnBeta1.fill(with: 0.0)
        lnGamma2.fill(with: 1.0)
        lnBeta2.fill(with: 0.0)
        
        // Scratch buffers
        self.lnOutput1 = Tensor(device: engine.device, rows: seqLen, cols: dModel)
        self.residual1 = Tensor(device: engine.device, rows: seqLen, cols: dModel)
        self.lnOutput2 = Tensor(device: engine.device, rows: seqLen, cols: dModel)
        self.output = Tensor(device: engine.device, rows: seqLen, cols: dModel)
    }
    
    /// Forward pass through the full Transformer Block.
    /// Input: [seqLen x dModel], Output: [seqLen x dModel]
    public func forward(input: Tensor) {
        let totalLen = seqLen * dModel
        
        // ========================================
        // Sub-block 1: x₁ = x + MHA(LayerNorm(x))
        // ========================================
        
        // Copy input into lnOutput1 (LayerNorm operates in-place)
        memcpy(lnOutput1.buffer.contents(), input.buffer.contents(), totalLen * MemoryLayout<Float>.stride)
        
        // LayerNorm #1
        dispatchLayerNorm(data: lnOutput1, gamma: lnGamma1, beta: lnBeta1)
        
        // Multi-Head Attention on normalized input
        mha.forward(input: lnOutput1)
        
        // Residual #1: residual1 = input + mha.output
        dispatchAdd(a: input, b: mha.output, out: residual1, length: totalLen)
        
        // ========================================
        // Sub-block 2: x₂ = x₁ + FFN(LayerNorm(x₁))
        // ========================================
        
        // Copy residual1 into lnOutput2
        memcpy(lnOutput2.buffer.contents(), residual1.buffer.contents(), totalLen * MemoryLayout<Float>.stride)
        
        // LayerNorm #2
        dispatchLayerNorm(data: lnOutput2, gamma: lnGamma2, beta: lnBeta2)
        
        // Feed-Forward: Linear → GELU → Linear
        ffnUp.forward(input: lnOutput2)
        dispatchGELU(data: ffnUp.output, length: seqLen * (4 * dModel))
        ffnDown.forward(input: ffnUp.output)
        
        // Residual #2: output = residual1 + ffnDown.output
        dispatchAdd(a: residual1, b: ffnDown.output, out: output, length: totalLen)
    }
    
    // MARK: - GPU Dispatch Helpers
    
    private func dispatchLayerNorm(data: Tensor, gamma: Tensor, beta: Tensor) {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        
        let pso = engine.getPipelineState(name: "layernorm_kernel")
        enc.setComputePipelineState(pso)
        enc.setBuffer(data.buffer, offset: 0, index: 0)
        enc.setBuffer(gamma.buffer, offset: 0, index: 1)
        enc.setBuffer(beta.buffer, offset: 0, index: 2)
        var rows = UInt32(seqLen)
        var cols = UInt32(dModel)
        var eps: Float = 1e-5
        enc.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&cols, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&eps, length: MemoryLayout<Float>.size, index: 5)
        enc.dispatchThreads(MTLSize(width: seqLen, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, seqLen), height: 1, depth: 1))
        
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
    
    private func dispatchGELU(data: Tensor, length: Int) {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        
        let pso = engine.getPipelineState(name: "gelu_kernel")
        enc.setComputePipelineState(pso)
        enc.setBuffer(data.buffer, offset: 0, index: 0)
        var len = UInt32(length)
        enc.setBytes(&len, length: MemoryLayout<UInt32>.size, index: 1)
        enc.dispatchThreads(MTLSize(width: length, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, length), height: 1, depth: 1))
        
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
    
    private func dispatchAdd(a: Tensor, b: Tensor, out: Tensor, length: Int) {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        
        let pso = engine.getPipelineState(name: "elementwise_add_kernel")
        enc.setComputePipelineState(pso)
        enc.setBuffer(a.buffer, offset: 0, index: 0)
        enc.setBuffer(b.buffer, offset: 0, index: 1)
        enc.setBuffer(out.buffer, offset: 0, index: 2)
        var len = UInt32(length)
        enc.setBytes(&len, length: MemoryLayout<UInt32>.size, index: 3)
        enc.dispatchThreads(MTLSize(width: length, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, length), height: 1, depth: 1))
        
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
    
    // ====================================================
    // KV-Cached Single-Token Decode
    // ====================================================
    
    private lazy var decodeFfnUp = LinearLayer(engine: engine, inputDim: dModel, outputDim: 4 * dModel, batchSize: 1, useReLU: false)
    private lazy var decodeFfnDown = LinearLayer(engine: engine, inputDim: 4 * dModel, outputDim: dModel, batchSize: 1, useReLU: false)
    private lazy var decodeOutput = Tensor(device: engine.device, rows: 1, cols: dModel)
    private var decodeFfnWeightsCopied = false
    
    private func ensureDecodeFfnWeights() {
        if decodeFfnWeightsCopied { return }
        memcpy(decodeFfnUp.weights.pointer(), ffnUp.weights.pointer(),
               ffnUp.weights.rows * ffnUp.weights.cols * MemoryLayout<Float>.stride)
        memcpy(decodeFfnUp.bias.pointer(), ffnUp.bias.pointer(),
               ffnUp.bias.cols * MemoryLayout<Float>.stride)
        memcpy(decodeFfnDown.weights.pointer(), ffnDown.weights.pointer(),
               ffnDown.weights.rows * ffnDown.weights.cols * MemoryLayout<Float>.stride)
        memcpy(decodeFfnDown.bias.pointer(), ffnDown.bias.pointer(),
               ffnDown.bias.cols * MemoryLayout<Float>.stride)
        decodeFfnWeightsCopied = true
    }
    
    /// CPU-side LayerNorm for a single row [1 x dModel].
    private func layerNormCPU(_ data: UnsafeMutablePointer<Float>, gamma: Tensor, beta: Tensor) {
        let gPtr = gamma.pointer()
        let bPtr = beta.pointer()
        var sum: Float = 0
        for c in 0..<dModel { sum += data[c] }
        let mean = sum / Float(dModel)
        var varSum: Float = 0
        for c in 0..<dModel { let d = data[c] - mean; varSum += d * d }
        let invStd = 1.0 / sqrt(varSum / Float(dModel) + 1e-5)
        for c in 0..<dModel {
            data[c] = gPtr[c] * (data[c] - mean) * invStd + bPtr[c]
        }
    }
    
    /// Forward pass for a single new token using KV cache.
    /// Input: [1 x dModel]. Returns output [1 x dModel].
    public func forwardCached(input: Tensor, cache: KVCache) -> Tensor {
        ensureDecodeFfnWeights()
        let outPtr = decodeOutput.pointer()
        let inPtr = input.pointer()
        
        // Sub-block 1: x₁ = x + MHA(LayerNorm(x))
        let ln1 = Tensor(device: engine.device, rows: 1, cols: dModel)
        memcpy(ln1.pointer(), inPtr, dModel * MemoryLayout<Float>.stride)
        layerNormCPU(ln1.pointer(), gamma: lnGamma1, beta: lnBeta1)
        
        let mhaOut = mha.forwardCached(input: ln1, cache: cache)
        
        // Residual 1
        let res1 = Tensor(device: engine.device, rows: 1, cols: dModel)
        let res1Ptr = res1.pointer()
        let mhaPtr = mhaOut.pointer()
        for d in 0..<dModel { res1Ptr[d] = inPtr[d] + mhaPtr[d] }
        
        // Sub-block 2: x₂ = x₁ + FFN(LayerNorm(x₁))
        let ln2 = Tensor(device: engine.device, rows: 1, cols: dModel)
        memcpy(ln2.pointer(), res1Ptr, dModel * MemoryLayout<Float>.stride)
        layerNormCPU(ln2.pointer(), gamma: lnGamma2, beta: lnBeta2)
        
        // FFN: Linear → GELU → Linear
        decodeFfnUp.forward(input: ln2)
        // CPU GELU on the 4*dModel values
        let ffnPtr = decodeFfnUp.output.pointer()
        let ffnDim = 4 * dModel
        for i in 0..<ffnDim {
            let x = ffnPtr[i]
            let cdf = 0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
            ffnPtr[i] = x * cdf
        }
        decodeFfnDown.forward(input: decodeFfnUp.output)
        
        // Residual 2
        let ffnOutPtr = decodeFfnDown.output.pointer()
        for d in 0..<dModel { outPtr[d] = res1Ptr[d] + ffnOutPtr[d] }
        
        return decodeOutput
    }
}
