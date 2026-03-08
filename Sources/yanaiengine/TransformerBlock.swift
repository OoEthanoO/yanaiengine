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
}
