import Foundation
import Metal

/// The Gating Router for Mixture of Experts (MoE).
/// Projects the hidden state to expert logits and selects Top-K experts.
public class Router {
    public let weights: Tensor // [numExperts x dModel]
    private let engine: MetalEngine
    private let dModel: Int
    private let numExperts: Int
    private let topK: Int
    
    public var gateLogits: Tensor // [totalTokens x numExperts]
    public var selectedIndices: Tensor // [totalTokens x topK] - Int32
    public var selectedWeights: Tensor // [totalTokens x topK] - Float
    
    public init(engine: MetalEngine, dModel: Int, numExperts: Int, topK: Int = 2) {
        self.engine = engine
        self.dModel = dModel
        self.numExperts = numExperts
        self.topK = topK
        
        self.weights = Tensor(device: engine.device, rows: numExperts, cols: dModel)
        
        // Output buffers (will be resized per batch in forward pass)
        self.gateLogits = Tensor(device: engine.device, rows: 1, cols: numExperts)
        self.selectedIndices = Tensor(device: engine.device, rows: 1, cols: topK)
        self.selectedWeights = Tensor(device: engine.device, rows: 1, cols: topK)
    }
    
    /// Projects input hidden states to expert weights and finds Top-K.
    /// - Parameter input: [totalTokens x dModel]
    public func forward(input: Tensor) {
        let totalTokens = input.rows
        
        // 1. Resize output buffers if needed
        if gateLogits.rows != totalTokens {
            gateLogits = Tensor(device: engine.device, rows: totalTokens, cols: numExperts)
            selectedIndices = Tensor(device: engine.device, rows: totalTokens, cols: topK)
            selectedWeights = Tensor(device: engine.device, rows: totalTokens, cols: topK)
        }
        
        // 2. Linear Projection: gateLogits = input @ weights.T
        // We use the standard GEMM kernel for this.
        dispatchRouterGEMM(input: input)
        
        // 3. Fused Softmax + Top-K
        dispatchSoftmaxTopK()
    }
    
    private func dispatchRouterGEMM(input: Tensor) {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        
        let pso = engine.getPipelineState(name: "gemm_kernel")
        enc.setComputePipelineState(pso)
        
        enc.setBuffer(input.buffer, offset: 0, index: 0)
        enc.setBuffer(weights.buffer, offset: 0, index: 1)
        enc.setBuffer(gateLogits.buffer, offset: 0, index: 2)
        
        var M = UInt32(input.rows)
        var N = UInt32(numExperts)
        var K = UInt32(dModel)
        
        enc.setBytes(&M, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&N, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&K, length: MemoryLayout<UInt32>.size, index: 5)
        
        // Standard tiling
        let grid = MTLSize(width: (Int(N) + 15) / 16, height: (Int(M) + 15) / 16, depth: 1)
        let threadgroup = MTLSize(width: 16, height: 16, depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
        
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
    
    private func dispatchSoftmaxTopK() {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        
        let pso = engine.getPipelineState(name: "router_softmax_topk_kernel")
        enc.setComputePipelineState(pso)
        
        enc.setBuffer(gateLogits.buffer, offset: 0, index: 0)
        enc.setBuffer(selectedIndices.buffer, offset: 0, index: 1)
        enc.setBuffer(selectedWeights.buffer, offset: 0, index: 2)
        
        var nExp = UInt32(numExperts)
        var k = UInt32(topK)
        
        enc.setBytes(&nExp, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 4)
        
        let totalTokens = gateLogits.rows
        enc.dispatchThreads(MTLSize(width: totalTokens, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, totalTokens), height: 1, depth: 1))
        
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
}
