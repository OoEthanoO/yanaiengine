import Foundation
import Metal

/// A Sparse Mixture of Experts (MoE) Layer.
/// Replaces the standard Feed-Forward Network (FFN) with a collection of expert FFNs.
public class MoELayer {
    public let router: MoERouter
    public let experts: [ExpertFFN]
    private let engine: MetalEngine
    private let numExperts: Int
    private let topK: Int
    
    public init(engine: MetalEngine, 
                dModel: Int, 
                ffnDim: Int, 
                numExperts: Int, 
                topK: Int = 2) {
        self.engine = engine
        self.numExperts = numExperts
        self.topK = topK
        
        self.router = MoERouter(engine: engine, dModel: dModel, numExperts: numExperts, topK: topK)
        
        self.experts = (0..<numExperts).map { _ in
            ExpertFFN(engine: engine, dModel: dModel, ffnDim: ffnDim)
        }
    }
    
    /// Forward pass for the MoE layer.
    /// 1. Route: gate hidden state to find top-K experts for each token.
    /// 2. Sparse Execution: dispatch tokens to their assigned experts.
    /// 3. Weighted Sum: combine expert outputs back into the hidden state.
    public func forward(input: Tensor) -> Tensor {
        let totalTokens = input.rows
        
        // 1. Routing
        router.forward(input: input)
        
        // 2. Expert Dispatch (Sparse execution)
        let expertOutputs = Tensor(device: engine.device, rows: numExperts * totalTokens, cols: input.cols)
        
        // Note: In a fully optimized version, we would group tokens by expert
        // to minimize redundant FFN passes. For this polymorphic implementation,
        // we execute experts and then combine via the routing weights.
        for i in 0..<numExperts {
            // Placeholder: experts[i].forward(input) but write to expertOutputs range
            // expertOutputs[i * totalTokens : (i+1) * totalTokens] = experts[i].forward(input)
        }
        
        // 3. Recombination (Combine)
        let output = Tensor(device: engine.device, rows: totalTokens, cols: input.cols)
        dispatchMoECombine(expertOutputs: expertOutputs, output: output)
        
        return output
    }
    
    private func dispatchMoECombine(expertOutputs: Tensor, output: Tensor) {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        
        let pso = engine.getPipelineState(name: "moe_combine_kernel")
        enc.setComputePipelineState(pso)
        
        enc.setBuffer(router.selectedIndices.buffer, offset: 0, index: 0)
        enc.setBuffer(router.selectedWeights.buffer, offset: 0, index: 1)
        enc.setBuffer(expertOutputs.buffer, offset: 0, index: 2)
        enc.setBuffer(output.buffer, offset: 0, index: 3)
        
        var d = UInt32(output.cols)
        var k = UInt32(topK)
        var totalTokens = UInt32(output.rows)
        
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&totalTokens, length: MemoryLayout<UInt32>.size, index: 6)
        
        enc.dispatchThreads(MTLSize(width: Int(d), height: Int(totalTokens), depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
}

/// A single expert within the MoE layer, effectively a standard FFN block.
public class ExpertFFN {
    public let gateProj: LinearLayer
    public let upProj: LinearLayer
    public let downProj: LinearLayer
    private let engine: MetalEngine
    
    public init(engine: MetalEngine, dModel: Int, ffnDim: Int) {
        self.engine = engine
        self.gateProj = LinearLayer(engine: engine, rows: ffnDim, cols: dModel)
        self.upProj = LinearLayer(engine: engine, rows: ffnDim, cols: dModel)
        self.downProj = LinearLayer(engine: engine, rows: ffnDim, cols: dModel)
    }
    
    public func forward(input: Tensor) -> Tensor {
        gateProj.forward(input: input)
        upProj.forward(input: input)
        
        // Apply SiLU and element-wise multiply (Simplified for now)
        // ... (This would reuse the SiLU/Mul kernels implemented earlier)
        
        downProj.forward(input: gateProj.output)
        return downProj.output
    }
}
