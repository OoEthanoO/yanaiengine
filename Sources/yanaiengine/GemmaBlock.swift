import Metal
import Foundation

/// Google Gemma 2 Transformer Block with:
/// - RMSNorm
/// - GeGLU Activation (uses GELU instead of SiLU)
/// - Logit Soft-Capping (Attention and Output)
/// - Alternating Sliding Window Attention (SWA)
public class GemmaBlock {
    private let engine: MetalEngine
    public let seqLen: Int
    public let dModel: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let dHead: Int
    private let ffnDim: Int
    
    // Configurable parameters
    public var logitCap: Float = 50.0
    public var windowSize: Int = 0 // 0 means global attention
    
    // Projections
    private let queryProj: LinearLayer
    private let keyProj: LinearLayer
    private let valueProj: LinearLayer
    private let outputProj: LinearLayer
    
    // GeGLU FFN: gate_proj, up_proj, down_proj
    private let gateProj: LinearLayer
    private let upProj: LinearLayer
    private let downProj: LinearLayer
    
    // RMSNorm weights
    private let rmsGamma1: Tensor
    private let rmsGamma2: Tensor
    
    // Scratch tensors
    private let rmsOut1: Tensor
    private let residual1: Tensor
    private let rmsOut2: Tensor
    private let gateOut: Tensor
    public let output: Tensor
    private let concatOutput: Tensor
    
    public init(engine: MetalEngine, seqLen: Int, dModel: Int, numHeads: Int, numKVHeads: Int, ffnDim: Int) {
        self.engine = engine
        self.seqLen = seqLen
        self.dModel = dModel
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.dHead = dModel / numHeads
        self.ffnDim = ffnDim
        
        // Gemma 2 projections
        self.queryProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: numHeads * dHead, batchSize: seqLen, useReLU: false)
        self.keyProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: numKVHeads * dHead, batchSize: seqLen, useReLU: false)
        self.valueProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: numKVHeads * dHead, batchSize: seqLen, useReLU: false)
        self.outputProj = LinearLayer(engine: engine, inputDim: numHeads * dHead, outputDim: dModel, batchSize: seqLen, useReLU: false)
        
        // GeGLU FFN
        self.gateProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: ffnDim, batchSize: seqLen, useReLU: false)
        self.upProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: ffnDim, batchSize: seqLen, useReLU: false)
        self.downProj = LinearLayer(engine: engine, inputDim: ffnDim, outputDim: dModel, batchSize: seqLen, useReLU: false)
        
        // RMSNorm
        self.rmsGamma1 = Tensor(device: engine.device, rows: 1, cols: dModel)
        self.rmsGamma2 = Tensor(device: engine.device, rows: 1, cols: dModel)
        rmsGamma1.fill(with: 1.0)
        rmsGamma2.fill(with: 1.0)
        
        // Scratch
        self.rmsOut1 = Tensor(device: engine.device, rows: seqLen, cols: dModel)
        self.residual1 = Tensor(device: engine.device, rows: seqLen, cols: dModel)
        self.rmsOut2 = Tensor(device: engine.device, rows: seqLen, cols: dModel)
        self.gateOut = Tensor(device: engine.device, rows: seqLen, cols: ffnDim)
        self.output = Tensor(device: engine.device, rows: seqLen, cols: dModel)
        self.concatOutput = Tensor(device: engine.device, rows: seqLen, cols: dModel)
    }
    
    public func forward(input: Tensor) {
        let totalLen = seqLen * dModel
        
        // 1. RMSNorm
        memcpy(rmsOut1.buffer.contents(), input.buffer.contents(), totalLen * MemoryLayout<Float>.stride)
        dispatchRMSNorm(data: rmsOut1, gamma: rmsGamma1)
        
        // 2. Attention Projections
        queryProj.forward(input: rmsOut1)
        keyProj.forward(input: rmsOut1)
        valueProj.forward(input: rmsOut1)
        
        // 3. Fused Attention with Soft-Capping and SWA
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        
        let fusedPSO = engine.getPipelineState(name: "fused_attention_kernel")
        enc.setComputePipelineState(fusedPSO)
        enc.setBuffer(queryProj.output.buffer, offset: 0, index: 0)
        enc.setBuffer(keyProj.output.buffer, offset: 0, index: 1)
        enc.setBuffer(valueProj.output.buffer, offset: 0, index: 2)
        enc.setBuffer(concatOutput.buffer, offset: 0, index: 3)
        
        var sl = UInt32(seqLen)
        var dh = UInt32(dHead)
        var sc = 1.0 / sqrt(Float(dHead))
        var cm = true
        var nh = UInt32(numHeads)
        var nkv = UInt32(numKVHeads)
        var cap = logitCap
        var ws = Int32(windowSize)
        
        enc.setBytes(&sl, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&dh, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&sc, length: MemoryLayout<Float>.size, index: 6)
        enc.setBytes(&cm, length: MemoryLayout<Bool>.size, index: 7)
        enc.setBytes(&nh, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&nkv, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&cap, length: MemoryLayout<Float>.size, index: 10)
        enc.setBytes(&ws, length: MemoryLayout<Int32>.size, index: 11)
        
        enc.dispatchThreads(MTLSize(width: seqLen, height: 1, depth: numHeads),
                            threadsPerThreadgroup: MTLSize(width: min(fusedPSO.maxTotalThreadsPerThreadgroup, seqLen), height: 1, depth: 1))
        
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        
        outputProj.forward(input: concatOutput)
        dispatchAdd(a: input, b: outputProj.output, out: residual1, length: totalLen)
        
        // 4. GeGLU FFN: down_proj(gelu(gate_proj(x)) * up_proj(x))
        memcpy(rmsOut2.buffer.contents(), residual1.buffer.contents(), totalLen * MemoryLayout<Float>.stride)
        dispatchRMSNorm(data: rmsOut2, gamma: rmsGamma2)
        
        gateProj.forward(input: rmsOut2)
        upProj.forward(input: rmsOut2)
        
        // Apply GELU to gate
        dispatchGELU(data: gateProj.output, length: seqLen * ffnDim)
        
        // Element-wise multiply: gate * up
        let gPtr = gateProj.output.pointer()
        let uPtr = upProj.output.pointer()
        for i in 0..<(seqLen * ffnDim) {
            gPtr[i] *= uPtr[i]
        }
        
        downProj.forward(input: gateProj.output)
        dispatchAdd(a: residual1, b: downProj.output, out: output, length: totalLen)
    }
    
    // GPU Helpers
    private func dispatchRMSNorm(data: Tensor, gamma: Tensor) {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        let pso = engine.getPipelineState(name: "rmsnorm_kernel")
        enc.setComputePipelineState(pso)
        enc.setBuffer(data.buffer, offset: 0, index: 0)
        enc.setBuffer(gamma.buffer, offset: 0, index: 1)
        var r = UInt32(seqLen); var c = UInt32(dModel); var eps: Float = 1e-6
        enc.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&eps, length: MemoryLayout<Float>.size, index: 4)
        enc.dispatchThreads(MTLSize(width: Int(r), height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, Int(r)), height: 1, depth: 1))
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
